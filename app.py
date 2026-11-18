"""FastAPI inference service for PathSnap."""

from __future__ import annotations

import json
import logging
import math
import re
import tempfile
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.data_utils import load_trip
from src.decoder import ViterbiDecoder
from src.features import extract_trip_features
from src.inference import model_probabilities, validate_feature_preprocessor
from src.train_baselines import build_models

LOGGER = logging.getLogger("pathsnap.api")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

TRIP_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
SUPPORTED_MODELS = frozenset(build_models())


class RawTripPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    track: list[list[float]] = Field(..., min_length=1, max_length=100_000)
    nodes: list[list[float]] = Field(..., min_length=1, max_length=250_000)
    arcs: list[list[int]] = Field(..., min_length=1, max_length=500_000)

    @model_validator(mode="after")
    def validate_trip_geometry(self) -> RawTripPayload:
        track_widths = {len(row) for row in self.track}
        if not track_widths.issubset({2, 3}) or len(track_widths) != 1:
            raise ValueError("Each track row must contain longitude, latitude, and optional time.")
        if any(len(row) != 2 for row in self.nodes):
            raise ValueError("Each node row must contain longitude and latitude.")
        if any(len(row) != 2 for row in self.arcs):
            raise ValueError("Each arc row must contain start and end node indices.")

        for label, coordinates in (
            ("track", self.track),
            ("nodes", self.nodes),
        ):
            for row in coordinates:
                longitude, latitude = row[:2]
                if (
                    not math.isfinite(longitude)
                    or not math.isfinite(latitude)
                    or not -180 <= longitude <= 180
                    or not -90 <= latitude <= 90
                    or any(not math.isfinite(value) for value in row[2:])
                ):
                    raise ValueError(f"{label} coordinates must be finite and in range.")

        node_count = len(self.nodes)
        if any(
            start < 0 or end < 0 or start >= node_count or end >= node_count
            for start, end in self.arcs
        ):
            raise ValueError("Arc endpoints must reference nodes in the payload.")
        return self


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trip_id: str | None = None
    trip: RawTripPayload | None = None

    @model_validator(mode="after")
    def validate_single_input(self) -> PredictionRequest:
        if (self.trip_id is None) == (self.trip is None):
            raise ValueError("Provide exactly one of trip_id or trip.")
        return self


class PredictionResponse(BaseModel):
    trip_id: str
    predicted_sequence: list[int | None]
    candidate_count: int


class InferenceService:
    def __init__(self, project_root: Path | None = None) -> None:
        self.project_root = project_root or Path(__file__).resolve().parent
        self.output_dir = self.project_root / "outputs"
        self.model_name = self._select_model_name()
        self.model = self._load_model()
        self.preprocessor = self._load_optional("feature_scaler.joblib")
        if self.preprocessor is None:
            raise RuntimeError(
                f"Feature preprocessor not found: "
                f"{self.output_dir / 'feature_scaler.joblib'}"
            )
        validate_feature_preprocessor(self.preprocessor)

    def _select_model_name(self) -> str:
        best_params_path = self.output_dir / "optimization_best_params.json"
        if best_params_path.is_file():
            try:
                with best_params_path.open("r", encoding="utf-8") as handle:
                    configuration = json.load(handle)
                name = (
                    configuration.get("model")
                    if isinstance(configuration, dict)
                    else None
                )
                if isinstance(name, str) and name in SUPPORTED_MODELS:
                    return name
                LOGGER.warning(
                    "Ignoring unsupported model name in %s", best_params_path
                )
            except (OSError, ValueError, json.JSONDecodeError):
                LOGGER.exception("Could not read optimization best parameters")

        comparison_path = self.output_dir / "model_comparison.csv"
        if comparison_path.is_file():
            try:
                comparison = pd.read_csv(comparison_path)
                rows = comparison.loc[
                    comparison["evaluation"].eq("train_cross_validation")
                ].dropna(subset=["macro_f1_score"])
                rows = rows.loc[rows["model"].isin(SUPPORTED_MODELS)]
                if not rows.empty:
                    return str(
                        rows.sort_values(
                            ["macro_f1_score", "model"],
                            ascending=[False, True],
                            kind="stable",
                        ).iloc[0]["model"]
                    )
            except (OSError, KeyError, pd.errors.ParserError, ValueError):
                LOGGER.exception("Could not read baseline model comparison")
        return "logistic_regression"

    def _load_optional(self, filename: str) -> object | None:
        path = self.output_dir / filename
        if not path.is_file():
            return None
        try:
            return joblib.load(path)
        except (OSError, ValueError, EOFError, ImportError) as error:
            LOGGER.exception("Could not load optional artifact %s: %s", path, error)
            raise RuntimeError(f"Artifact could not be loaded: {filename}") from error

    def _load_model(self) -> object:
        candidates = [
            self.output_dir / f"{self.model_name}_tuned.joblib",
            self.output_dir / f"{self.model_name}.joblib",
        ]
        for path in candidates:
            if path.is_file():
                try:
                    model = joblib.load(path)
                except (OSError, ValueError, EOFError, ImportError) as error:
                    raise RuntimeError(
                        f"Model could not be loaded: {path.name}"
                    ) from error
                LOGGER.info("Loaded model artifact %s", path)
                return model
        raise RuntimeError(
            "No trained model artifact found. Run the feature and training pipeline "
            f"first (expected {self.output_dir / (self.model_name + '.joblib')})."
        )

    def resolve_trip(self, trip_id: str) -> Path:
        if not TRIP_ID_PATTERN.fullmatch(trip_id):
            raise HTTPException(status_code=422, detail="Invalid trip_id.")
        for dataset_dir in ("student_test", "Train"):
            trip_dir = self.project_root / dataset_dir / trip_id
            if trip_dir.is_dir():
                return trip_dir
        raise HTTPException(status_code=404, detail=f"Trip {trip_id!r} not found.")

    def write_raw_trip(
        self, trip: RawTripPayload
    ) -> tuple[Path, tempfile.TemporaryDirectory[str]]:
        temporary = tempfile.TemporaryDirectory(prefix="pathsnap-trip-")
        trip_dir = Path(temporary.name) / "raw_trip"
        trip_dir.mkdir()
        trip_id = "raw_trip"

        def write_rows(filename: str, rows: list[list[object]]) -> None:
            if not rows:
                return
            pd.DataFrame(rows).to_csv(
                trip_dir / f"{trip_id}.{filename}",
                sep=" ",
                header=False,
                index=False,
            )

        write_rows("track", trip.track)
        write_rows("nodes", trip.nodes)
        write_rows("arcs", trip.arcs)
        (trip_dir / f"{trip_id}.route").touch()
        return trip_dir, temporary

    def predict_trip(self, trip_dir: Path) -> tuple[list[int | None], int]:
        candidate_features = extract_trip_features(trip_dir, "student_test")
        if candidate_features.empty:
            raise HTTPException(
                status_code=422,
                detail="Trip data did not produce any candidate road arcs.",
            )

        probabilities = model_probabilities(
            self.model,
            candidate_features,
            preprocessor=self.preprocessor,
        )
        trip = load_trip(trip_dir)
        candidates = candidate_features[
            [
                "observation_index",
                "arc_id",
                "distance_to_segment_m",
                "heading_alignment",
            ]
        ].copy()
        candidates["classifier_probability"] = probabilities
        decoder = ViterbiDecoder(trip["nodes"], trip["arcs"])
        predicted_sequence = decoder.decode(trip["track"], candidates)
        return predicted_sequence, len(candidate_features)


SERVICE: InferenceService | None = None


def get_service() -> InferenceService:
    global SERVICE
    if SERVICE is None:
        SERVICE = InferenceService()
    return SERVICE


app = FastAPI(
    title="PathSnap Route Inference",
    version="1.0.0",
    description="Secure route-candidate inference without route labels.",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict_route", response_model=PredictionResponse)
def predict_route(request: PredictionRequest) -> PredictionResponse:
    temporary = None
    try:
        service = get_service()
        if request.trip_id is not None:
            trip_dir = service.resolve_trip(request.trip_id)
            response_trip_id = request.trip_id
        elif request.trip is not None:
            trip_dir, temporary = service.write_raw_trip(request.trip)
            response_trip_id = "raw"
        else:
            raise HTTPException(
                status_code=422,
                detail="Provide exactly one of trip_id or trip.",
            )

        sequence, candidate_count = service.predict_trip(trip_dir)
        response = PredictionResponse(
            trip_id=response_trip_id,
            predicted_sequence=sequence,
            candidate_count=candidate_count,
        )
        LOGGER.info(
            "Successful inference request trip_id=%s candidates=%d",
            response_trip_id,
            candidate_count,
        )
        LOGGER.info("Inference response: %s", response.model_dump_json())
        return response
    except HTTPException:
        raise
    except (OSError, ValueError, RuntimeError) as error:
        LOGGER.exception("Inference failed for trip_id=%s", request.trip_id or "raw")
        raise HTTPException(status_code=500, detail="Inference failed.") from error
    finally:
        if temporary is not None:
            temporary.cleanup()
