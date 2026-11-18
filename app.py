"""FastAPI inference service for PathSnap."""

from __future__ import annotations

import logging
import re
import tempfile
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.features import extract_trip_features
from src.inference import model_probabilities, viterbi_route

LOGGER = logging.getLogger("pathsnap.api")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

TRIP_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class RawTripPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    track: list[list[float]] = Field(..., min_length=1)
    nodes: list[list[float]] = Field(..., min_length=1)
    arcs: list[list[int]] = Field(..., min_length=1)


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trip_id: str | None = None
    trip: RawTripPayload | None = None

    def has_input(self) -> bool:
        return self.trip_id is not None or self.trip is not None


class PredictionResponse(BaseModel):
    trip_id: str
    predicted_sequence: list[int]
    candidate_count: int


class InferenceService:
    def __init__(self, project_root: Path | None = None) -> None:
        self.project_root = project_root or Path(__file__).resolve().parent
        self.output_dir = self.project_root / "outputs"
        self.model_name = self._select_model_name()
        self.model = self._load_model()
        self.preprocessor = self._load_optional("feature_scaler.joblib")
        self.decoder = self._load_optional("hmm_decoder.joblib")

    def _select_model_name(self) -> str:
        best_params_path = self.output_dir / "optimization_best_params.json"
        if best_params_path.is_file():
            try:
                import json

                with best_params_path.open("r", encoding="utf-8") as handle:
                    name = json.load(handle).get("model")
                if isinstance(name, str):
                    return name
            except (OSError, ValueError, json.JSONDecodeError):
                LOGGER.exception("Could not read optimization best parameters")

        comparison_path = self.output_dir / "model_comparison.csv"
        if comparison_path.is_file():
            try:
                comparison = pd.read_csv(comparison_path)
                rows = comparison.loc[
                    comparison["evaluation"].eq("train_cross_validation")
                ].dropna(subset=["macro_f1_score"])
                if not rows.empty:
                    return str(
                        rows.sort_values(
                            ["macro_f1_score", "model"],
                            ascending=[False, True],
                            kind="stable",
                        ).iloc[0]["model"]
                    )
            except (OSError, pd.errors.ParserError, ValueError):
                LOGGER.exception("Could not read baseline model comparison")
        return "logistic_regression"

    def _load_optional(self, filename: str) -> object | None:
        path = self.output_dir / filename
        if not path.is_file():
            return None
        try:
            return joblib.load(path)
        except (OSError, ValueError) as error:
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
                except (OSError, ValueError) as error:
                    raise RuntimeError(f"Model could not be loaded: {path.name}") from error
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

    def write_raw_trip(self, trip: RawTripPayload) -> tuple[Path, tempfile.TemporaryDirectory[str]]:
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
        return trip_dir, temporary

    def predict_trip(self, trip_dir: Path) -> tuple[list[int], int]:
        candidate_features = extract_trip_features(trip_dir, "student_test")
        if candidate_features.empty:
            raise HTTPException(
                status_code=422,
                detail="Trip data did not produce any candidate road arcs.",
            )

        probabilities = model_probabilities(self.model, candidate_features)
        arcs: dict[int, tuple[int, int]] = {}
        arcs_path = trip_dir / f"{trip_dir.name}.arcs"
        if arcs_path.is_file():
            arcs_frame = pd.read_csv(
                arcs_path, sep=r"\s+", header=None, names=["from_node", "to_node"]
            )
            arcs = {
                arc_id: (int(row.from_node), int(row.to_node))
                for arc_id, row in enumerate(
                    arcs_frame.itertuples(index=False), start=0
                )
            }

        predicted_sequence = viterbi_route(
            candidate_features,
            probabilities,
            arcs,
            decoder=self.decoder,
        )
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
    if not request.has_input():
        raise HTTPException(
            status_code=422,
            detail="Provide exactly one of trip_id or trip.",
        )
    if request.trip_id is not None and request.trip is not None:
        raise HTTPException(
            status_code=422,
            detail="Provide exactly one of trip_id or trip.",
        )

    service = get_service()
    temporary = None
    try:
        if request.trip_id is not None:
            trip_dir = service.resolve_trip(request.trip_id)
            response_trip_id = request.trip_id
        else:
            trip_dir, temporary = service.write_raw_trip(request.trip)
            response_trip_id = "raw"

        sequence, candidate_count = service.predict_trip(trip_dir)
        LOGGER.info(
            "Successful inference trip_id=%s candidates=%d predicted_arcs=%d",
            response_trip_id,
            candidate_count,
            len(sequence),
        )
        return PredictionResponse(
            trip_id=response_trip_id,
            predicted_sequence=sequence,
            candidate_count=candidate_count,
        )
    except HTTPException:
        raise
    except (OSError, ValueError, RuntimeError) as error:
        LOGGER.exception("Inference failed for trip_id=%s", request.trip_id or "raw")
        raise HTTPException(status_code=500, detail="Inference failed.") from error
    finally:
        if temporary is not None:
            temporary.cleanup()
