"""FastAPI service for map-matching reserved trips or supplied GPS payloads."""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, model_validator

if __package__:
    from .data_utils import discover_trip_dirs, load_trip
    from .decoder import ViterbiDecoder
    from .features import FEATURE_COLUMNS, extract_trip_features
    from .train_baselines import positive_class_probabilities
else:
    from data_utils import discover_trip_dirs, load_trip
    from decoder import ViterbiDecoder
    from features import FEATURE_COLUMNS, extract_trip_features
    from train_baselines import positive_class_probabilities

LOGGER = logging.getLogger("pathsnap.api")
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MAX_OBSERVATIONS = 10_000
MAX_NETWORK_ROWS = 100_000


class GPSObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    longitude: float = Field(ge=-180.0, le=180.0)
    latitude: float = Field(ge=-90.0, le=90.0)
    sample_time: float | None = None


class RoadNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    longitude: float = Field(ge=-180.0, le=180.0)
    latitude: float = Field(ge=-90.0, le=90.0)
    node_id: int | None = None


class RoadArc(BaseModel):
    model_config = ConfigDict(extra="forbid")

    from_node: int
    to_node: int
    arc_id: int | None = None


class PredictRouteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trip_id: str | None = Field(default=None, min_length=1, max_length=128)
    observations: list[GPSObservation] | None = Field(
        default=None, min_length=1, max_length=MAX_OBSERVATIONS
    )
    nodes: list[RoadNode] | None = Field(
        default=None, min_length=1, max_length=MAX_NETWORK_ROWS
    )
    arcs: list[RoadArc] | None = Field(
        default=None, min_length=1, max_length=MAX_NETWORK_ROWS
    )

    @model_validator(mode="after")
    def validate_request_shape(self) -> PredictRouteRequest:
        raw_fields = (self.observations, self.nodes, self.arcs)
        if self.trip_id is not None:
            if any(value is not None for value in raw_fields):
                raise ValueError("Provide either trip_id or raw trip data, not both.")
            return self
        if any(value is None for value in raw_fields):
            raise ValueError(
                "Raw requests require observations, nodes, and arcs."
            )
        node_ids = [node.node_id for node in self.nodes or []]
        if any(node_id is None for node_id in node_ids) and not all(
            node_id is None for node_id in node_ids
        ):
            raise ValueError("Raw node_id values must be either all present or all omitted.")
        arc_ids = [arc.arc_id for arc in self.arcs or []]
        if any(arc_id is None for arc_id in arc_ids) and not all(
            arc_id is None for arc_id in arc_ids
        ):
            raise ValueError("Raw arc_id values must be either all present or all omitted.")
        explicit_node_ids = [node_id for node_id in node_ids if node_id is not None]
        explicit_arc_ids = [arc_id for arc_id in arc_ids if arc_id is not None]
        if len(set(explicit_node_ids)) != len(explicit_node_ids):
            raise ValueError("Raw node_id values must be unique.")
        if len(set(explicit_arc_ids)) != len(explicit_arc_ids):
            raise ValueError("Raw arc_id values must be unique.")
        return self


class RoutePoint(BaseModel):
    observation_index: int
    longitude: float
    latitude: float
    sample_time: float
    arc_id: int | None
    geometry: list[list[float]] | None


class PredictRouteResponse(BaseModel):
    trip_id: str
    observation_count: int
    matched_observation_count: int
    observations: list[RoutePoint]


class RoutePredictor:
    def __init__(
        self, model_path: Path, decoder_config_path: Path, data_dir: Path
    ) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Serving classifier not found at {model_path}. "
                "Run src/prepare_artifacts.py with a tuned classifier first."
            )
        if not decoder_config_path.is_file():
            raise FileNotFoundError(
                f"HMM decoder configuration not found: {decoder_config_path}"
            )
        if not data_dir.is_dir():
            raise NotADirectoryError(f"Trip data directory not found: {data_dir}")

        self.model = joblib.load(model_path)
        if not hasattr(self.model, "classes_") or 1 not in self.model.classes_:
            raise ValueError(
                f"Classifier at {model_path} is not fitted with positive class 1."
            )
        with decoder_config_path.open(encoding="utf-8") as config_file:
            self.decoder_config = json.load(config_file)
        self.data_dir = data_dir
        self.trip_dirs = {
            trip_dir.name: trip_dir
            for trip_dir in discover_trip_dirs(data_dir)
            if all(
                (trip_dir / f"{trip_dir.name}{suffix}").is_file()
                for suffix in (".track", ".nodes", ".arcs")
            )
        }

    def predict(
        self, request: PredictRouteRequest
    ) -> PredictRouteResponse:
        if request.trip_id is not None:
            trip_dir = self.trip_dirs.get(request.trip_id)
            if trip_dir is None:
                raise KeyError(request.trip_id)
            trip_id = request.trip_id
            trip = load_trip(trip_dir, include_route=False)
        else:
            trip_id = "raw"
            trip = self._raw_trip_data(request)

        track, nodes, arcs = trip["track"], trip["nodes"], trip["arcs"]
        if track.empty:
            raise ValueError("The trip has no valid GPS observations.")
        if nodes.empty or arcs.empty:
            raise ValueError("The trip road network has no valid nodes or arcs.")

        candidates = extract_trip_features(
            Path(trip_id),
            "student_test",
            trip_data=trip,
        )
        if candidates.empty:
            raise ValueError("No road candidates could be generated for this trip.")
        candidates["classifier_probability"] = positive_class_probabilities(
            self.model, candidates[FEATURE_COLUMNS]
        )
        decoder = ViterbiDecoder(nodes, arcs, **self.decoder_config)
        predicted_arc_ids = decoder.decode(track, candidates)

        points: list[RoutePoint] = []
        for index, (_, observation) in enumerate(track.reset_index(drop=True).iterrows()):
            arc_id = predicted_arc_ids[index]
            geometry: list[list[float]] | None = None
            if arc_id is not None and arc_id in decoder.arcs:
                _, _, _, start, end = decoder.arcs[arc_id]
                geometry = [
                    [float(start[0]), float(start[1])],
                    [float(end[0]), float(end[1])],
                ]
            points.append(
                RoutePoint(
                    observation_index=index,
                    longitude=float(observation["longitude"]),
                    latitude=float(observation["latitude"]),
                    sample_time=float(observation["sample_time"]),
                    arc_id=arc_id,
                    geometry=geometry,
                )
            )
        matched_count = sum(point.arc_id is not None for point in points)
        return PredictRouteResponse(
            trip_id=trip_id,
            observation_count=len(points),
            matched_observation_count=matched_count,
            observations=points,
        )

    @staticmethod
    def _raw_trip_data(
        request: PredictRouteRequest,
    ) -> dict[str, pd.DataFrame]:
        observations = request.observations or []
        raw_nodes = request.nodes or []
        raw_arcs = request.arcs or []
        nodes = pd.DataFrame(
            {
                "longitude": [node.longitude for node in raw_nodes],
                "latitude": [node.latitude for node in raw_nodes],
            },
            index=[
                node.node_id if node.node_id is not None else index
                for index, node in enumerate(raw_nodes)
            ],
        )
        arcs = pd.DataFrame(
            {
                "from_node": [arc.from_node for arc in raw_arcs],
                "to_node": [arc.to_node for arc in raw_arcs],
                "arc_id": [
                    arc.arc_id if arc.arc_id is not None else index
                    for index, arc in enumerate(raw_arcs)
                ],
            }
        )
        if not arcs["from_node"].isin(nodes.index).all() or not arcs[
            "to_node"
        ].isin(nodes.index).all():
            raise ValueError("Every raw road arc must reference an existing node.")
        track = pd.DataFrame(
            {
                "longitude": [point.longitude for point in observations],
                "latitude": [point.latitude for point in observations],
                "sample_time": [
                    point.sample_time if point.sample_time is not None else index
                    for index, point in enumerate(observations)
                ],
            }
        )
        return {"track": track, "nodes": nodes, "arcs": arcs, "route": pd.DataFrame()}


@asynccontextmanager
async def lifespan(application: FastAPI):
    model_path = Path(
        os.environ.get(
            "PATHSNAP_MODEL_PATH",
            REPOSITORY_ROOT / "outputs/pathsnap_artifacts/model.joblib",
        )
    )
    decoder_config_path = Path(
        os.environ.get(
            "PATHSNAP_DECODER_CONFIG",
            REPOSITORY_ROOT / "config/hmm_decoder.json",
        )
    )
    data_dir = Path(
        os.environ.get(
            "PATHSNAP_DATA_DIR",
            REPOSITORY_ROOT / "student_test",
        )
    )
    application.state.predictor = RoutePredictor(
        model_path, decoder_config_path, data_dir
    )
    LOGGER.info(
        "PathSnap API ready: classifier=%s, trips=%d, data=%s",
        model_path,
        len(application.state.predictor.trip_dirs),
        data_dir,
    )
    yield


app = FastAPI(
    title="PathSnap Route API",
    description="Classify road candidates and decode GPS trips with HMM/Viterbi.",
    lifespan=lifespan,
)


@app.middleware("http")
async def log_request_metrics(request: Request, call_next):
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        LOGGER.exception(
            "Request failed: %s %s (%.1f ms)",
            request.method,
            request.url.path,
            (time.perf_counter() - started) * 1000,
        )
        raise
    LOGGER.info(
        "Request %s %s -> %d (%.1f ms)",
        request.method,
        request.url.path,
        response.status_code,
        (time.perf_counter() - started) * 1000,
    )
    return response


@app.get("/health")
def health(request: Request) -> dict[str, str]:
    return {"status": "ok", "service": "PathSnap Route API"}


@app.get("/trips", response_model=list[str])
def list_reserved_trips(request: Request) -> list[str]:
    return sorted(request.app.state.predictor.trip_dirs)


@app.post("/predict_route", response_model=PredictRouteResponse)
def predict_route(
    payload: PredictRouteRequest, request: Request
) -> PredictRouteResponse:
    try:
        result = request.app.state.predictor.predict(payload)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="Reserved trip not found.") from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    LOGGER.info(
        "Prediction complete: trip=%s observations=%d matched=%d",
        result.trip_id,
        result.observation_count,
        result.matched_observation_count,
    )
    return result
