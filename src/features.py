"""Candidate-road matching and numerical feature extraction for PathSnap."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler

if __package__:
    from .data_utils import discover_trip_dirs, load_trip
else:
    from data_utils import discover_trip_dirs, load_trip

LOGGER = logging.getLogger(__name__)
EARTH_RADIUS_METERS = 6_371_000.0
DEFAULT_MATCH_RADIUS_METERS = 100.0
FEATURE_COLUMNS = [
    "distance_to_segment_m",
    "distance_to_arc_center_m",
    "heading_alignment",
    "speed_mps",
    "acceleration_mps2",
]
MAX_CANDIDATES_PER_OBSERVATION = 10


def _project_coordinates(
    longitude: np.ndarray,
    latitude: np.ndarray,
    reference_longitude: float,
    reference_latitude: float,
) -> np.ndarray:
    longitude_radians = np.radians(longitude - reference_longitude)
    latitude_radians = np.radians(latitude - reference_latitude)
    x = (
        EARTH_RADIUS_METERS
        * longitude_radians
        * np.cos(np.radians(reference_latitude))
    )
    y = EARTH_RADIUS_METERS * latitude_radians
    return np.column_stack((x, y))


def _bearing_degrees(start: np.ndarray, end: np.ndarray) -> np.ndarray:
    lon1, lat1 = np.radians(start[:, 0]), np.radians(start[:, 1])
    lon2, lat2 = np.radians(end[:, 0]), np.radians(end[:, 1])
    delta_lon = lon2 - lon1
    y = np.sin(delta_lon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(delta_lon)
    return (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0


def _point_segment_distances(point: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    vectors = ends - starts
    lengths_squared = np.einsum("ij,ij->i", vectors, vectors)
    fractions = np.divide(
        np.einsum("ij,ij->i", point - starts, vectors),
        lengths_squared,
        out=np.zeros_like(lengths_squared),
        where=lengths_squared > 0,
    )
    fractions = np.clip(fractions, 0.0, 1.0)
    projections = starts + fractions[:, None] * vectors
    return np.linalg.norm(projections - point, axis=1)


def _track_motion(track: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coordinates = track[["longitude", "latitude"]].to_numpy(dtype=float)
    times = track["sample_time"].to_numpy(dtype=float)
    step_distances = np.zeros(len(track), dtype=float)
    headings = np.full(len(track), np.nan, dtype=float)
    if len(track) > 1:
        reference_lon, reference_lat = coordinates[0]
        projected = _project_coordinates(
            coordinates[:, 0], coordinates[:, 1], reference_lon, reference_lat
        )
        vectors = np.diff(projected, axis=0)
        step_distances[1:] = np.linalg.norm(vectors, axis=1)
        headings[1:] = _bearing_degrees(coordinates[:-1], coordinates[1:])
    intervals = np.diff(times, prepend=np.nan)
    speeds = np.divide(
        step_distances,
        intervals,
        out=np.full_like(step_distances, np.nan),
        where=np.isfinite(intervals) & (intervals > 0),
    )
    accelerations = np.full(len(track), np.nan, dtype=float)
    if len(track) > 1:
        delta_time = np.diff(times)
        speed_change = np.diff(speeds)
        valid = np.isfinite(delta_time) & (delta_time > 0) & np.isfinite(speed_change)
        acceleration = np.full(len(delta_time), np.nan, dtype=float)
        acceleration[valid] = speed_change[valid] / delta_time[valid]
        accelerations[1:] = acceleration
    return headings, speeds, accelerations


def extract_trip_features(
    trip_dir: str | Path,
    dataset: str,
    match_radius_meters: float = DEFAULT_MATCH_RADIUS_METERS,
    *,
    trip_data: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Create one feature row per observation/candidate road arc."""
    trip_dir = Path(trip_dir)
    trip = (
        trip_data
        if trip_data is not None
        else load_trip(trip_dir, include_route=dataset == "train")
    )
    track, nodes, arcs = trip["track"], trip["nodes"], trip["arcs"]
    if track.empty:
        LOGGER.warning("Skipping trip %s: no valid GPS observations", trip_dir)
        return pd.DataFrame()
    if nodes.empty or arcs.empty:
        LOGGER.warning("Skipping trip %s: nodes or arcs are missing", trip_dir)
        return pd.DataFrame()

    arc_pairs = arcs[["from_node", "to_node"]].to_numpy(dtype=np.int64)
    node_coordinates = {
        int(node_id): coordinate
        for node_id, coordinate in zip(
            nodes.index,
            nodes[["longitude", "latitude"]].to_numpy(dtype=float),
        )
    }
    track_coordinates = track[["longitude", "latitude"]].to_numpy(dtype=float)
    reference_lon = float(np.mean(track_coordinates[:, 0]))
    reference_lat = float(np.mean(track_coordinates[:, 1]))
    valid_arc_ids = arcs["arc_id"].to_numpy(dtype=np.int64)
    valid_arc_mask = np.fromiter(
        (
            from_node in node_coordinates and to_node in node_coordinates
            for from_node, to_node in arc_pairs
        ),
        dtype=bool,
        count=len(arc_pairs),
    )
    if (~valid_arc_mask).any():
        LOGGER.warning(
            "Ignoring %d arc(s) with invalid node references in %s",
            (~valid_arc_mask).sum(),
            trip_dir,
        )
    valid_arc_ids = valid_arc_ids[valid_arc_mask]
    arc_pairs = arc_pairs[valid_arc_mask]
    if not len(arc_pairs):
        LOGGER.warning("Skipping trip %s: no arcs have valid endpoints", trip_dir)
        return pd.DataFrame()
    start_geographic = np.asarray(
        [node_coordinates[from_node] for from_node, _ in arc_pairs]
    )
    end_geographic = np.asarray(
        [node_coordinates[to_node] for _, to_node in arc_pairs]
    )
    starts = _project_coordinates(
        start_geographic[:, 0],
        start_geographic[:, 1],
        reference_lon,
        reference_lat,
    )
    ends = _project_coordinates(
        end_geographic[:, 0],
        end_geographic[:, 1],
        reference_lon,
        reference_lat,
    )
    centers = (starts + ends) / 2.0
    half_lengths = np.linalg.norm(ends - starts, axis=1) / 2.0
    center_tree = cKDTree(centers)
    short_arc_indices = np.flatnonzero(half_lengths <= match_radius_meters)
    long_arc_indices = np.flatnonzero(half_lengths > match_radius_meters)
    short_arc_tree = (
        cKDTree(centers[short_arc_indices]) if len(short_arc_indices) else None
    )
    long_arc_tree = cKDTree(centers[long_arc_indices]) if len(long_arc_indices) else None
    long_arc_query_radius = (
        match_radius_meters + float(half_lengths[long_arc_indices].max())
        if len(long_arc_indices)
        else 0.0
    )
    arc_bearings = _bearing_degrees(start_geographic, end_geographic)
    projected_track = _project_coordinates(
        track_coordinates[:, 0],
        track_coordinates[:, 1],
        reference_lon,
        reference_lat,
    )
    headings, speeds, accelerations = _track_motion(track)
    route_arc_ids = (
        set(trip["route"]["arc_id"].to_numpy(dtype=np.int64))
        if dataset == "train"
        else set()
    )
    records: list[dict[str, float | int | str | bool]] = []

    for observation_index, point in enumerate(projected_track):
        candidate_indices = []
        if short_arc_tree is not None:
            short_candidates = short_arc_tree.query_ball_point(
                point, 2.0 * match_radius_meters
            )
            candidate_indices.extend(short_arc_indices[short_candidates])
        if long_arc_tree is not None:
            long_candidates = long_arc_tree.query_ball_point(
                point, long_arc_query_radius
            )
            candidate_indices.extend(long_arc_indices[long_candidates])
        if not candidate_indices:
            _, nearest = center_tree.query(point, k=1)
            candidate_indices = [int(nearest)]
        candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
        center_distances = np.linalg.norm(centers[candidate_indices] - point, axis=1)
        segment_distances = _point_segment_distances(
            point, starts[candidate_indices], ends[candidate_indices]
        )
        close = segment_distances <= match_radius_meters
        if close.any():
            candidate_indices = candidate_indices[close]
            center_distances = center_distances[close]
            segment_distances = segment_distances[close]
            nearest_positions = np.argsort(segment_distances)[
                :MAX_CANDIDATES_PER_OBSERVATION
            ]
            candidate_indices = candidate_indices[nearest_positions]
            center_distances = center_distances[nearest_positions]
            segment_distances = segment_distances[nearest_positions]
        else:
            nearest_positions = np.argsort(segment_distances)[:5]
            candidate_indices = candidate_indices[nearest_positions]
            center_distances = center_distances[nearest_positions]
            segment_distances = segment_distances[nearest_positions]

        for arc_position, center_distance, segment_distance in zip(
            candidate_indices, center_distances, segment_distances
        ):
            arc_bearing = arc_bearings[arc_position]
            if np.isfinite(headings[observation_index]):
                angular_difference = np.radians(
                    headings[observation_index] - arc_bearing
                )
                heading_alignment = float(np.cos(angular_difference))
            else:
                heading_alignment = 0.0
            arc_id = int(valid_arc_ids[arc_position])
            records.append(
                {
                    "dataset": dataset,
                    "trip_id": trip_dir.name,
                    "observation_index": observation_index,
                    "sample_time": float(track.iloc[observation_index]["sample_time"]),
                    "arc_id": arc_id,
                    "distance_to_segment_m": float(segment_distance),
                    "distance_to_arc_center_m": float(center_distance),
                    "heading_alignment": heading_alignment,
                    "speed_mps": float(speeds[observation_index])
                    if np.isfinite(speeds[observation_index])
                    else 0.0,
                    "acceleration_mps2": float(accelerations[observation_index])
                    if np.isfinite(accelerations[observation_index])
                    else 0.0,
                    "is_route_arc": (
                        float(arc_id in route_arc_ids) if dataset == "train" else np.nan
                    ),
                }
            )
    return pd.DataFrame.from_records(records)


def run_feature_pipeline(
    train_dir: str | Path = "Train",
    test_dir: str | Path = "student_test",
    output_dir: str | Path = "outputs",
    match_radius_meters: float = DEFAULT_MATCH_RADIUS_METERS,
) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_frames = [
        extract_trip_features(trip_dir, "train", match_radius_meters)
        for trip_dir in discover_trip_dirs(train_dir)
    ]
    train_frames = [frame for frame in train_frames if not frame.empty]
    if not train_frames:
        raise ValueError("No training candidate features could be extracted.")
    test_frames = [
        extract_trip_features(trip_dir, "student_test", match_radius_meters)
        for trip_dir in discover_trip_dirs(test_dir)
    ]
    frames = train_frames + [frame for frame in test_frames if not frame.empty]
    features = pd.concat(frames, ignore_index=True)

    train_mask = features["dataset"].eq("train")
    scaler = StandardScaler()
    scaler.fit(features.loc[train_mask, FEATURE_COLUMNS])
    scaled_values = scaler.transform(features[FEATURE_COLUMNS])
    scaled_columns = [f"{column}_scaled" for column in FEATURE_COLUMNS]
    features[scaled_columns] = scaled_values

    features_path = output_dir / "candidate_features.csv"
    scaler_path = output_dir / "feature_scaler.joblib"
    features.to_csv(features_path, index=False)
    joblib.dump(
        {
            "scaler": scaler,
            "feature_columns": FEATURE_COLUMNS,
            "fit_partition": "train",
            "match_radius_meters": match_radius_meters,
        },
        scaler_path,
    )
    LOGGER.info(
        "Wrote %d candidate rows to %s; scaler fitted on %d training rows",
        len(features),
        features_path,
        int(train_mask.sum()),
    )
    return features_path, scaler_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", default="Train")
    parser.add_argument("--test-dir", default="student_test")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument(
        "--match-radius-meters", type=float, default=DEFAULT_MATCH_RADIUS_METERS
    )
    args = parser.parse_args()
    if args.match_radius_meters <= 0:
        parser.error("--match-radius-meters must be positive")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run_feature_pipeline(
        args.train_dir,
        args.test_dir,
        args.output_dir,
        args.match_radius_meters,
    )


if __name__ == "__main__":
    main()
