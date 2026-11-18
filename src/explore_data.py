"""Dataset inventory and interactive map generation for PathSnap."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import folium
import numpy as np
import pandas as pd

if __package__:
    from .data_utils import discover_trip_dirs, load_trip
else:
    from data_utils import discover_trip_dirs, load_trip

LOGGER = logging.getLogger(__name__)


def _interval_statistics(track: pd.DataFrame) -> dict[str, float]:
    if len(track) < 2:
        return {
            "sampling_interval_mean": np.nan,
            "sampling_interval_median": np.nan,
            "sampling_interval_min": np.nan,
            "sampling_interval_max": np.nan,
        }
    intervals = track["sample_time"].diff().dropna()
    intervals = intervals[np.isfinite(intervals)]
    if intervals.empty:
        return {
            "sampling_interval_mean": np.nan,
            "sampling_interval_median": np.nan,
            "sampling_interval_min": np.nan,
            "sampling_interval_max": np.nan,
        }
    return {
        "sampling_interval_mean": intervals.mean(),
        "sampling_interval_median": intervals.median(),
        "sampling_interval_min": intervals.min(),
        "sampling_interval_max": intervals.max(),
    }


def build_inventory(dataset_roots: dict[str, str | Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create per-trip file/observation inventory and per-dataset totals."""
    rows = []
    for dataset_name, root in dataset_roots.items():
        for trip_dir in discover_trip_dirs(root):
            trip = load_trip(trip_dir)
            row = {
                "dataset": dataset_name,
                "trip_id": trip_dir.name,
                "trip_path": str(trip_dir),
                "observation_count": len(trip["track"]),
                "node_count": len(trip["nodes"]),
                "arc_count": len(trip["arcs"]),
                "route_arc_count": len(trip["route"]),
            }
            row.update(_interval_statistics(trip["track"]))
            for suffix in (".track", ".nodes", ".arcs", ".route"):
                file_path = trip_dir / f"{trip_dir.name}{suffix}"
                row[f"{suffix[1:]}_size_bytes"] = (
                    file_path.stat().st_size if file_path.is_file() else 0
                )
            rows.append(row)

    inventory = pd.DataFrame(rows)
    if inventory.empty:
        summary = pd.DataFrame(
            columns=[
                "dataset",
                "total_trips",
                "total_observations",
                "mean_observations_per_trip",
            ]
        )
    else:
        summary = (
            inventory.groupby("dataset", as_index=False)
            .agg(
                total_trips=("trip_id", "nunique"),
                total_observations=("observation_count", "sum"),
                mean_observations_per_trip=("observation_count", "mean"),
                mean_sampling_interval=("sampling_interval_mean", "mean"),
                total_file_size_bytes=(
                    "track_size_bytes",
                    lambda values: int(
                        values.sum()
                        + inventory.loc[values.index, [
                            "nodes_size_bytes",
                            "arcs_size_bytes",
                            "route_size_bytes",
                        ]].to_numpy().sum()
                    ),
                ),
            )
        )
    return inventory, summary


def create_trip_map(
    trip_dir: str | Path,
    output_path: str | Path,
    bbox_buffer_degrees: float = 0.003,
) -> Path:
    """Save a Folium map with the GPS track and nearby road-network arcs."""
    trip_dir = Path(trip_dir)
    trip = load_trip(trip_dir)
    track, nodes, arcs = trip["track"], trip["nodes"], trip["arcs"]
    if track.empty:
        raise ValueError(f"Cannot map trip without GPS observations: {trip_dir}")

    center = [float(track["latitude"].mean()), float(track["longitude"].mean())]
    map_view = folium.Map(location=center, zoom_start=14, control_scale=True)
    if not nodes.empty and not arcs.empty:
        arc_pairs = arcs[["from_node", "to_node"]].to_numpy(dtype=np.int64)
        node_coordinates = {
            int(node_id): coordinate
            for node_id, coordinate in zip(
                nodes.index,
                nodes[["longitude", "latitude"]].to_numpy(dtype=float),
            )
        }
        valid = (
            np.isin(arc_pairs[:, 0], list(node_coordinates))
            & np.isin(arc_pairs[:, 1], list(node_coordinates))
        )
        arc_pairs = arc_pairs[valid]
        min_lon = track["longitude"].min() - bbox_buffer_degrees
        max_lon = track["longitude"].max() + bbox_buffer_degrees
        min_lat = track["latitude"].min() - bbox_buffer_degrees
        max_lat = track["latitude"].max() + bbox_buffer_degrees
        nearby = []
        for from_node, to_node in arc_pairs:
            start = node_coordinates[from_node]
            end = node_coordinates[to_node]
            if (
                max(start[0], end[0]) >= min_lon
                and min(start[0], end[0]) <= max_lon
                and max(start[1], end[1]) >= min_lat
                and min(start[1], end[1]) <= max_lat
            ):
                nearby.append(
                    {
                        "type": "Feature",
                        "properties": {},
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [start.tolist(), end.tolist()],
                        },
                    }
                )
        if nearby:
            folium.GeoJson(
                {"type": "FeatureCollection", "features": nearby},
                name="Nearby road arcs",
                style_function=lambda _feature: {
                    "color": "#64748b",
                    "weight": 1,
                    "opacity": 0.55,
                },
            ).add_to(map_view)

    track_points = track[["latitude", "longitude"]].to_numpy().tolist()
    folium.PolyLine(
        track_points,
        color="#e11d48",
        weight=4,
        opacity=0.9,
        tooltip="Raw GPS track",
        name="GPS track",
    ).add_to(map_view)
    folium.CircleMarker(
        track_points[0], radius=5, color="#16a34a", tooltip="Start"
    ).add_to(map_view)
    folium.CircleMarker(
        track_points[-1], radius=5, color="#7c3aed", tooltip="End"
    ).add_to(map_view)
    folium.LayerControl().add_to(map_view)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    map_view.save(output_path)
    return output_path


def run_eda(
    train_dir: str | Path = "Train",
    test_dir: str | Path = "student_test",
    output_dir: str | Path = "outputs",
    map_output_dir: str | Path = "eda_outputs",
    sample_trip: str | None = None,
) -> tuple[Path, Path, Path]:
    output_dir, map_output_dir = Path(output_dir), Path(map_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    map_output_dir.mkdir(parents=True, exist_ok=True)
    inventory, summary = build_inventory(
        {"train": train_dir, "student_test": test_dir}
    )
    inventory_path = output_dir / "dataset_inventory.csv"
    summary_path = output_dir / "dataset_summary.csv"
    inventory.to_csv(inventory_path, index=False)
    summary.to_csv(summary_path, index=False)

    train_trip_dirs = discover_trip_dirs(train_dir)
    test_trip_dirs = discover_trip_dirs(test_dir)
    candidate_dirs = train_trip_dirs or test_trip_dirs
    if not candidate_dirs:
        raise FileNotFoundError("No trip folders containing PathSnap files were found.")
    if sample_trip:
        matches = [
            path
            for path in train_trip_dirs + test_trip_dirs
            if path.name == sample_trip
        ]
        if not matches:
            raise ValueError(f"Sample trip {sample_trip!r} was not found.")
        map_trip = matches[0]
    else:
        map_trip = candidate_dirs[0]
    map_path = create_trip_map(
        map_trip, map_output_dir / f"{map_trip.name}_map.html"
    )
    LOGGER.info("Wrote %s, %s, and %s", inventory_path, summary_path, map_path)
    return inventory_path, summary_path, map_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", default="Train")
    parser.add_argument("--test-dir", default="student_test")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--map-output-dir", default="eda_outputs")
    parser.add_argument("--sample-trip", help="Trip folder name to use for the map.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run_eda(
        args.train_dir,
        args.test_dir,
        args.output_dir,
        args.map_output_dir,
        args.sample_trip,
    )


if __name__ == "__main__":
    main()
