"""Generate interactive Folium maps for PathSnap trips and route predictions."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

import folium
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

if __package__:
    from .data_utils import discover_trip_dirs, load_trip, read_numeric_file
else:
    from data_utils import discover_trip_dirs, load_trip, read_numeric_file

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _arc_geometries(
    trip: dict[str, pd.DataFrame],
    arc_ids: Sequence[int] | None = None,
) -> list[list[list[float]]]:
    """Return valid arc geometries as [latitude, longitude] coordinate pairs."""
    nodes, arcs = trip["nodes"], trip["arcs"]
    if nodes.empty or arcs.empty:
        return []

    node_coordinates = {
        int(node_id): coordinates
        for node_id, coordinates in zip(
            nodes.index,
            nodes[["latitude", "longitude"]].to_numpy(dtype=float),
        )
    }
    if arc_ids is None:
        selected_arcs = arcs
    else:
        selected_arcs = (
            arcs.set_index("arc_id", drop=False)
            .reindex([int(value) for value in arc_ids])
            .dropna(subset=["from_node", "to_node"])
        )

    geometries = []
    for arc in selected_arcs.itertuples():
        start = node_coordinates.get(int(arc.from_node))
        end = node_coordinates.get(int(arc.to_node))
        if start is not None and end is not None:
            geometries.append([start.tolist(), end.tolist()])
    return geometries


def _relevant_arc_ids(
    trip: dict[str, pd.DataFrame],
    buffer_meters: float,
) -> list[int]:
    """Find arcs incident to road nodes within a buffer of GPS observations."""
    track, nodes, arcs = trip["track"], trip["nodes"], trip["arcs"]
    if track.empty or nodes.empty or arcs.empty:
        return []

    longitude_scale = 111_320 * np.cos(
        np.deg2rad(float(track["latitude"].mean()))
    )
    latitude_scale = 111_320
    node_coordinates = nodes[["longitude", "latitude"]].to_numpy(dtype=float)
    track_coordinates = track[["longitude", "latitude"]].to_numpy(dtype=float)
    node_xy = node_coordinates * [longitude_scale, latitude_scale]
    track_xy = track_coordinates * [longitude_scale, latitude_scale]
    neighbors = cKDTree(node_xy).query_ball_point(track_xy, r=buffer_meters)
    nearby_node_ids = {
        int(nodes.index[position])
        for positions in neighbors
        for position in positions
    }
    valid_arc_endpoints = arcs["from_node"].isin(nodes.index) & arcs[
        "to_node"
    ].isin(nodes.index)
    relevant_arcs = arcs.loc[
        valid_arc_endpoints
        & (
            arcs["from_node"].isin(nearby_node_ids)
            | arcs["to_node"].isin(nearby_node_ids)
        )
    ]
    return relevant_arcs["arc_id"].astype(int).tolist()


def load_predicted_arc_ids(
    predictions_path: str | Path,
    trip_id: str,
    model: str | None = None,
) -> list[int]:
    """Load a trip's predicted arc IDs from classifier or decoded-route CSV data.

    Classifier output may use the project's ``selected_top1`` and ``model``
    columns. Decoded-route output may instead provide ``predicted_arc_id``,
    ``decoded_arc_id``, ``route_arc_id``, or ``arc_id``. If multiple classifier
    models are present and ``model`` is omitted, the first model in the file is
    selected.
    """
    predictions_path = Path(predictions_path)
    if not predictions_path.is_file():
        raise FileNotFoundError(f"Prediction file not found: {predictions_path}")
    if predictions_path.suffix.lower() == ".route":
        route = read_numeric_file(predictions_path, 1, max_columns=1)
        if route.empty:
            return []
        values = pd.to_numeric(route.iloc[:, 0], errors="coerce").dropna()
        if not np.equal(values, np.floor(values)).all():
            raise ValueError(f"Route arc IDs must be integers: {predictions_path}")
        return values.astype("int64").tolist()

    predictions = pd.read_csv(predictions_path)
    if "trip_id" in predictions.columns:
        trip_ids = predictions["trip_id"].astype(str)
        matches = trip_ids.eq(str(trip_id)) | trip_ids.str.lstrip("0").eq(
            str(trip_id).lstrip("0")
        )
        predictions = predictions.loc[matches]
    if predictions.empty:
        return []

    if "model" in predictions.columns:
        available_models = predictions["model"].dropna().astype(str).unique()
        if model is not None:
            if model not in available_models:
                raise ValueError(
                    f"Model {model!r} was not found in {predictions_path}."
                )
            predictions = predictions.loc[predictions["model"].astype(str).eq(model)]
        elif len(available_models):
            selected_model = available_models[0]
            predictions = predictions.loc[
                predictions["model"].astype(str).eq(selected_model)
            ]
            if len(available_models) > 1:
                LOGGER.info(
                    "Using the first available prediction model, %s; "
                    "select another with --model.",
                    selected_model,
                )

    if "selected_top1" in predictions.columns:
        selected = predictions["selected_top1"]
        if selected.dtype != bool:
            selected = selected.astype(str).str.strip().str.lower().isin(
                {"true", "1", "yes"}
            )
        predictions = predictions.loc[selected]

    arc_column = next(
        (
            column
            for column in (
                "predicted_arc_id",
                "decoded_arc_id",
                "route_arc_id",
                "arc_id",
            )
            if column in predictions.columns
        ),
        None,
    )
    if arc_column is None:
        raise ValueError(
            f"Prediction file {predictions_path} must contain an arc ID column."
        )
    if "observation_index" in predictions.columns:
        predictions = predictions.sort_values("observation_index", kind="stable")
    arc_ids = pd.to_numeric(predictions[arc_column], errors="coerce").dropna()
    return arc_ids.astype("int64").tolist()


def _add_legend(map_view: folium.Map, entries: Sequence[tuple[str, str]]) -> None:
    """Add a compact color key to the map."""
    items = "".join(
        (
            '<div style="margin:3px 0">'
            f'<span style="display:inline-block;width:11px;height:11px;'
            f'background:{color};margin-right:7px"></span>{label}</div>'
        )
        for label, color in entries
    )
    legend = (
        '<div style="position:fixed;bottom:28px;left:28px;z-index:9999;'
        'background:white;padding:9px 12px;border:1px solid #999;'
        'border-radius:4px;font-size:12px">'
        '<b>PathSnap layers</b>'
        f"{items}</div>"
    )
    map_view.get_root().html.add_child(folium.Element(legend))


def create_trip_map(
    trip_dir: str | Path,
    output_path: str | Path,
    predictions_path: str | Path | None = None,
    model: str | None = None,
    road_buffer_meters: float = 50,
    hmm_predictions_path: str | Path | None = None,
) -> Path:
    """Create and save an interactive map for a trip.

    Raw observations, nearby road arcs and nodes, the reference route, and
    optional predicted arcs are added as independent toggleable overlays.
    """
    if road_buffer_meters < 0:
        raise ValueError("road_buffer_meters must be non-negative.")
    trip_dir = Path(trip_dir)
    trip = load_trip(trip_dir)
    track = trip["track"]
    if track.empty:
        raise ValueError(f"Cannot map trip without GPS observations: {trip_dir}")

    prediction_sources = [
        (
            "Classifier-predicted route",
            predictions_path,
            "#f97316",
            model,
        ),
        (
            "HMM-decoded route",
            hmm_predictions_path,
            "#2563eb",
            None,
        ),
    ]
    valid_arc_ids = set(trip["arcs"]["arc_id"].astype(int))
    predicted_routes: list[tuple[str, list[int], str]] = []
    for route_name, route_path, color, route_model in prediction_sources:
        if route_path is None:
            continue
        route_arc_ids = load_predicted_arc_ids(
            route_path, trip_dir.name, route_model
        )
        unknown_ids = set(route_arc_ids) - valid_arc_ids
        if unknown_ids:
            raise ValueError(
                f"{route_name} contains arc IDs not present in {trip_dir}: "
                f"{sorted(unknown_ids)[:10]}"
            )
        predicted_routes.append((route_name, route_arc_ids, color))
    relevant_arc_ids = _relevant_arc_ids(trip, road_buffer_meters)
    track_points = track[["latitude", "longitude"]].to_numpy(dtype=float).tolist()
    center = [
        float(track["latitude"].mean()),
        float(track["longitude"].mean()),
    ]
    map_view = folium.Map(location=center, zoom_start=14, control_scale=True)
    fit_points = list(track_points)
    legend_entries = [
        ("GPS observations", "#e11d48"),
        ("Relevant road network", "#64748b"),
    ]

    gps_layer = folium.FeatureGroup(name="Original GPS observations", show=True)
    folium.PolyLine(
        track_points,
        color="#e11d48",
        weight=3,
        opacity=0.75,
        tooltip="GPS observation sequence",
    ).add_to(gps_layer)
    for index, point in enumerate(track_points):
        folium.CircleMarker(
            point,
            radius=3,
            color="#e11d48",
            weight=1,
            fill=True,
            fill_opacity=0.85,
            tooltip=f"Observation {index + 1}",
        ).add_to(gps_layer)
    folium.Marker(
        track_points[0],
        tooltip="Trip start",
        icon=folium.Icon(color="green", icon="play"),
    ).add_to(gps_layer)
    folium.Marker(
        track_points[-1],
        tooltip="Trip end",
        icon=folium.Icon(color="purple", icon="flag"),
    ).add_to(gps_layer)
    gps_layer.add_to(map_view)

    road_layer = folium.FeatureGroup(name="Relevant road network", show=True)
    road_geometries = _arc_geometries(trip, relevant_arc_ids)
    relevant_arc_id_set = set(relevant_arc_ids)
    relevant_arcs = trip["arcs"].loc[
        trip["arcs"]["arc_id"].isin(relevant_arc_id_set)
    ]
    relevant_nodes = {
        int(node_id)
        for arc in relevant_arcs.itertuples()
        for node_id in (arc.from_node, arc.to_node)
    }
    if road_geometries:
        folium.GeoJson(
            {
                "type": "MultiLineString",
                "coordinates": [
                    [[longitude, latitude] for latitude, longitude in geometry]
                    for geometry in road_geometries
                ],
            },
            style_function=lambda _feature: {
                "color": "#64748b",
                "weight": 2,
                "opacity": 0.6,
            },
        ).add_to(road_layer)
    if relevant_nodes:
        road_node_coordinates = trip["nodes"].loc[
            sorted(relevant_nodes), ["longitude", "latitude"]
        ].to_numpy(dtype=float)
        folium.GeoJson(
            {
                "type": "MultiPoint",
                "coordinates": road_node_coordinates.tolist(),
            },
            marker=folium.CircleMarker(
                radius=2,
                color="#334155",
                fill=True,
                fill_opacity=0.7,
            ),
        ).add_to(road_layer)
    road_layer.add_to(map_view)

    reference_arc_ids = trip["route"]["arc_id"].astype(int).tolist()
    reference_geometries = _arc_geometries(trip, reference_arc_ids)
    if reference_geometries:
        reference_layer = folium.FeatureGroup(name="Reference route", show=True)
        for geometry in reference_geometries:
            folium.PolyLine(
                geometry,
                color="#16a34a",
                weight=6,
                opacity=0.9,
                tooltip="Ground-truth reference route",
            ).add_to(reference_layer)
        reference_layer.add_to(map_view)
        fit_points.extend(
            point for geometry in reference_geometries for point in geometry
        )
        legend_entries.append(("Reference route", "#16a34a"))

    for route_name, route_arc_ids, color in predicted_routes:
        route_geometries = _arc_geometries(trip, route_arc_ids)
        if not route_geometries:
            continue
        predicted_layer = folium.FeatureGroup(name=route_name, show=True)
        for geometry in route_geometries:
            folium.PolyLine(
                geometry,
                color=color,
                weight=5,
                opacity=0.9,
                tooltip=route_name,
            ).add_to(predicted_layer)
        predicted_layer.add_to(map_view)
        fit_points.extend(
            point for geometry in route_geometries for point in geometry
        )
        legend_entries.append((route_name, color))

    latitudes, longitudes = zip(*fit_points)
    map_view.fit_bounds(
        [
            [min(latitudes), min(longitudes)],
            [max(latitudes), max(longitudes)],
        ],
        padding=(20, 20),
    )
    folium.LayerControl(collapsed=False).add_to(map_view)
    _add_legend(map_view, legend_entries)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    map_view.save(output_path)
    return output_path


def generate_sample_maps(
    workspace: str | Path = PROJECT_ROOT,
    output_dir: str | Path | None = None,
    trip_id: str | None = None,
    predictions_path: str | Path | None = None,
    model: str | None = None,
    max_maps: int = 2,
    hmm_predictions_path: str | Path | None = None,
) -> list[Path]:
    """Map a sample from each data partition, then fill up to ``max_maps``."""
    workspace = Path(workspace).resolve()
    if max_maps < 1:
        raise ValueError("max_maps must be at least 1.")
    trip_dirs = [
        trip_dir
        for trip_dir in discover_trip_dirs(workspace)
        if (trip_dir / f"{trip_dir.name}.track").is_file()
    ]
    if trip_id is not None:
        trip_dirs = [trip_dir for trip_dir in trip_dirs if trip_dir.name == trip_id]
        if not trip_dirs:
            raise ValueError(f"Trip {trip_id!r} was not found under {workspace}.")
    if not trip_dirs:
        raise FileNotFoundError(f"No PathSnap trip files were found under {workspace}.")

    sampled_dirs = []
    seen_datasets = set()
    for trip_dir in trip_dirs:
        if trip_id is not None or trip_dir.parent not in seen_datasets:
            sampled_dirs.append(trip_dir)
            seen_datasets.add(trip_dir.parent)
    if len(sampled_dirs) < max_maps:
        sampled_set = set(sampled_dirs)
        sampled_dirs.extend(
            trip_dir
            for trip_dir in trip_dirs
            if trip_dir not in sampled_set and len(sampled_dirs) < max_maps
        )

    output_dir = (
        Path(output_dir)
        if output_dir is not None
        else workspace / "outputs" / "maps"
    )
    if not output_dir.is_absolute():
        output_dir = workspace / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if predictions_path is None:
        default_predictions = workspace / "outputs" / "student_test_predictions.csv"
        predictions_path = default_predictions if default_predictions.is_file() else None
    elif not Path(predictions_path).is_absolute():
        predictions_path = workspace / predictions_path
    if hmm_predictions_path is not None and not Path(
        hmm_predictions_path
    ).is_absolute():
        hmm_predictions_path = workspace / hmm_predictions_path

    generated = []
    for trip_dir in sampled_dirs[:max_maps]:
        dataset_name = trip_dir.parent.name
        output_path = output_dir / f"{dataset_name}_{trip_dir.name}_map.html"
        generated.append(
            create_trip_map(
                trip_dir,
                output_path,
                predictions_path=predictions_path,
                model=model,
                hmm_predictions_path=hmm_predictions_path,
            )
        )
        LOGGER.info("Wrote interactive map: %s", output_path)
    return generated


def main() -> None:
    """Parse command-line options and generate maps for discovered trips."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace",
        type=Path,
        default=PROJECT_ROOT,
        help="Repository or dataset root to scan (defaults to this project).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Map output directory (defaults to <workspace>/outputs/maps).",
    )
    parser.add_argument("--trip-id", help="Generate a map for this trip ID.")
    parser.add_argument(
        "--max-maps",
        type=int,
        default=2,
        help="Maximum number of discovered maps to generate (default: 2).",
    )
    parser.add_argument(
        "--predictions",
        "--classifier-predictions",
        dest="predictions",
        type=Path,
        help="Optional classifier prediction CSV.",
    )
    parser.add_argument(
        "--hmm-predictions",
        type=Path,
        help="Optional HMM-decoded route CSV or .route file.",
    )
    parser.add_argument(
        "--model",
        help="Prediction model name when the CSV contains multiple models.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    paths = generate_sample_maps(
        workspace=args.workspace,
        output_dir=args.output_dir,
        trip_id=args.trip_id,
        predictions_path=args.predictions,
        model=args.model,
        max_maps=args.max_maps,
        hmm_predictions_path=args.hmm_predictions,
    )
    print("\n".join(str(path) for path in paths))


if __name__ == "__main__":
    main()
