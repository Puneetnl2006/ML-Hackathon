"""Shared readers and trip discovery helpers for the PathSnap dataset."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)
FILE_SUFFIXES = (".track", ".nodes", ".arcs", ".route")


def discover_trip_dirs(root: str | Path) -> list[Path]:
    """Return directories containing one or more PathSnap data files."""
    root = Path(root)
    if not root.exists():
        LOGGER.warning("Dataset directory does not exist: %s", root)
        return []
    if not root.is_dir():
        raise NotADirectoryError(f"Dataset path is not a directory: {root}")
    return sorted(
        {
            path.parent
            for suffix in FILE_SUFFIXES
            for path in root.rglob(f"*{suffix}")
            if path.is_file()
        }
    )


def read_numeric_file(
    path: str | Path,
    min_columns: int,
    max_columns: int | None = None,
    preserve_row_index: bool = False,
) -> pd.DataFrame:
    """Read a whitespace-delimited numeric file with no required header."""
    path = Path(path)
    if not path.exists():
        LOGGER.warning("Missing data file: %s", path)
        return pd.DataFrame()
    if path.stat().st_size == 0:
        return pd.DataFrame()

    try:
        frame = pd.read_csv(
            path,
            sep=r"\s+",
            header=None,
            engine="python",
            on_bad_lines="warn",
        )
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as error:
        raise ValueError(f"Could not parse data file {path}: {error}") from error

    if frame.shape[1] < min_columns:
        LOGGER.warning(
            "Ignoring %s: expected at least %d columns, found %d",
            path,
            min_columns,
            frame.shape[1],
        )
        return pd.DataFrame()

    if max_columns is not None:
        frame = frame.iloc[:, :max_columns]
    frame = frame.apply(pd.to_numeric, errors="coerce")
    invalid_rows = frame.isna().any(axis=1)
    if invalid_rows.any():
        LOGGER.warning("Skipping %d malformed row(s) in %s", invalid_rows.sum(), path)
        frame = frame.loc[~invalid_rows].copy()
    return frame if preserve_row_index else frame.reset_index(drop=True)


def load_trip(
    trip_dir: str | Path, *, include_route: bool = True
) -> dict[str, pd.DataFrame]:
    """Load available files from a trip folder using the documented column order."""
    trip_dir = Path(trip_dir)
    trip_id = trip_dir.name
    track = read_numeric_file(trip_dir / f"{trip_id}.track", 2, max_columns=3)
    if not track.empty:
        if track.shape[1] == 2:
            track.columns = ["longitude", "latitude"]
            track["sample_time"] = range(len(track))
        else:
            track.columns = ["longitude", "latitude", "sample_time"]
        valid_coordinates = (
            track["longitude"].between(-180, 180)
            & track["latitude"].between(-90, 90)
        )
        if (~valid_coordinates).any():
            LOGGER.warning(
                "Skipping %d out-of-range GPS row(s) in %s.track",
                (~valid_coordinates).sum(),
                trip_id,
            )
            track = track.loc[valid_coordinates].reset_index(drop=True)
    else:
        track = pd.DataFrame(columns=["longitude", "latitude", "sample_time"])

    nodes = read_numeric_file(
        trip_dir / f"{trip_id}.nodes",
        2,
        max_columns=2,
        preserve_row_index=True,
    )
    nodes.columns = ["longitude", "latitude"] if not nodes.empty else []
    if not nodes.empty:
        valid_coordinates = (
            nodes["longitude"].between(-180, 180)
            & nodes["latitude"].between(-90, 90)
        )
        if (~valid_coordinates).any():
            LOGGER.warning(
                "Skipping %d out-of-range node row(s) in %s.nodes",
                (~valid_coordinates).sum(),
                trip_id,
            )
            nodes = nodes.loc[valid_coordinates]

    arcs = read_numeric_file(
        trip_dir / f"{trip_id}.arcs",
        2,
        max_columns=2,
        preserve_row_index=True,
    )
    if not arcs.empty:
        arcs.columns = ["from_node", "to_node"]
        arcs["arc_id"] = arcs.index.astype("int64")
    else:
        arcs = pd.DataFrame(columns=["from_node", "to_node", "arc_id"])

    route = (
        read_numeric_file(trip_dir / f"{trip_id}.route", 1, max_columns=1)
        if include_route
        else pd.DataFrame()
    )
    if not route.empty:
        route.columns = ["arc_id"]
        route["arc_id"] = route["arc_id"].astype("int64")
    else:
        route = pd.DataFrame(columns=["arc_id"])

    return {"track": track, "nodes": nodes, "arcs": arcs, "route": route}
