"""Compare independent candidate classification with HMM route decoding."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

if __package__:
    from .data_utils import discover_trip_dirs, load_trip
    from .decoder import ViterbiDecoder
    from .features import FEATURE_COLUMNS
else:
    from data_utils import discover_trip_dirs, load_trip
    from decoder import ViterbiDecoder
    from features import FEATURE_COLUMNS

LOGGER = logging.getLogger(__name__)
DEFAULT_CANDIDATE_FEATURES = Path("outputs/candidate_features.csv")


def _validate_features(features: pd.DataFrame) -> None:
    required = {
        "dataset",
        "trip_id",
        "observation_index",
        "arc_id",
        "is_route_arc",
        *FEATURE_COLUMNS,
    }
    missing = required.difference(features.columns)
    if missing:
        raise ValueError(
            "Candidate feature file is missing required columns: "
            + ", ".join(sorted(missing))
        )


def _fit_candidate_classifier(
    features: pd.DataFrame, validation_trip_ids: set[str], seed: int
) -> Pipeline:
    training = features.loc[
        features["dataset"].eq("train")
        & ~features["trip_id"].astype(str).isin(validation_trip_ids)
    ]
    training = training.dropna(subset=[*FEATURE_COLUMNS, "is_route_arc"])
    labels = training["is_route_arc"].astype(int)
    if len(training) == 0 or labels.nunique() != 2:
        raise ValueError(
            "Training candidates must contain both route and non-route labels."
        )
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1_000,
            class_weight="balanced",
            random_state=seed,
        ),
    )
    classifier.fit(training[FEATURE_COLUMNS], labels)
    return classifier


def _validation_split(
    train_trip_ids: list[str], validation_fraction: float, seed: int
) -> set[str]:
    if len(train_trip_ids) < 2:
        raise ValueError("At least two labeled training trips are required.")
    if not 0 < validation_fraction < 1:
        raise ValueError("Validation fraction must be between zero and one.")
    shuffled = np.random.default_rng(seed).permutation(sorted(train_trip_ids))
    count = min(max(1, int(round(len(shuffled) * validation_fraction))), len(shuffled) - 1)
    return {str(trip_id) for trip_id in shuffled[:count]}


def _arc_geometry(
    decoder: ViterbiDecoder, arc_id: int | None
) -> tuple[float, float, float, float]:
    if arc_id is None or arc_id not in decoder.arcs:
        return (np.nan, np.nan, np.nan, np.nan)
    _, _, _, start, end = decoder.arcs[arc_id]
    return (float(start[0]), float(start[1]), float(end[0]), float(end[1]))


def _trip_metrics(
    dataset: str,
    trip_id: str,
    pipeline: str,
    predictions: list[int | None],
    true_route_arc_ids: set[int] | None,
    track: pd.DataFrame,
    decoder: ViterbiDecoder,
) -> dict[str, Any]:
    has_truth = true_route_arc_ids is not None
    correct = (
        sum(prediction in true_route_arc_ids for prediction in predictions)
        if has_truth
        else 0
    )
    point_count = len(predictions)
    transition_count = invalid_transitions = 0
    for index, (previous_arc, current_arc) in enumerate(
        zip(predictions, predictions[1:])
    ):
        if previous_arc is None or current_arc is None:
            continue
        if decoder.is_gap_between(track, index, index + 1):
            continue
        transition_count += 1
        if not decoder.is_transition_reachable(previous_arc, current_arc):
            invalid_transitions += 1
    return {
        "scope": "trip",
        "dataset": dataset,
        "trip_id": trip_id,
        "pipeline": pipeline,
        "observation_count": point_count,
        "pointwise_accuracy": correct / point_count if has_truth and point_count else np.nan,
        "correct_points": correct if has_truth else np.nan,
        "labeled_points": point_count if has_truth else 0,
        "transition_count": transition_count,
        "invalid_transition_count": invalid_transitions,
        "route_continuity": (
            1.0 - invalid_transitions / transition_count
            if transition_count
            else np.nan
        ),
    }


def _overall_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return metrics
    summaries: list[dict[str, Any]] = []
    for (dataset, pipeline), rows in metrics.groupby(
        ["dataset", "pipeline"], sort=True
    ):
        transition_count = int(rows["transition_count"].sum())
        invalid_count = int(rows["invalid_transition_count"].sum())
        labeled_points = int(rows["labeled_points"].sum())
        correct_points = int(rows["correct_points"].fillna(0).sum())
        summaries.append(
            {
                "scope": "overall",
                "dataset": dataset,
                "trip_id": "ALL",
                "pipeline": pipeline,
                "observation_count": int(rows["observation_count"].sum()),
                "pointwise_accuracy": (
                    correct_points / labeled_points if labeled_points else np.nan
                ),
                "correct_points": correct_points if labeled_points else np.nan,
                "labeled_points": labeled_points,
                "transition_count": transition_count,
                "invalid_transition_count": invalid_count,
                "route_continuity": (
                    1.0 - invalid_count / transition_count
                    if transition_count
                    else np.nan
                ),
            }
        )
    return pd.concat([metrics, pd.DataFrame.from_records(summaries)], ignore_index=True)


def evaluate_decoder(
    candidate_features_path: str | Path = DEFAULT_CANDIDATE_FEATURES,
    train_dir: str | Path = "Train",
    test_dir: str | Path = "student_test",
    output_dir: str | Path = "decoder_outputs",
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[Path, Path]:
    """Train a trip-held-out classifier, decode routes, and write evaluation CSVs.

    The route files provide weak point labels: a prediction is correct when its
    arc belongs to the trajectory's known route. ``student_test`` has no route
    labels, so its accuracy is reported as unavailable while continuity remains
    measurable.
    """
    candidate_features_path = Path(candidate_features_path)
    features = pd.read_csv(candidate_features_path, dtype={"trip_id": str})
    _validate_features(features)
    train_trip_ids = sorted(
        features.loc[features["dataset"].eq("train"), "trip_id"]
        .dropna()
        .astype(str)
        .unique()
    )
    validation_trip_ids = _validation_split(train_trip_ids, validation_fraction, seed)
    classifier = _fit_candidate_classifier(features, validation_trip_ids, seed)

    partitions = [
        ("validation", validation_trip_ids, Path(train_dir)),
        (
            "student_test",
            set(
                features.loc[features["dataset"].eq("student_test"), "trip_id"]
                .dropna()
                .astype(str)
                .unique()
            ),
            Path(test_dir),
        ),
    ]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    metric_records: list[dict[str, Any]] = []
    visualization_records: list[dict[str, Any]] = []

    for dataset_name, selected_trip_ids, partition_root in partitions:
        selected_features = features.loc[
            features["trip_id"].astype(str).isin(selected_trip_ids)
            & features["dataset"].eq(
                "train" if dataset_name == "validation" else "student_test"
            )
        ].copy()
        if selected_features.empty:
            LOGGER.info("No candidate rows to evaluate for %s.", dataset_name)
            continue
        probabilities = classifier.predict_proba(selected_features[FEATURE_COLUMNS])
        positive_column = list(classifier.classes_).index(1)
        selected_features["classifier_probability"] = probabilities[:, positive_column]

        for trip_dir in discover_trip_dirs(partition_root):
            trip_id = trip_dir.name
            trip_candidates = selected_features.loc[
                selected_features["trip_id"].astype(str).eq(trip_id)
            ].copy()
            if trip_candidates.empty:
                continue
            LOGGER.info("Evaluating %s trip %s.", dataset_name, trip_id)
            trip = load_trip(trip_dir)
            track, nodes, arcs = trip["track"], trip["nodes"], trip["arcs"]
            if track.empty or nodes.empty or arcs.empty:
                LOGGER.warning("Skipping %s: track or road graph is missing.", trip_id)
                continue

            classifier_predictions: list[int | None] = [None] * len(track)
            for observation_index, rows in trip_candidates.groupby(
                "observation_index", sort=False
            ):
                index = int(observation_index)
                if 0 <= index < len(track):
                    best = rows["classifier_probability"].astype(float).idxmax()
                    classifier_predictions[index] = int(rows.loc[best, "arc_id"])

            decoder = ViterbiDecoder(nodes, arcs)
            decoded_predictions = decoder.decode(track, trip_candidates)
            route_ids = (
                set(trip["route"]["arc_id"].astype(int))
                if dataset_name == "validation" and not trip["route"].empty
                else None
            )
            metric_records.append(
                _trip_metrics(
                    dataset_name,
                    trip_id,
                    "classifier_only",
                    classifier_predictions,
                    route_ids,
                    track,
                    decoder,
                )
            )
            metric_records.append(
                _trip_metrics(
                    dataset_name,
                    trip_id,
                    "hmm_viterbi",
                    decoded_predictions,
                    route_ids,
                    track,
                    decoder,
                )
            )

            candidate_groups = {
                int(index): rows
                for index, rows in trip_candidates.groupby(
                    "observation_index", sort=False
                )
            }
            for index, row in track.reset_index(drop=True).iterrows():
                classifier_arc = classifier_predictions[index]
                decoded_arc = decoded_predictions[index]
                candidate_rows = candidate_groups.get(index)
                classifier_geometry = _arc_geometry(decoder, classifier_arc)
                decoded_geometry = _arc_geometry(decoder, decoded_arc)
                visualization_records.append(
                    {
                        "dataset": dataset_name,
                        "trip_id": trip_id,
                        "observation_index": index,
                        "longitude": float(row["longitude"]),
                        "latitude": float(row["latitude"]),
                        "true_route_arc_ids": (
                            ";".join(map(str, sorted(route_ids)))
                            if route_ids is not None
                            else ""
                        ),
                        "classifier_arc_id": classifier_arc,
                        "classifier_probability": (
                            float(
                                candidate_rows.loc[
                                    candidate_rows["arc_id"].eq(classifier_arc),
                                    "classifier_probability",
                                ].iloc[0]
                            )
                            if classifier_arc is not None and candidate_rows is not None
                            else np.nan
                        ),
                        "decoded_arc_id": decoded_arc,
                        "classifier_is_route_arc": (
                            classifier_arc in route_ids
                            if classifier_arc is not None and route_ids is not None
                            else np.nan
                        ),
                        "decoded_is_route_arc": (
                            decoded_arc in route_ids
                            if decoded_arc is not None and route_ids is not None
                            else np.nan
                        ),
                        "classifier_arc_start_lon": classifier_geometry[0],
                        "classifier_arc_start_lat": classifier_geometry[1],
                        "classifier_arc_end_lon": classifier_geometry[2],
                        "classifier_arc_end_lat": classifier_geometry[3],
                        "decoded_arc_start_lon": decoded_geometry[0],
                        "decoded_arc_start_lat": decoded_geometry[1],
                        "decoded_arc_end_lon": decoded_geometry[2],
                        "decoded_arc_end_lat": decoded_geometry[3],
                    }
                )

    trip_metrics = pd.DataFrame.from_records(metric_records)
    metrics = _overall_metrics(trip_metrics)
    metrics_path = output_path / "decoder_metrics.csv"
    visualization_path = output_path / "decoder_visualization.csv"
    metrics.to_csv(metrics_path, index=False)
    pd.DataFrame.from_records(visualization_records).to_csv(
        visualization_path, index=False
    )
    overall = metrics.loc[metrics["scope"].eq("overall")] if not metrics.empty else metrics
    if not overall.empty:
        print(
            overall[
                [
                    "dataset",
                    "pipeline",
                    "observation_count",
                    "pointwise_accuracy",
                    "route_continuity",
                    "invalid_transition_count",
                ]
            ].to_string(index=False)
        )
    LOGGER.info("Wrote evaluation metrics to %s", metrics_path)
    LOGGER.info("Wrote route visualization data to %s", visualization_path)
    return metrics_path, visualization_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-features", default=str(DEFAULT_CANDIDATE_FEATURES))
    parser.add_argument("--train-dir", default="Train")
    parser.add_argument("--test-dir", default="student_test")
    parser.add_argument("--output-dir", default="decoder_outputs")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    evaluate_decoder(
        args.candidate_features,
        args.train_dir,
        args.test_dir,
        args.output_dir,
        args.validation_fraction,
        args.seed,
    )


if __name__ == "__main__":
    main()
