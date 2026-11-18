"""Train and compare baseline route-candidate classifiers."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, log_loss
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_sample_weight

if __package__:
    from .features import FEATURE_COLUMNS
else:
    from features import FEATURE_COLUMNS

LOGGER = logging.getLogger(__name__)
CV_SPLITS = 5
MAX_TRAINING_ROWS = 250_000
MAX_KNN_TRAINING_ROWS = 50_000
RANDOM_STATE = 42
OBSERVATION_COLUMNS = ["trip_id", "observation_index"]
REQUIRED_COLUMNS = [
    *FEATURE_COLUMNS,
    "dataset",
    "trip_id",
    "observation_index",
    "is_route_arc",
]


def build_models(random_state: int = RANDOM_STATE) -> dict[str, object]:
    """Create five models from distinct classifier families."""
    return {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=random_state,
            ),
        ),
        "random_forest": RandomForestClassifier(
            class_weight="balanced_subsample",
            max_depth=16,
            min_samples_leaf=2,
            n_estimators=100,
            n_jobs=-1,
            random_state=random_state,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            early_stopping=True,
            l2_regularization=1.0,
            learning_rate=0.1,
            max_iter=100,
            max_leaf_nodes=31,
            random_state=random_state,
        ),
        "knn": make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(
                algorithm="kd_tree",
                n_jobs=1,
                n_neighbors=15,
                weights="distance",
            ),
        ),
        "linear_svm": make_pipeline(
            StandardScaler(),
            LinearSVC(
                class_weight="balanced",
                max_iter=5000,
                random_state=random_state,
            ),
        ),
    }


def positive_class_probabilities(model: object, features: pd.DataFrame) -> np.ndarray:
    """Return scores suitable for ranking and binary log-loss evaluation."""
    if hasattr(model, "predict_proba"):
        classes = np.asarray(model.classes_)
        positive_column = np.flatnonzero(classes == 1)
        if not len(positive_column):
            return np.zeros(len(features), dtype=float)
        return model.predict_proba(features)[:, positive_column[0]]

    decision = np.asarray(model.decision_function(features), dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(decision, -50.0, 50.0)))


def evaluate_predictions(
    frame: pd.DataFrame, y_true: np.ndarray, probabilities: np.ndarray
) -> dict[str, object]:
    """Calculate candidate-level metrics and observation-level route ranking."""
    y_true = np.asarray(y_true, dtype=np.int8)
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-15, 1 - 1e-15)
    predicted = (probabilities >= 0.5).astype(np.int8)

    scored = frame[OBSERVATION_COLUMNS].reset_index(drop=True).copy()
    scored["truth"] = y_true
    scored["probability"] = probabilities
    winners = scored.loc[
        scored.groupby(OBSERVATION_COLUMNS, sort=False)["probability"].idxmax()
    ]
    observation_labels = scored.groupby(OBSERVATION_COLUMNS, sort=False)["truth"].max()
    matrix = confusion_matrix(y_true, predicted, labels=[0, 1])
    return {
        "observation_top1_accuracy": float(winners["truth"].mean()),
        "macro_f1_score": float(
            f1_score(y_true, predicted, average="macro", zero_division=0)
        ),
        "log_loss": float(log_loss(y_true, probabilities, labels=[0, 1])),
        "candidate_miss_rate": float((observation_labels == 0).mean()),
        "confusion_matrix": matrix,
        "candidate_rows": int(len(y_true)),
        "observations": int(len(observation_labels)),
    }


def _sample_training_positions(
    frame: pd.DataFrame,
    labels: np.ndarray,
    positions: np.ndarray,
    row_limit: int,
    random_state: int,
) -> np.ndarray:
    """Bound fit cost without excluding any trip from a training fold."""
    if len(positions) <= row_limit:
        return positions
    rng = np.random.default_rng(random_state)
    trip_rows = pd.DataFrame(
        {"position": positions, "trip_id": frame.iloc[positions]["trip_id"].to_numpy()}
    )
    representatives = trip_rows.groupby("trip_id", sort=False).sample(
        n=1, random_state=random_state
    )["position"].to_numpy(dtype=np.int64)
    remaining = np.setdiff1d(positions, representatives, assume_unique=False)
    budget = row_limit - len(representatives)
    if budget <= 0:
        return rng.choice(representatives, size=row_limit, replace=False)

    available_labels = labels[remaining]
    chosen: list[np.ndarray] = []
    for label in np.unique(available_labels):
        label_positions = remaining[available_labels == label]
        count = round(budget * len(label_positions) / len(remaining))
        if count:
            chosen.append(
                rng.choice(label_positions, size=min(count, len(label_positions)), replace=False)
            )
    sampled = np.concatenate([representatives, *chosen]).astype(np.int64)
    if len(sampled) < row_limit:
        unselected = np.setdiff1d(remaining, np.concatenate(chosen) if chosen else [])
        extra = rng.choice(
            unselected,
            size=min(row_limit - len(sampled), len(unselected)),
            replace=False,
        )
        sampled = np.concatenate([sampled, extra])
    return np.sort(sampled[:row_limit])


def _fit(model: object, x_train: pd.DataFrame, y_train: np.ndarray) -> object:
    if isinstance(model, HistGradientBoostingClassifier):
        model.fit(x_train, y_train, sample_weight=compute_sample_weight("balanced", y_train))
    else:
        model.fit(x_train, y_train)
    return model


def run_experiment(
    features: pd.DataFrame,
    output_dir: str | Path = "outputs",
    cv_splits: int = CV_SPLITS,
    max_training_rows: int = MAX_TRAINING_ROWS,
    max_knn_training_rows: int = MAX_KNN_TRAINING_ROWS,
) -> pd.DataFrame:
    """Cross-validate by trip, then fit each model and score student-test trips."""
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(features.columns))
    if missing_columns:
        raise ValueError(f"Candidate feature data is missing columns: {missing_columns}")

    training = features.loc[
        features["dataset"].eq("train") & features["is_route_arc"].notna()
    ].reset_index(drop=True)
    student_test = features.loc[features["dataset"].eq("student_test")].reset_index(
        drop=True
    )
    if training.empty:
        raise ValueError("No labeled training candidate rows were found.")
    raw_labels = training["is_route_arc"].to_numpy()
    if not np.isin(raw_labels, [0, 1]).all():
        raise ValueError("Training labels must be binary values, 0 or 1.")
    labels = raw_labels.astype(np.int8)
    if len(np.unique(labels)) != 2:
        raise ValueError("Training labels must contain both binary classes, 0 and 1.")
    trip_groups = training["trip_id"].to_numpy()
    unique_trips = np.unique(trip_groups)
    if len(unique_trips) < cv_splits:
        raise ValueError(
            f"At least {cv_splits} training trips are required for grouped CV; "
            f"found {len(unique_trips)}."
        )
    if student_test.empty:
        raise ValueError("No student-testing candidate rows were found.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_templates = build_models()
    oof_scores = {
        name: np.full(len(training), np.nan, dtype=float) for name in model_templates
    }
    fold_ids = np.full(len(training), -1, dtype=np.int8)
    inference_seconds = {name: 0.0 for name in model_templates}
    student_inference_seconds = {name: 0.0 for name in model_templates}
    splitter = GroupKFold(n_splits=cv_splits)
    x_training = training[FEATURE_COLUMNS]

    for fold, (fit_positions, validation_positions) in enumerate(
        splitter.split(x_training, labels, groups=trip_groups), start=1
    ):
        fold_ids[validation_positions] = fold
        for model_index, (name, template) in enumerate(model_templates.items()):
            row_limit = (
                max_knn_training_rows if name == "knn" else max_training_rows
            )
            sampled_positions = _sample_training_positions(
                training,
                labels,
                fit_positions,
                row_limit,
                RANDOM_STATE + fold * 100 + model_index,
            )
            model = clone(template)
            _fit(model, x_training.iloc[sampled_positions], labels[sampled_positions])
            start = time.perf_counter()
            scores = positive_class_probabilities(
                model, x_training.iloc[validation_positions]
            )
            inference_seconds[name] += time.perf_counter() - start
            oof_scores[name][validation_positions] = scores
            LOGGER.info(
                "Fold %d/%d: fitted %s on %d candidate rows",
                fold,
                cv_splits,
                name,
                len(sampled_positions),
            )

    if (fold_ids < 1).any() or any(np.isnan(scores).any() for scores in oof_scores.values()):
        raise RuntimeError("Cross-validation did not produce predictions for every row.")

    metric_rows: list[dict[str, object]] = []
    confusion_rows: list[dict[str, object]] = []
    oof_output = training[
        ["trip_id", "observation_index", "arc_id", "is_route_arc"]
    ].copy()
    oof_output["fold"] = fold_ids
    for name, scores in oof_scores.items():
        metrics = evaluate_predictions(training, labels, scores)
        metric_rows.append(
            _metric_row(
                name,
                "train_cross_validation",
                metrics,
                inference_seconds[name],
            )
        )
        _append_confusion_rows(confusion_rows, name, metrics["confusion_matrix"])
        oof_output[f"{name}_probability"] = scores
        oof_output[f"{name}_prediction"] = (scores >= 0.5).astype(np.int8)

    oof_output.to_csv(output_dir / "cross_validation_predictions.csv", index=False)
    test_prediction_frames: list[pd.DataFrame] = []
    test_features = student_test[FEATURE_COLUMNS]
    test_observations = student_test[OBSERVATION_COLUMNS]
    for model_index, (name, template) in enumerate(model_templates.items()):
        row_limit = max_knn_training_rows if name == "knn" else max_training_rows
        sampled_positions = _sample_training_positions(
            training,
            labels,
            np.arange(len(training)),
            row_limit,
            RANDOM_STATE + 1000 + model_index,
        )
        model = clone(template)
        _fit(model, x_training.iloc[sampled_positions], labels[sampled_positions])
        start = time.perf_counter()
        scores = positive_class_probabilities(model, test_features)
        student_inference_seconds[name] += time.perf_counter() - start
        test_output = student_test[
            ["trip_id", "observation_index", "arc_id"]
        ].copy()
        test_output["model"] = name
        test_output["route_probability"] = scores
        test_output["selected_top1"] = False
        winner_positions = (
            pd.DataFrame(
                {
                    "position": np.arange(len(test_output)),
                    "trip_id": test_observations["trip_id"].to_numpy(),
                    "observation_index": test_observations[
                        "observation_index"
                    ].to_numpy(),
                    "score": scores,
                }
            )
            .groupby(OBSERVATION_COLUMNS, sort=False)["score"]
            .idxmax()
            .to_numpy()
        )
        test_output.iloc[winner_positions, test_output.columns.get_loc("selected_top1")] = True
        test_prediction_frames.append(test_output)
        joblib.dump(model, output_dir / f"{name}.joblib")
        metric_rows.append(
            _metric_row(
                name,
                "student_test_inference",
                None,
                student_inference_seconds[name],
                candidate_rows=len(student_test),
                observations=student_test[OBSERVATION_COLUMNS]
                .drop_duplicates()
                .shape[0],
            )
        )

    pd.concat(test_prediction_frames, ignore_index=True).to_csv(
        output_dir / "student_test_predictions.csv", index=False
    )
    pd.DataFrame(metric_rows).to_csv(output_dir / "model_comparison.csv", index=False)
    pd.DataFrame(confusion_rows).to_csv(
        output_dir / "confusion_matrices.csv", index=False
    )
    _plot_confusion_matrices(confusion_rows, output_dir / "confusion_matrices.png")
    return pd.DataFrame(metric_rows)


def _metric_row(
    model: str,
    evaluation: str,
    metrics: dict[str, object] | None,
    inference_seconds: float,
    candidate_rows: int | None = None,
    observations: int | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "model": model,
        "evaluation": evaluation,
        "observation_top1_accuracy": np.nan,
        "macro_f1_score": np.nan,
        "log_loss": np.nan,
        "candidate_miss_rate": np.nan,
        "inference_seconds": inference_seconds,
        "inference_ms_per_observation": np.nan,
        "candidate_rows": candidate_rows,
        "observations": observations,
    }
    if metrics is not None:
        for key in (
            "observation_top1_accuracy",
            "macro_f1_score",
            "log_loss",
            "candidate_miss_rate",
            "candidate_rows",
            "observations",
        ):
            result[key] = metrics[key]
        observations = int(metrics["observations"])
    if observations:
        result["inference_ms_per_observation"] = (
            inference_seconds * 1000.0 / observations
        )
    return result


def _append_confusion_rows(
    rows: list[dict[str, object]], model: str, matrix: np.ndarray
) -> None:
    for actual_index, actual_label in enumerate((0, 1)):
        for predicted_index, predicted_label in enumerate((0, 1)):
            rows.append(
                {
                    "model": model,
                    "evaluation": "train_cross_validation",
                    "actual_label": actual_label,
                    "predicted_label": predicted_label,
                    "count": int(matrix[actual_index, predicted_index]),
                }
            )


def _plot_confusion_matrices(
    rows: list[dict[str, object]], output_path: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = list(dict.fromkeys(str(row["model"]) for row in rows))
    if not models:
        raise ValueError("No confusion matrices are available to plot.")
    figure, axes = plt.subplots(
        1, len(models), figsize=(4 * len(models), 4), squeeze=False
    )
    for axis, model in zip(axes[0], models):
        matrix_rows = [row for row in rows if row["model"] == model]
        matrix = np.zeros((2, 2), dtype=np.int64)
        for row in matrix_rows:
            matrix[int(row["actual_label"]), int(row["predicted_label"])] = int(
                row["count"]
            )
        axis.imshow(matrix, cmap="Blues")
        axis.set_title(model.replace("_", " ").title())
        axis.set_xlabel("Predicted route membership")
        axis.set_ylabel("Actual route membership")
        axis.set_xticks([0, 1], labels=["No", "Yes"])
        axis.set_yticks([0, 1], labels=["No", "Yes"])
        threshold = matrix.max() / 2 if matrix.size else 0
        for row_index in range(2):
            for column_index in range(2):
                axis.text(
                    column_index,
                    row_index,
                    f"{matrix[row_index, column_index]:,}",
                    ha="center",
                    va="center",
                    color="white" if matrix[row_index, column_index] > threshold else "black",
                )
    figure.suptitle("Out-of-fold candidate-level confusion matrices")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-csv", default="outputs/candidate_features.csv")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--cv-splits", type=int, default=CV_SPLITS)
    args = parser.parse_args()
    if args.cv_splits < 2:
        parser.error("--cv-splits must be at least 2")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    features_path = Path(args.features_csv)
    if not features_path.is_file():
        raise FileNotFoundError(
            f"Candidate features not found at {features_path}; run src/features.py first."
        )
    features = pd.read_csv(features_path)
    comparison = run_experiment(features, args.output_dir, args.cv_splits)
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
