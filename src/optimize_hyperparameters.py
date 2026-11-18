"""Tune the best baseline classifier with group-aware Bayesian optimization."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

if __package__:
    from .data_utils import discover_trip_dirs, load_trip
    from .features import FEATURE_COLUMNS
    from .train_baselines import (
        OBSERVATION_COLUMNS,
        RANDOM_STATE,
        _fit,
        _sample_training_positions,
        build_models,
        evaluate_predictions,
        positive_class_probabilities,
    )
else:
    from data_utils import discover_trip_dirs, load_trip
    from features import FEATURE_COLUMNS
    from train_baselines import (
        OBSERVATION_COLUMNS,
        RANDOM_STATE,
        _fit,
        _sample_training_positions,
        build_models,
        evaluate_predictions,
        positive_class_probabilities,
    )

LOGGER = logging.getLogger(__name__)
DEFAULT_TRIALS = 25
DEFAULT_CV_SPLITS = 5
DEFAULT_MAX_TRAINING_ROWS = 50_000
DEFAULT_MAX_KNN_TRAINING_ROWS = 25_000
TRAINING_EVALUATION = "train_cross_validation"


def select_best_baseline(comparison: pd.DataFrame) -> str:
    """Choose the model with the highest out-of-fold macro F1 score."""
    required = {"model", "evaluation", "macro_f1_score"}
    missing = required - set(comparison.columns)
    if missing:
        raise ValueError(f"Baseline comparison is missing columns: {sorted(missing)}")
    candidates = comparison.loc[
        comparison["evaluation"].eq(TRAINING_EVALUATION)
    ].dropna(subset=["macro_f1_score"])
    candidates = candidates.loc[candidates["model"].isin(build_models())]
    if candidates.empty:
        raise ValueError(
            "Baseline comparison has no valid train cross-validation scores."
        )
    ranked = candidates.sort_values(
        ["macro_f1_score", "model"], ascending=[False, True], kind="stable"
    )
    return str(ranked.iloc[0]["model"])


def _suggest_parameters(trial: optuna.Trial, model_name: str) -> dict[str, object]:
    """Search the main capacity and regularization controls for each classifier."""
    if model_name == "logistic_regression":
        return {"C": trial.suggest_float("C", 1e-3, 1e3, log=True)}
    if model_name == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
            "max_depth": trial.suggest_categorical(
                "max_depth", [8, 12, 16, 24, 32, None]
            ),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", 0.5, 1.0]
            ),
        }
    if model_name == "hist_gradient_boosting":
        return {
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.02, 0.2, log=True
            ),
            "max_iter": trial.suggest_int("max_iter", 50, 250, step=25),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 7, 63, step=4),
            "l2_regularization": trial.suggest_float(
                "l2_regularization", 1e-4, 10.0, log=True
            ),
        }
    if model_name == "knn":
        return {
            "n_neighbors": trial.suggest_int("n_neighbors", 3, 31, step=2),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            "p": trial.suggest_int("p", 1, 2),
        }
    if model_name == "linear_svm":
        return {"C": trial.suggest_float("C", 1e-3, 1e3, log=True)}
    raise ValueError(f"No hyperparameter search space is defined for {model_name!r}.")


def _estimator_parameters(
    model_name: str, parameters: dict[str, object]
) -> dict[str, object]:
    if model_name == "logistic_regression":
        return {
            f"logisticregression__{key}": value
            for key, value in parameters.items()
        }
    if model_name == "knn":
        return {
            f"kneighborsclassifier__{key}": value
            for key, value in parameters.items()
        }
    if model_name == "linear_svm":
        return {f"linearsvc__{key}": value for key, value in parameters.items()}
    return parameters


def _route_labels(student_test: pd.DataFrame, test_dir: str | Path) -> np.ndarray:
    """Load held-out labels from the route files for final evaluation only."""
    trip_dirs = discover_trip_dirs(test_dir)
    exact_paths = {path.name: path for path in trip_dirs}
    normalized_paths = {path.name.lstrip("0") or "0": path for path in trip_dirs}
    route_arcs: dict[str, set[int]] = {}
    for raw_trip_id in student_test["trip_id"].drop_duplicates():
        trip_id = str(raw_trip_id)
        trip_dir = exact_paths.get(trip_id) or normalized_paths.get(
            trip_id.lstrip("0") or "0"
        )
        if trip_dir is None:
            raise FileNotFoundError(
                f"No student-test trip directory found for trip {trip_id!r} "
                f"under {test_dir}."
            )
        route_path = trip_dir / f"{trip_dir.name}.route"
        if not route_path.is_file():
            raise FileNotFoundError(
                f"Student-test route labels are required for final evaluation: "
                f"{route_path}"
            )
        route = load_trip(trip_dir)["route"]
        if route.empty:
            raise ValueError(f"Student-test route labels are empty: {route_path}")
        route_arcs[trip_id] = set(route["arc_id"].astype(int))

    labels = np.fromiter(
        (
            int(int(arc_id) in route_arcs[str(trip_id)])
            for trip_id, arc_id in zip(
                student_test["trip_id"], student_test["arc_id"]
            )
        ),
        dtype=np.int8,
        count=len(student_test),
    )
    return labels


def _fit_rows_limit(
    model_name: str, max_training_rows: int, max_knn_training_rows: int
) -> int:
    return max_knn_training_rows if model_name == "knn" else max_training_rows


def run_optimization(
    features: pd.DataFrame,
    baseline_comparison: pd.DataFrame,
    output_dir: str | Path = "outputs",
    test_dir: str | Path = "student_test",
    n_trials: int = DEFAULT_TRIALS,
    cv_splits: int = DEFAULT_CV_SPLITS,
    max_training_rows: int = DEFAULT_MAX_TRAINING_ROWS,
    max_knn_training_rows: int = DEFAULT_MAX_KNN_TRAINING_ROWS,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Tune the best baseline on grouped CV and compare both fits on held-out trips."""
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1.")
    if cv_splits < 2:
        raise ValueError("cv_splits must be at least 2.")
    required_columns = {
        *FEATURE_COLUMNS,
        "dataset",
        "trip_id",
        "observation_index",
        "arc_id",
        "is_route_arc",
    }
    missing_columns = sorted(required_columns - set(features.columns))
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
    if student_test.empty:
        raise ValueError("No student-testing candidate rows were found.")
    raw_labels = training["is_route_arc"].to_numpy()
    if not np.isin(raw_labels, [0, 1]).all():
        raise ValueError("Training labels must be binary values, 0 or 1.")
    labels = raw_labels.astype(np.int8)
    if len(np.unique(labels)) != 2:
        raise ValueError("Training labels must contain both binary classes, 0 and 1.")

    model_name = select_best_baseline(baseline_comparison)
    model_templates = build_models()
    template = model_templates[model_name]
    groups = training["trip_id"].to_numpy()
    if len(np.unique(groups)) < cv_splits:
        raise ValueError(
            f"At least {cv_splits} training trips are required for grouped CV; "
            f"found {len(np.unique(groups))}."
        )
    x_training = training[FEATURE_COLUMNS]
    splitter = GroupKFold(n_splits=cv_splits)
    fit_limit = _fit_rows_limit(
        model_name, max_training_rows, max_knn_training_rows
    )

    def objective(trial: optuna.Trial) -> float:
        parameters = _suggest_parameters(trial, model_name)
        oof_scores = np.full(len(training), np.nan, dtype=float)
        for fold, (fit_positions, validation_positions) in enumerate(
            splitter.split(x_training, labels, groups=groups), start=1
        ):
            sampled_positions = _sample_training_positions(
                training,
                labels,
                fit_positions,
                fit_limit,
                random_state + fold * 100,
            )
            model = clone(template).set_params(
                **_estimator_parameters(model_name, parameters)
            )
            _fit(model, x_training.iloc[sampled_positions], labels[sampled_positions])
            oof_scores[validation_positions] = positive_class_probabilities(
                model, x_training.iloc[validation_positions]
            )
        if np.isnan(oof_scores).any():
            raise RuntimeError("Grouped CV did not score every training candidate.")
        return float(
            f1_score(
                labels,
                oof_scores >= 0.5,
                average="macro",
                zero_division=0,
            )
        )

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(objective, n_trials=n_trials)

    test_labels = _route_labels(student_test, test_dir)
    x_test = student_test[FEATURE_COLUMNS]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    test_predictions = student_test[
        ["trip_id", "observation_index", "arc_id"]
    ].copy()
    test_predictions["is_route_arc"] = test_labels
    metric_rows: list[dict[str, object]] = []
    for configuration, parameters in (
        ("untuned", {}),
        ("tuned", study.best_params),
    ):
        sampled_positions = _sample_training_positions(
            training,
            labels,
            np.arange(len(training)),
            fit_limit,
            random_state + 1000,
        )
        model = clone(template).set_params(
            **_estimator_parameters(model_name, parameters)
        )
        _fit(model, x_training.iloc[sampled_positions], labels[sampled_positions])
        probabilities = positive_class_probabilities(model, x_test)
        metrics = evaluate_predictions(student_test, test_labels, probabilities)
        metric_rows.append(
            {
                "model": model_name,
                "configuration": configuration,
                "evaluation": "student_test",
                "observation_top1_accuracy": metrics["observation_top1_accuracy"],
                "macro_f1_score": metrics["macro_f1_score"],
                "log_loss": metrics["log_loss"],
                "candidate_miss_rate": metrics["candidate_miss_rate"],
                "candidate_rows": metrics["candidate_rows"],
                "observations": metrics["observations"],
            }
        )
        test_predictions[f"{configuration}_probability"] = probabilities
        winners = (
            test_predictions.assign(_probability=probabilities)
            .groupby(OBSERVATION_COLUMNS, sort=False)["_probability"]
            .idxmax()
        )
        test_predictions[f"{configuration}_selected_top1"] = False
        test_predictions.loc[
            winners, f"{configuration}_selected_top1"
        ] = True
        if configuration == "tuned":
            joblib.dump(model, output_dir / f"{model_name}_tuned.joblib")

    comparison = pd.DataFrame(metric_rows)
    comparison.to_csv(output_dir / "optimization_comparison.csv", index=False)
    test_predictions.to_csv(
        output_dir / "optimization_student_test_predictions.csv", index=False
    )
    study.trials_dataframe().to_csv(
        output_dir / "optimization_trials.csv", index=False
    )
    with (output_dir / "optimization_best_params.json").open(
        "w", encoding="utf-8"
    ) as best_params_file:
        json.dump(
            {
                "model": model_name,
                "best_cv_macro_f1_score": study.best_value,
                "best_parameters": study.best_params,
                "n_trials": n_trials,
                "cv_splits": cv_splits,
                "cv_strategy": "GroupKFold by trip_id",
            },
            best_params_file,
            indent=2,
            sort_keys=True,
        )
        best_params_file.write("\n")
    LOGGER.info(
        "Optimized %s: best grouped-CV macro F1 %.4f",
        model_name,
        study.best_value,
    )
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-csv", default="outputs/candidate_features.csv")
    parser.add_argument("--comparison-csv", default="outputs/model_comparison.csv")
    parser.add_argument("--test-dir", default="student_test")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--n-trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--cv-splits", type=int, default=DEFAULT_CV_SPLITS)
    parser.add_argument(
        "--max-training-rows", type=int, default=DEFAULT_MAX_TRAINING_ROWS
    )
    parser.add_argument(
        "--max-knn-training-rows",
        type=int,
        default=DEFAULT_MAX_KNN_TRAINING_ROWS,
    )
    args = parser.parse_args()
    if args.n_trials < 1:
        parser.error("--n-trials must be at least 1")
    if args.cv_splits < 2:
        parser.error("--cv-splits must be at least 2")
    for argument in ("max_training_rows", "max_knn_training_rows"):
        if getattr(args, argument) < 1:
            parser.error(f"--{argument.replace('_', '-')} must be positive")

    features_path = Path(args.features_csv)
    comparison_path = Path(args.comparison_csv)
    if not features_path.is_file():
        raise FileNotFoundError(f"Candidate features not found: {features_path}")
    if not comparison_path.is_file():
        raise FileNotFoundError(
            f"Baseline comparison not found: {comparison_path}; "
            "run src/train_baselines.py first."
        )
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    comparison = run_optimization(
        pd.read_csv(features_path),
        pd.read_csv(comparison_path),
        args.output_dir,
        args.test_dir,
        args.n_trials,
        args.cv_splits,
        args.max_training_rows,
        args.max_knn_training_rows,
    )
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
