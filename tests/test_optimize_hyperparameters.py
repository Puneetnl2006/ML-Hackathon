import json
import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np
import pandas as pd

from src.features import FEATURE_COLUMNS
from src.optimize_hyperparameters import run_optimization, select_best_baseline


class HyperparameterOptimizationTests(unittest.TestCase):
    def test_selects_best_cross_validated_baseline(self):
        comparison = pd.DataFrame(
            [
                {
                    "model": "random_forest",
                    "evaluation": "train_cross_validation",
                    "macro_f1_score": 0.6,
                },
                {
                    "model": "logistic_regression",
                    "evaluation": "student_test_inference",
                    "macro_f1_score": np.nan,
                },
                {
                    "model": "logistic_regression",
                    "evaluation": "train_cross_validation",
                    "macro_f1_score": 0.7,
                },
            ]
        )

        self.assertEqual(select_best_baseline(comparison), "logistic_regression")

    def test_runs_grouped_tuning_and_scores_held_out_route_labels(self):
        rows = []
        for dataset, trip_count in (("train", 6), ("student_test", 2)):
            for trip_number in range(trip_count):
                trip_id = f"{dataset}_{trip_number}"
                for observation in range(4):
                    for candidate in range(2):
                        row = {
                            "dataset": dataset,
                            "trip_id": trip_id,
                            "observation_index": observation,
                            "arc_id": candidate,
                            "is_route_arc": (
                                float(candidate == 1) if dataset == "train" else np.nan
                            ),
                        }
                        row.update(
                            {
                                column: float(
                                    candidate + observation * 0.1 + offset * 0.01
                                )
                                for offset, column in enumerate(FEATURE_COLUMNS)
                            }
                        )
                        rows.append(row)

        comparison = pd.DataFrame(
            [
                {
                    "model": "logistic_regression",
                    "evaluation": "train_cross_validation",
                    "macro_f1_score": 0.8,
                }
            ]
        )
        with TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            test_dir = root / "student_test"
            for trip_number in range(2):
                trip_dir = test_dir / f"student_test_{trip_number}"
                trip_dir.mkdir(parents=True)
                (trip_dir / f"{trip_dir.name}.route").write_text("1\n", encoding="utf-8")
            output_dir = root / "outputs"

            result = run_optimization(
                pd.DataFrame(rows),
                comparison,
                output_dir=output_dir,
                test_dir=test_dir,
                n_trials=2,
                cv_splits=2,
                max_training_rows=32,
            )

            predictions = pd.read_csv(
                output_dir / "optimization_student_test_predictions.csv"
            )
            best_params = json.loads(
                (output_dir / "optimization_best_params.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(result["configuration"].tolist(), ["untuned", "tuned"])
        self.assertTrue(result["macro_f1_score"].between(0, 1).all())
        self.assertTrue(np.isfinite(result["log_loss"]).all())
        self.assertEqual(len(predictions), 16)
        self.assertIn("GroupKFold", best_params["cv_strategy"])
        self.assertTrue(predictions["tuned_selected_top1"].any())


if __name__ == "__main__":
    unittest.main()
