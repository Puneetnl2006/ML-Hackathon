import unittest
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from src.features import FEATURE_COLUMNS
from src.train_baselines import build_models, evaluate_predictions, run_experiment


class BaselineEvaluationTests(unittest.TestCase):
    def test_metrics_rank_the_best_candidate_per_observation(self):
        frame = pd.DataFrame(
            {
                "trip_id": ["a", "a", "a", "b", "b"],
                "observation_index": [0, 0, 1, 0, 0],
            }
        )
        labels = np.array([1, 0, 0, 0, 1])
        scores = np.array([0.8, 0.2, 0.7, 0.1, 0.9])

        result = evaluate_predictions(frame, labels, scores)

        self.assertEqual(result["observations"], 3)
        self.assertEqual(result["candidate_rows"], 5)
        self.assertAlmostEqual(result["observation_top1_accuracy"], 2 / 3)
        self.assertAlmostEqual(result["candidate_miss_rate"], 1 / 3)
        self.assertEqual(result["confusion_matrix"].shape, (2, 2))
        self.assertTrue(np.isfinite(result["log_loss"]))

    def test_five_distinct_baseline_models_are_defined(self):
        models = build_models()

        self.assertEqual(
            set(models),
            {
                "logistic_regression",
                "random_forest",
                "hist_gradient_boosting",
                "knn",
                "linear_svm",
            },
        )

    def test_grouped_experiment_writes_comparison_and_test_predictions(self):
        rows = []
        for dataset, trip_count in (("train", 6), ("student_test", 2)):
            for trip_number in range(trip_count):
                for observation in range(4):
                    for candidate in range(2):
                        row = {
                            "dataset": dataset,
                            "trip_id": f"{dataset}_{trip_number}",
                            "observation_index": observation,
                            "arc_id": candidate,
                            "is_route_arc": (
                                float(candidate == observation % 2)
                                if dataset == "train"
                                else np.nan
                            ),
                        }
                        row.update(
                            {
                                column: float(
                                    (trip_number + observation + candidate + offset) % 3
                                )
                                for offset, column in enumerate(FEATURE_COLUMNS)
                            }
                        )
                        rows.append(row)

        with TemporaryDirectory() as output_dir:
            comparison = run_experiment(
                pd.DataFrame(rows),
                output_dir,
                cv_splits=2,
                max_training_rows=16,
                max_knn_training_rows=16,
            )
            cv_rows = comparison[
                comparison["evaluation"] == "train_cross_validation"
            ]
            test_predictions = pd.read_csv(
                f"{output_dir}/student_test_predictions.csv"
            )

        self.assertEqual(len(cv_rows), 5)
        self.assertEqual(len(test_predictions), 2 * 4 * 2 * 5)
        self.assertTrue(cv_rows["observation_top1_accuracy"].between(0, 1).all())


if __name__ == "__main__":
    unittest.main()
