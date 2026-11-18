import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

import pandas as pd

from src.data_utils import load_trip
from src.visualize import (
    _arc_geometries,
    generate_sample_maps,
    load_predicted_arc_ids,
)


class VisualizationTests(unittest.TestCase):
    def test_discovers_trip_creates_map_layers_and_selects_classifier_route(self):
        with TemporaryDirectory() as temporary_dir:
            workspace = Path(temporary_dir)
            train_trip_dir = workspace / "Train" / "00000001"
            trip_dir = workspace / "student_test" / "00000007"
            train_trip_dir.mkdir(parents=True)
            trip_dir.mkdir(parents=True)
            for sample_dir, sample_id in (
                (train_trip_dir, "00000001"),
                (trip_dir, "00000007"),
            ):
                (sample_dir / f"{sample_id}.track").write_text(
                    "8.0 49.0 1\n8.001 49.001 2\n", encoding="utf-8"
                )
                (sample_dir / f"{sample_id}.nodes").write_text(
                    "8.0 49.0\n8.001 49.001\n8.002 49.002\n",
                    encoding="utf-8",
                )
                (sample_dir / f"{sample_id}.arcs").write_text(
                    "0 1\n1 2\n", encoding="utf-8"
                )
                (sample_dir / f"{sample_id}.route").write_text(
                    "1\n0\n", encoding="utf-8"
                )
            predictions_path = workspace / "predictions.csv"
            pd.DataFrame(
                [
                    {
                        "trip_id": "7",
                        "observation_index": 0,
                        "arc_id": 1,
                        "model": "random_forest",
                        "selected_top1": True,
                    },
                    {
                        "trip_id": "7",
                        "observation_index": 0,
                        "arc_id": 0,
                        "model": "logistic_regression",
                        "selected_top1": True,
                    },
                    {
                        "trip_id": "7",
                        "observation_index": 0,
                        "arc_id": 0,
                        "model": "random_forest",
                        "selected_top1": False,
                    },
                ]
            ).to_csv(predictions_path, index=False)
            decoded_path = workspace / "decoded.csv"
            pd.DataFrame(
                [
                    {"trip_id": "7", "decoded_arc_id": 0},
                    {"trip_id": "7", "decoded_arc_id": 1},
                ]
            ).to_csv(decoded_path, index=False)
            decoded_route_path = workspace / "hmm_decoded.route"
            decoded_route_path.write_text("0\n1\n", encoding="utf-8")

            predicted_arc_ids = load_predicted_arc_ids(
                predictions_path, "00000007", model="random_forest"
            )
            decoded_arc_ids = load_predicted_arc_ids(decoded_path, "00000007")
            decoded_route_arc_ids = load_predicted_arc_ids(
                decoded_route_path, "00000007"
            )
            reference_geometries = _arc_geometries(
                load_trip(trip_dir), [1, 0]
            )
            maps = generate_sample_maps(
                workspace=workspace,
                predictions_path=predictions_path,
                model="random_forest",
                hmm_predictions_path=decoded_path,
            )
            html = next(
                path.read_text(encoding="utf-8")
                for path in maps
                if path.name.startswith("student_test_")
            )

        self.assertEqual(predicted_arc_ids, [1])
        self.assertEqual(decoded_arc_ids, [0, 1])
        self.assertEqual(decoded_route_arc_ids, [0, 1])
        self.assertEqual(
            reference_geometries,
            [
                [[49.001, 8.001], [49.002, 8.002]],
                [[49.0, 8.0], [49.001, 8.001]],
            ],
        )
        self.assertEqual(len(maps), 2)
        self.assertEqual({path.parent.name for path in maps}, {"maps"})
        self.assertEqual(
            {path.name for path in maps},
            {"Train_00000001_map.html", "student_test_00000007_map.html"},
        )
        self.assertIn("Original GPS observations", html)
        self.assertIn("Relevant road network", html)
        self.assertIn("Reference route", html)
        self.assertIn("Classifier-predicted route", html)
        self.assertIn("HMM-decoded route", html)
        self.assertIn("layer_control", html)


if __name__ == "__main__":
    unittest.main()
