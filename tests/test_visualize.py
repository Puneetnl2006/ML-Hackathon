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
            trip_dir = workspace / "student_test" / "00000007"
            trip_dir.mkdir(parents=True)
            (trip_dir / "00000007.track").write_text(
                "8.0 49.0 1\n8.001 49.001 2\n", encoding="utf-8"
            )
            (trip_dir / "00000007.nodes").write_text(
                "8.0 49.0\n8.001 49.001\n8.002 49.002\n",
                encoding="utf-8",
            )
            (trip_dir / "00000007.arcs").write_text(
                "0 1\n1 2\n", encoding="utf-8"
            )
            (trip_dir / "00000007.route").write_text("1\n0\n", encoding="utf-8")
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

            predicted_arc_ids = load_predicted_arc_ids(
                predictions_path, "00000007", model="random_forest"
            )
            decoded_arc_ids = load_predicted_arc_ids(decoded_path, "00000007")
            reference_geometries = _arc_geometries(
                load_trip(trip_dir), [1, 0]
            )
            maps = generate_sample_maps(
                workspace=workspace,
                trip_id="00000007",
                predictions_path=predictions_path,
                model="random_forest",
            )
            html = maps[0].read_text(encoding="utf-8")

        self.assertEqual(predicted_arc_ids, [1])
        self.assertEqual(decoded_arc_ids, [0, 1])
        self.assertEqual(
            reference_geometries,
            [
                [[49.001, 8.001], [49.002, 8.002]],
                [[49.0, 8.0], [49.001, 8.001]],
            ],
        )
        self.assertEqual(len(maps), 1)
        self.assertEqual(maps[0].parent.name, "maps")
        self.assertIn("Original GPS observations", html)
        self.assertIn("Relevant road network", html)
        self.assertIn("Reference route", html)
        self.assertIn("Predicted route", html)
        self.assertIn("layer_control", html)


if __name__ == "__main__":
    unittest.main()
