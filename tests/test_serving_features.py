"""Serving feature extraction must not load held-out route labels."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.features import extract_trip_features


class ServingFeatureTests(unittest.TestCase):
    def test_student_test_features_do_not_open_route_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            trip_dir = Path(temporary_directory) / "trip-1"
            trip_dir.mkdir()
            (trip_dir / "trip-1.track").write_text(
                "0.0005 0 1\n0.0015 0 2\n", encoding="utf-8"
            )
            (trip_dir / "trip-1.nodes").write_text(
                "0 0\n0.001 0\n0.002 0\n", encoding="utf-8"
            )
            (trip_dir / "trip-1.arcs").write_text("0 1\n1 2\n", encoding="utf-8")
            (trip_dir / "trip-1.route").mkdir()

            features = extract_trip_features(trip_dir, "student_test")

            self.assertFalse(features.empty)
            self.assertTrue(features["is_route_arc"].isna().all())


if __name__ == "__main__":
    unittest.main()
