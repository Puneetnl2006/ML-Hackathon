"""Focused tests for the map-matching API prediction contract."""

from __future__ import annotations

import unittest

import numpy as np
from pydantic import ValidationError

from src.app import (
    PredictRouteRequest,
    RoutePredictor,
    _has_valid_bearer_token,
)


class FixedProbabilityModel:
    classes_ = np.asarray([0, 1])

    def predict_proba(self, features):
        return np.tile([0.1, 0.9], (len(features), 1))


class RouteApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.predictor = RoutePredictor.__new__(RoutePredictor)
        self.predictor.model = FixedProbabilityModel()
        self.predictor.decoder_config = {}
        self.predictor.trip_dirs = {}

    def test_raw_trip_returns_route_and_observations_without_labels(self) -> None:
        request = PredictRouteRequest(
            observations=[
                {"longitude": 0.0005, "latitude": 0.0, "sample_time": 1.0},
                {"longitude": 0.0015, "latitude": 0.0, "sample_time": 2.0},
            ],
            nodes=[
                {"longitude": 0.0, "latitude": 0.0},
                {"longitude": 0.001, "latitude": 0.0},
                {"longitude": 0.002, "latitude": 0.0},
            ],
            arcs=[{"from_node": 0, "to_node": 1}, {"from_node": 1, "to_node": 2}],
        )

        result = self.predictor.predict(request)

        self.assertEqual(result.observation_count, 2)
        self.assertTrue(all(point.arc_id is not None for point in result.observations))
        self.assertTrue(all(point.geometry is not None for point in result.observations))
        self.assertNotIn("route", result.model_dump())
        self.assertNotIn("ground_truth", result.model_dump())

    def test_request_must_choose_trip_id_or_raw_trip(self) -> None:
        with self.assertRaises(ValidationError):
            PredictRouteRequest(
                trip_id="00000000",
                observations=[{"longitude": 0.0, "latitude": 0.0}],
            )

    def test_bearer_token_is_required_when_configured(self) -> None:
        self.assertTrue(_has_valid_bearer_token("demo-secret", "Bearer demo-secret"))
        self.assertFalse(_has_valid_bearer_token("demo-secret", "Bearer wrong"))
        self.assertFalse(_has_valid_bearer_token("demo-secret", "demo-secret"))
        self.assertFalse(_has_valid_bearer_token("demo-secret", "Bearer wrong-🔒"))


if __name__ == "__main__":
    unittest.main()
