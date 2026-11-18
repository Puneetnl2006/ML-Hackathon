import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from fastapi import HTTPException
from pydantic import ValidationError
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler

from app import (
    InferenceService,
    PredictionRequest,
    RawTripPayload,
    app,
    health,
    predict_route,
)
from src.features import FEATURE_COLUMNS
from src.inference import model_probabilities, validate_feature_preprocessor


class InferenceTests(unittest.TestCase):
    def test_model_probabilities_validate_saved_feature_order(self):
        features = pd.DataFrame(
            np.ones((2, len(FEATURE_COLUMNS))),
            columns=FEATURE_COLUMNS,
        )
        model = DummyClassifier(strategy="constant", constant=1).fit(
            np.zeros((2, len(FEATURE_COLUMNS))), np.ones(2, dtype=int)
        )
        preprocessor = {
            "scaler": StandardScaler().fit(
                np.zeros((2, len(FEATURE_COLUMNS)))
            ),
            "feature_columns": FEATURE_COLUMNS,
        }

        probabilities = model_probabilities(model, features, preprocessor)

        np.testing.assert_array_equal(probabilities, [1 - 1e-15, 1 - 1e-15])
        invalid_preprocessor = dict(preprocessor, feature_columns=FEATURE_COLUMNS[::-1])
        with self.assertRaises(ValueError):
            validate_feature_preprocessor(invalid_preprocessor)

    def test_predict_route_runs_classifier_and_viterbi_without_returning_labels(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            project_root = Path(temporary_dir)
            output_dir = project_root / "outputs"
            trip_dir = project_root / "Train" / "00000001"
            output_dir.mkdir()
            trip_dir.mkdir(parents=True)
            (trip_dir / "00000001.track").write_text(
                "8.0001 49.0 1\n8.0009 49.0 2\n", encoding="utf-8"
            )
            (trip_dir / "00000001.nodes").write_text(
                "8.0 49.0\n8.001 49.0\n", encoding="utf-8"
            )
            (trip_dir / "00000001.arcs").write_text("0 1\n", encoding="utf-8")
            (trip_dir / "00000001.route").write_text("0\n", encoding="utf-8")

            model = DummyClassifier(strategy="constant", constant=1).fit(
                np.zeros((2, len(FEATURE_COLUMNS))), np.ones(2, dtype=int)
            )
            joblib.dump(model, output_dir / "logistic_regression_tuned.joblib")
            (output_dir / "optimization_best_params.json").write_text(
                json.dumps({"model": "logistic_regression"}),
                encoding="utf-8",
            )
            joblib.dump(
                {
                    "scaler": StandardScaler().fit(
                        np.zeros((2, len(FEATURE_COLUMNS)))
                    ),
                    "feature_columns": FEATURE_COLUMNS,
                },
                output_dir / "feature_scaler.joblib",
            )
            service = InferenceService(project_root)
            self.assertEqual(service.model_name, "logistic_regression")
            with self.assertRaises(HTTPException):
                service.resolve_trip("../outputs")

            with patch("app.get_service", return_value=service):
                response = predict_route(
                    PredictionRequest(trip_id="00000001")
                ).model_dump()
                raw_response = predict_route(
                    PredictionRequest(
                        trip=RawTripPayload(
                            track=[[8.0001, 49.0, 1], [8.0009, 49.0, 2]],
                            nodes=[[8.0, 49.0], [8.001, 49.0]],
                            arcs=[[0, 1]],
                        )
                    )
                ).model_dump()

        self.assertEqual(response["predicted_sequence"], [0, 0])
        self.assertEqual(response["candidate_count"], 2)
        self.assertNotIn("is_route_arc", response)
        self.assertEqual(raw_response["trip_id"], "raw")
        self.assertEqual(raw_response["predicted_sequence"], [0, 0])
        self.assertEqual(health(), {"status": "ok"})
        self.assertIn(
            "/predict_route",
            {route.path for route in app.routes},
        )
        with self.assertRaises(ValidationError):
            PredictionRequest()
        with self.assertRaises(ValidationError):
            PredictionRequest(
                trip_id="00000001",
                trip=RawTripPayload(
                    track=[[8.0, 49.0]],
                    nodes=[[8.0, 49.0]],
                    arcs=[[0, 0]],
                ),
            )
        with self.assertRaises(ValidationError):
            RawTripPayload(
                track=[[0.0, 0.0]],
                nodes=[[0.0, 0.0]],
                arcs=[[0, 1]],
            )


if __name__ == "__main__":
    unittest.main()
