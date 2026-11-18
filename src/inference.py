"""Model-scoring helpers for PathSnap inference."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from .features import FEATURE_COLUMNS
from .train_baselines import positive_class_probabilities


def validate_feature_preprocessor(preprocessor: object | None) -> None:
    """Validate the saved feature schema without double-scaling model inputs.

    The baseline estimators are fitted on raw ``FEATURE_COLUMNS``; estimators
    that need standardization carry their scaler inside the saved model
    pipeline. The separate feature artifact is used to verify feature ordering.
    """
    if preprocessor is None:
        return
    if not isinstance(preprocessor, Mapping):
        raise ValueError("Feature preprocessor must contain feature metadata.")
    if list(preprocessor.get("feature_columns", [])) != FEATURE_COLUMNS:
        raise ValueError("Saved feature preprocessor has an incompatible schema.")
    if not hasattr(preprocessor.get("scaler"), "transform"):
        raise ValueError("Saved feature preprocessor has no fitted scaler.")


def model_probabilities(
    model: object,
    candidate_features: pd.DataFrame,
    preprocessor: object | None = None,
) -> np.ndarray:
    """Score candidates using the trained model and verify its feature schema."""
    validate_feature_preprocessor(preprocessor)
    missing = sorted(set(FEATURE_COLUMNS) - set(candidate_features.columns))
    if missing:
        raise ValueError(f"Candidate features are missing columns: {missing}")
    probabilities = positive_class_probabilities(
        model, candidate_features[FEATURE_COLUMNS]
    )
    probabilities = np.asarray(probabilities, dtype=float)
    if len(probabilities) != len(candidate_features) or not np.isfinite(
        probabilities
    ).all():
        raise ValueError("The classifier returned invalid candidate probabilities.")
    return np.clip(probabilities, 1e-15, 1 - 1e-15)
