"""Prepare a serving artifact from a tuned PathSnap classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

if __package__:
    from .features import FEATURE_COLUMNS
else:
    from features import FEATURE_COLUMNS


def prepare_artifacts(
    model_path: str | Path,
    output_dir: str | Path = "outputs/pathsnap_artifacts",
) -> tuple[Path, Path]:
    """Validate and copy the tuned pipeline and its serving metadata."""
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Tuned classifier not found: {model_path}")

    model = joblib.load(model_path)
    if not hasattr(model, "classes_") or 1 not in model.classes_:
        raise ValueError(
            "The tuned classifier must be fitted and contain the positive class 1."
        )
    if not hasattr(model, "predict_proba") and not hasattr(
        model, "decision_function"
    ):
        raise ValueError(
            "The tuned classifier must expose predict_proba or decision_function."
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    serving_model = output_dir / "model.joblib"
    manifest_path = output_dir / "manifest.json"
    joblib.dump(model, serving_model)
    manifest_path.write_text(
        json.dumps(
            {
                "model_source": str(model_path),
                "feature_columns": FEATURE_COLUMNS,
                "preprocessing": "the classifier's required preprocessing is serialized with it",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return serving_model, manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        help="Path to the tuned .joblib classifier produced by hyperparameter optimization.",
    )
    parser.add_argument("--output-dir", default="outputs/pathsnap_artifacts")
    args = parser.parse_args()
    serving_model, manifest = prepare_artifacts(args.model, args.output_dir)
    print(f"Serving classifier saved to {serving_model}")
    print(f"Serving manifest saved to {manifest}")


if __name__ == "__main__":
    main()
