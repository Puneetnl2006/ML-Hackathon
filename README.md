# PathSnap

PathSnap matches GPS observations to candidate road arcs. The repository
contains 80 labeled trips in `Train/` and 10 held-out trips in `student_test/`.
Each trip directory contains headerless, whitespace-delimited `.track`,
`.nodes`, `.arcs`, and `.route` files.

## Set up

Use Python 3.12 or another version supported by `requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Extract candidate features

```bash
python src/features.py \
  --train-dir Train \
  --test-dir student_test \
  --output-dir outputs
```

This writes `outputs/candidate_features.csv`. Candidate route membership is
available for training trips only; student-test labels are not used.

## Train and compare baseline models

```bash
python src/train_baselines.py \
  --features-csv outputs/candidate_features.csv \
  --output-dir outputs
```

The script compares logistic regression, random forest, histogram gradient
boosting, k-nearest neighbors, and a linear support-vector machine. It uses
five-fold `GroupKFold` cross-validation, keeping every trip in exactly one
validation fold, then fits each model on training trips and scores the
student-test candidates. Each model fit is capped at 250,000 candidate rows
(50,000 for k-nearest neighbors) to keep the baseline run tractable; sampling
preserves at least one row per training trip.

Outputs include:

- `model_comparison.csv`: out-of-fold metrics and student-test inference time.
- `cross_validation_predictions.csv`: out-of-fold candidate predictions.
- `student_test_predictions.csv`: candidate probabilities and each observation's
  selected top candidate.
- `confusion_matrices.csv` and `confusion_matrices.png`: out-of-fold
  candidate-level confusion matrices.
- One `.joblib` model per baseline, fitted on the training trips.

Observation-level top-1 accuracy selects the highest-scored candidate for each
observation and checks whether it belongs to that trip's labeled route.
Macro-F1 and log loss are candidate-level metrics; candidate miss rate is the
fraction of observations with no route candidate. The student-test route labels
are withheld, so supervised metrics and confusion matrices are reported for
out-of-fold training predictions only.