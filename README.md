# PathSnap

PathSnap matches GPS observations to candidate road arcs. Trip data is stored
in `Train/` and `student_test/`, with one trip per subdirectory. Files are
whitespace-delimited and headerless:

- `.track`: longitude, latitude, and sample time/index (two-column tracks are
  also accepted; row order is used as the sample time).
- `.nodes`: longitude and latitude.
- `.arcs`: zero-based start-node and end-node indices.
- `.route`: ordered arc-row indices.

## Set up

Use Python 3.12 or another version supported by `requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Explore the datasets

From the repository root:

```bash
python src/explore_data.py \
  --train-dir Train \
  --test-dir student_test \
  --output-dir outputs \
  --map-output-dir eda_outputs
```

The script writes per-trip file sizes, observation counts, sampling interval
statistics, and dataset totals to `outputs/dataset_inventory.csv` and
`outputs/dataset_summary.csv`. It also saves an interactive Folium map of the
first available trip to `eda_outputs/`; select another trip with
`--sample-trip TRIP_ID`.

## Generate route comparison maps

Generate sample maps for the first training trip and first student-testing trip
discovered beneath the repository root:

```bash
python src/visualize.py
```

The script scans the workspace for PathSnap trip files and saves interactive
maps under `outputs/maps/`. Select a particular trip or generate more samples
with:

```bash
python src/visualize.py --trip-id 00000001
python src/visualize.py --max-maps 5
```

Maps include toggleable layers for GPS observations, nearby road nodes and
arcs, the reference route when available, and separate classifier and HMM
prediction layers when their outputs are provided:

```bash
python src/visualize.py \
  --trip-id 00000007 \
  --classifier-predictions outputs/student_test_predictions.csv \
  --model random_forest \
  --hmm-predictions outputs/hmm_decoded_routes.csv
```

Classifier CSVs can use the `selected_top1` and `model` columns produced by
`src/train_baselines.py`. HMM CSVs can provide predicted arc IDs in
`predicted_arc_id`, `decoded_arc_id`, `route_arc_id`, or `arc_id`; a plain
whitespace-delimited `.route` file is also accepted. Use `--max-maps` to set the
maximum number of sample maps (default: two, one per dataset partition). The
HTML maps can be opened in a browser for visual inspection or technical-report
screenshots.

## Build candidate features

After exploring the datasets, extract candidate-road features with:

```bash
python src/features.py \
  --train-dir Train \
  --test-dir student_test \
  --output-dir outputs
```

Feature extraction writes raw and training-scaled candidate features to
`outputs/candidate_features.csv` and the fitted transformer to
`outputs/feature_scaler.joblib`. The scaler is fitted exclusively on candidate
rows from `Train/` and then applied to both partitions. Candidate route
membership is available for training trips only; student-test labels are not
used.

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
fraction of observations with no route candidate. Baseline evaluation uses
out-of-fold training predictions only.

## Optimize the best baseline

After running the baseline comparison, tune its highest macro-F1 model with
Optuna's Bayesian TPE sampler:

```bash
python src/optimize_hyperparameters.py \
  --features-csv outputs/candidate_features.csv \
  --comparison-csv outputs/model_comparison.csv \
  --test-dir student_test \
  --output-dir outputs \
  --n-trials 25
```

Each trial uses grouped cross-validation by training trip. The search space
targets the selected classifier's main regularization, capacity, or neighborhood
parameters. The final tuned and untuned fits are compared on the student-test
routes using observation-level top-1 accuracy, candidate-level macro-F1, and
log loss. Student-test route files are used only for this final comparison, not
for model selection or optimization. Outputs include
`optimization_comparison.csv`, `optimization_student_test_predictions.csv`,
`optimization_trials.csv`, and
`optimization_best_params.json`. Increase `--n-trials` for a broader search;
`--cv-splits` controls the number of GroupKFold folds.

## GitHub Actions

The workflow in `.github/workflows/run_pipeline.yml` runs on every push and can
also be started from the Actions tab with **Run workflow**. Download the
`pathsnap-eda-and-features` artifact from a workflow run to inspect the CSV
reports, fitted transformer, and interactive map.
