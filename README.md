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
membership is available for training trips only; student-test route files are
not read during feature extraction or inference.

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

## Evaluate HMM route reconstruction

The decoder uses directed `.arcs` connectivity and shortest-path network
distances to find a likely candidate-arc sequence with Viterbi. Missing
candidates and large time or spatial gaps split a track into independent
decoding runs. Disconnected transitions receive a strong penalty rather than
causing decoding to fail.

Generate candidate features, fit the evaluation classifier on training trips,
and compare independent top-1 predictions with HMM decoding:

```bash
python src/features.py \
  --train-dir Train \
  --test-dir student_test \
  --output-dir outputs
python src/evaluate_decoder.py \
  --candidate-features outputs/candidate_features.csv \
  --train-dir Train \
  --test-dir student_test \
  --output-dir decoder_outputs
```

The evaluation holds out complete `Train/` trips (20% by default) and fits a
logistic-regression candidate classifier on the remaining trips. Accuracy is
the fraction of selected arcs that appear in the held-out trip's `.route`;
`student_test/` has no route labels, so only its route-continuity results are
available. `decoder_outputs/decoder_metrics.csv` contains per-trip and
aggregate pointwise accuracy, route continuity, and invalid-transition counts.
`decoder_outputs/decoder_visualization.csv` contains GPS points, both predicted
arc IDs and geometries, and route-membership labels for mapping or plotting.
Use `--validation-fraction` and `--seed` to control the trip-level validation
split.

## GitHub Actions

The workflow in `.github/workflows/run_pipeline.yml` runs on every push and can
also be started from the Actions tab with **Run workflow**. Download the
`pathsnap-eda-and-features` artifact from a workflow run to inspect the CSV
reports, fitted transformer, and interactive map.

## Two-laptop deployment: FastAPI and evaluator UI

The backend runs on Laptop 1 with the serving classifier and reserved trips.
The Streamlit evaluator runs on Laptop 2 and sends requests to Laptop 1. The
API reads only `.track`, `.nodes`, and `.arcs` for reserved trips; it does not
load or return `.route` evaluation labels. When `PATHSNAP_API_TOKEN` is set,
the `/trips` and `/predict_route` endpoints require the same bearer token.

### Prepare the serving classifier on Laptop 1

Use the tuned classifier produced by the optimization workflow. For example,
after running feature extraction, baseline comparison, and hyperparameter
optimization as described above:

```bash
python src/prepare_artifacts.py \
  --model outputs/<tuned-model-name>_tuned.joblib
```

This validates and copies the fitted classifier pipeline (including its
preprocessing) to `outputs/pathsnap_artifacts/model.joblib`. Keep this artifact
private to the backend machine; it is not served by an API route. The HMM
settings are loaded from `config/hmm_decoder.json`.

Install backend dependencies on Laptop 1:

```bash
python -m pip install -r requirements-api.txt
```

Start the API from the repository root:

```bash
export PATHSNAP_API_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
printf 'Copy this token securely to Laptop 2: %s\n' "$PATHSNAP_API_TOKEN"
export PATHSNAP_DATA_DIR=student_test
uvicorn src.app:app --host 0.0.0.0 --port 8000
```

Copy the token securely to Laptop 2. Keep it set on Laptop 1 for the life of
the API process. The public `/health` endpoint can be used without a token.
For a GitHub Codespace backend, make port 8000 reachable from Laptop 2:

```bash
gh codespace ports visibility 8000:public -c "$CODESPACE_NAME"
```

The Codespaces API address is
`https://<CODESPACE-NAME>-8000.app.github.dev`. The trip-list and prediction
endpoints require the bearer token; `/health` remains public for availability
checks. Do not share the token or commit it to the repository. If both
computers are on the same trusted LAN and Laptop 1 is not a Codespace, use
Laptop 1's LAN IP instead and allow TCP port 8000 only on that network.

The terminal reports startup readiness, each request's status and duration,
and prediction counts. The backend exposes `GET /health`, `GET /trips`, and
`POST /predict_route`. The prediction endpoint accepts either
`{"trip_id":"00000000"}` or raw `observations`, `nodes`, and `arcs` arrays.
Raw node and arc IDs may be omitted to use their array positions. A raw request
uses GPS longitude/latitude pairs, optional sample times, node coordinates,
and directed arcs referencing node IDs:

```json
{
  "observations": [{"longitude": -73.98, "latitude": 40.75, "sample_time": 1}],
  "nodes": [
    {"node_id": 0, "longitude": -73.98, "latitude": 40.75},
    {"node_id": 1, "longitude": -73.979, "latitude": 40.75}
  ],
  "arcs": [{"arc_id": 0, "from_node": 0, "to_node": 1}]
}
```

### Start the evaluator on Laptop 2

Install UI dependencies on Laptop 2:

```bash
python -m pip install -r requirements-ui.txt
```

Set Laptop 1's reachable local-network IP address and start Streamlit:

```bash
export PATHSNAP_API_URL=https://<CODESPACE-NAME>-8000.app.github.dev
export PATHSNAP_API_TOKEN='<copy the same token from Laptop 1>'
streamlit run src/ui.py --server.address 0.0.0.0 --server.port 8501
```

For a backend running directly on the trusted LAN, set
`PATHSNAP_API_URL=http://<LAPTOP-1-LAN-IP>:8000` instead. Open
`http://<LAPTOP-2-LAN-IP>:8501` in the evaluator's browser and select a
reserved trip. Its prediction and map load automatically, with GPS observations
overlaid on the predicted road-segment sequence. The map uses Esri World Street
Map tiles rather than OpenStreetMap's volunteer-run tile servers. No ground-truth
route or accuracy metrics are sent to the UI.

Allow TCP ports 8000 (Laptop 1) and 8501 (Laptop 2) only on the trusted local
network when using the LAN setup.
