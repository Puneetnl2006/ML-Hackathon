# PathSnap

PathSnap is a map-matching and route-reconstruction project. Trip data is stored
in `Train/` and `student_test/`, with one trip per subdirectory. Files are
whitespace-delimited and headerless:

- `.track`: longitude, latitude, and sample time/index (two-column tracks are
  also accepted; row order is used as the sample time).
- `.nodes`: longitude and latitude.
- `.arcs`: zero-based start-node and end-node indices.
- `.route`: ordered arc-row indices.

## Set up

Use Python 3.12 or another version supported by the packages in
`requirements.txt`.

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
`outputs/feature_scaler.joblib`. Candidates are spatially matched to road arcs
and include segment/center distance, heading alignment, speed, and
acceleration (up to 10 nearest in-radius candidates per observation). The
scaler is fitted exclusively on candidate rows from `Train/` and then applied
to both partitions.

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