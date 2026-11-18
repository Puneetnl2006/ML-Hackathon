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

## GitHub Actions

The workflow in `.github/workflows/run_pipeline.yml` runs on every push and can
also be started from the Actions tab with **Run workflow**. Download the
`pathsnap-eda` artifact from a workflow run to inspect the CSV reports and
interactive map.