# Local research compute and shared data plan

> Planning document. Proposed work is not a statement of shipped behavior; use the [plans index](README.md) for scope and status.

Status: planned work, October 2, 2026. Marimo and notebooks always run locally.
Shared storage is **GCS + Parquet with a verified local cache**. Live deployment,
bucket provisioning, and authentication setup are deferred. This document does not
implement storage, train models, upload data, or commit the workspace.

Application and repository name: **Armchair Labs** / **`armchair-labs`**, selected by
the user. Repository: `git@github.com:herman-rogers/armchair-labs.git`. The existing
Python package and CLI remain `patron` for compatibility.

## Objective and scope

Make it practical to explore many model families, tree configurations, feature
representations, predictive distributions, and ensembles across local machines.
Measure reproducible experiments per hour and predictive improvement per compute
budget. A finite search cannot exhaust every possibility.

The design has three parts:

| Part | Responsibility |
|---|---|
| Git | Source notebooks, Python helpers, tests, experiment configurations, dependency locks, documentation, and small dataset references |
| GCS | Versioned datasets, source evidence, completed predictions, model artifacts, and experiment results |
| Local SSD cache | Verified copies of required inputs, temporary training state, and results awaiting upload |

Use Parquet for tabular datasets, nested injury evidence, predictions, and metrics.
JSON manifests describe versions and provenance. Preserve existing NumPy matrices,
compressed raw captures, and native model artifacts where converting to Parquet
would be inappropriate or would change a sealed release. CSV is only an optional
spreadsheet export.

## Data layout

Proposed logical layout within GCS; actual bucket names are deferred:

```text
datasets/
  RELEASE_ID/
    manifest.json
    files/...                    # original logical paths and file formats
experiments/
  RUN_ID/
    attempts/ATTEMPT_ID/...
    manifest.json                # completed, verified artifact set
    predictions.parquet
    metrics.parquet
    config.json
    models/...
```

Dataset releases are immutable. Corrections, new weeks, and new injury evidence
produce a new release. Each experiment pins a release ID and manifest hash when it
starts. A latest pointer may help discover releases, but is resolved once and never
silently changes the inputs of an active experiment.

Keep the existing gold, prepared-feature, and injury archive bytes and sealed
manifests unchanged. A transport index maps logical paths to GCS objects, generations,
sizes, and SHA-256 hashes. It also enumerates the dependencies needed to reconstruct
the existing local layout. Refactor hard-coded data roots behind a resolver rather
than editing historical manifests to point at cloud paths.

For new large tables, partition by season/source where useful, avoiding many tiny
per-player files. Unchanged objects can be reused across releases when the manifest
references them explicitly. Do not rewrite existing releases just to adopt a new layout.

## Local cache behavior

1. Select a dataset release in the notebook or experiment configuration.
2. Read its manifest and determine which files this experiment needs.
3. Reuse matching local files; download only missing or invalid files.
4. Download to temporary paths, verify size/hash, then atomically move each verified
   file into place. Mark the release ready only after all required files validate.
5. Load cached Parquet into Polars and prepare the local model inputs.
6. Train and evaluate entirely from local storage, without network reads in the
   fitting loop. Repeated trials reuse the same verified inputs.
7. Save each completed run locally and explicitly publish it to GCS when desired.
   Another machine can download its results without retraining.

Use a configurable cache directory with release IDs and/or content hashes in the
paths. A complete cached release remains usable offline. Local locks coordinate
concurrent downloads on one machine. Cache cleanup must not evict files used by
active runs or delete results that have not been safely preserved.

GCS is object storage, not a shared local filesystem. Memory-mapped matrices and
active fit state stay on local disks. The initial adapter downloads files explicitly;
it does not depend on a bucket mount behaving like a POSIX filesystem.
[Cloud Storage objects](https://docs.cloud.google.com/storage/docs/objects).

## Publishing and reproducibility

Each machine writes unique run and attempt IDs. Upload artifacts first, verify them,
then publish the completed manifest. Readers discover runs through that manifest,
not through the presence of a folder. Failed uploads can resume without retraining.
A dataset publisher follows the same process for new data releases.

Use create-only writes for immutable artifacts and generation preconditions for any
shared pointer. GCS generation-match zero prevents overwriting a live object; it
does not make an entire multi-file upload atomic. Completion manifests provide the
application-level publication boundary.
[Request preconditions](https://docs.cloud.google.com/storage/docs/request-preconditions).

Every run records:

- Code commit and a source/patch snapshot if the workspace is dirty.
- Dataset release, input hashes, feature order, transformations, targets, and folds.
- Model configuration, seeds, requested/effective device, hardware, package and driver versions.
- Predictions, point/distribution metrics, calibration diagnostics, timings, and failures.
- Fitted models and preprocessing needed to reproduce predictions where supported.

Preserve source publication/knowledge timestamps separately from collection time.
Storage versions support reproducibility; they do not themselves prevent future-data
leakage. Raw and availability-adjusted forecasts remain distinguishable.

## Local execution and GPU work

The current Mac is an M1 Max with 64 GB RAM. The user also has an RTX 4090 machine;
its OS, driver, CPU/RAM, and storage need verification before implementation.

The latest RB run used about 4,600 training examples and 2,900 inputs across seven
stat models. The complete prepared matrix across positions/horizons is about 777 MiB.
The current [RB notebook](../../experiments/future_player_lab/notebooks/rb_stats.py)
trains sequentially with one CPU thread. With three calibration years and four
learning-history windows, the current implementation makes 63 fits across seven stats.

Planned work:

1. Provide an explicit portable research environment, including optional booster
   dependencies. Replace reliance on local environment links and copied packages.
2. Share training/evaluation helpers between notebook and CLI while keeping estimator
   construction and the training call easy to inspect from the notebook.
3. Save complete runs automatically so results survive local marimo restarts.
4. Add requested/effective device, CPU-thread, and worker-count settings. Fail clearly
   if an explicitly requested GPU is unavailable; do not silently fall back.
5. Benchmark XGBoost CPU versus CUDA on the 4090 machine, then CatBoost and LightGBM
   where compatible. Keep folds, data, and supported statistical settings matched.
6. Measure 1/2/4/8 CPU threads where available and bounded independent trials. Start
   with one fit per GPU and measure whether concurrency helps total throughput.
7. Compare cold/warm single fits and complete seven-stat runs, with at least three
   timing repetitions, memory use, score differences, and trials/hour.
8. Reuse verified inputs and compatible preprocessing. Avoid duplicate learning-curve
   fits and reserve expensive diagnostics for shortlisted candidates.

Library GPU paths: [XGBoost](https://xgboost.readthedocs.io/en/stable/gpu/),
[CatBoost](https://catboost.ai/docs/en/concepts/python-reference_catboostregressor_fit),
[LightGBM](https://lightgbm.readthedocs.io/en/stable/Installation-Guide.html).
CPU/GPU settings and results are not necessarily identical. Measure speedups on our
actual workload; hardware alone does not establish a faster complete experiment.

## Exploration program

Preserve the broader research objective while making execution efficient:

- Tree rounds/depth/leaves, learning rate, bins, regularization, and row/column sampling.
- Boosting families, random/extra trees, kernel/SVR controls, and neural challengers.
- Feature histories, learned interactions/representations, family ablations, and
  comparisons with and without dated market information.
- Direct stat totals versus availability/production and workload/efficiency models.
- Quantiles, residual distributions, suitable count models, and joint simulations.
- Ensembles across seeds, configurations, and families using chronological out-of-fold
  predictions for selection and blending.

Use staged searches with explicit trial/time budgets, resumable tasks, and recorded
failures and untested ranges. Preprocessing, feature discovery, calibration, and
ensemble selection learn only from earlier folds. Preserve unknown-label masks and
forecast-cutoff eligibility. Known full-period absences can constrain forecasts;
actual outcomes remain unchanged and unforeseen absences remain uncertain.

Evaluate MAE/RMSE, CRPS, interval calibration/width, and appropriate likelihood scores
when the model defines a probability mass/density. Marginal stat distributions do not
automatically define a joint fantasy-point distribution. Compare simple baselines,
original boosters, and same-date ECR on matched scoring/populations and prespecified
ranking metrics. Do not equate broad-pool correlation with draft value.

Because 2025 has already been inspected repeatedly, treat it as development evidence
for later changes. Use chronological multi-season evaluation and record genuinely
prospective forecasts for future validation. Faster searches must retain these rules.

## Implementation and Git sequence

| Step | Deliverable | Completion check |
|---|---|---|
| 1 | Inventory source/data and package local dependencies | Clean checkout opens notebooks and runs fixture checks on both machines |
| 2 | Define release/run manifests and local cache resolver | Missing/corrupt downloads recover; identical releases yield identical input hashes |
| 3 | Implement Parquet artifact loading/saving and GCS transport | Transfer behavior is tested with fixtures; completed runs reopen without fitting |
| 4 | Add local GPU/thread controls and benchmark | Representative timings and score comparisons, with effective backend recorded |
| 5 | Run broader versioned campaigns | Reproducible trial ledger and multi-season performance/compute report |
| Later | Provision storage and configure authentication | Live deployment/auth is a separate follow-up |

This repository has substantial uncommitted application and research changes.
Inventory them before staging. Commit source, tests, configurations, locks, and small
release references in focused changes; preserve large research artifacts outside Git.
Audit generated `data/research/` before changing ignore rules because it also contains
source snapshots and provenance. Do not blanket stage or delete existing work.

The `armchair-labs` repository includes the application and research source together,
preserving dependencies on the existing data helpers and configuration. Source history
is retained; generated research data remains local pending the GCS migration.

For storage implementation, use small deterministic fixtures and local test doubles
before live GCS access. Check interrupted downloads/uploads, checksum mismatches,
concurrent local cache access, unique run publication, and manifest completeness.
Bucket names, IAM, credentials, region, lifecycle configuration, and live transfers
are deferred. No infrastructure or authentication setup is part of this planning change.
