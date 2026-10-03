# Shared research data

This runbook is for larger source/research archives needed to reproduce builds.
For accepted SQL tables use [the data pipeline](tables.md) and `just data-fetch`.
New source archives include validated batches when present; they exclude the
active query catalog. Only `engine tables fetch` selects queryable tables.


For named DuckDB tables and selected research archives, use the separate
[analytical-table workflow](tables.md). It reuses this transport and cache while
keeping small query-only bundles separate from the broader notebook inputs below.

Notebooks and training run locally. Shared data uses a private GCS bucket, immutable
release manifests, and a verified local SSD cache. Git stores source, cleaned
notebooks, profile definitions, and small release references. It does not store
research tables, saved model binaries, cached sessions, or generated notebook exports.

The transport is implemented in `engine.data.shared`. The Google Cloud project is
`armchair-labs` (project number `1062502564356`), and its private Standard-storage
bucket is `gs://armchair-labs-data` in `us-east1`. Shared objects use the
`armchair-labs/` prefix. Uniform bucket-level access and public access prevention
are enabled. Billing is linked to the existing open Definetti account.

A release is available only after its reference has been published to
`data/releases/`; the presence of local staging files does not mean an upload completed.

## Another machine: download once

Install Python 3.12+, uv, and Google Cloud CLI, then from the checkout:

```sh
uv sync --extra shared-data
gcloud auth login
just source-fetch notebooks
just source-verify notebooks
bash experiments/future_player_lab/notebooks/launch.sh
```

The commands use `data/releases/current.json`, a small Git-versioned reference to
one immutable remote manifest with its SHA-256 and GCS generation. To use a
particular release:

```sh
just source-fetch notebooks --reference data/releases/RELEASE.json
```

Available download profiles (each larger profile includes the previous one):

| Profile | Contents |
|---|---|
| `notebooks` | Preserved raw objects, table batches, enriched releases, legacy source archives, auxiliary source catalog and its selected datasets, prepared feature matrices, injury archive, original benchmark inputs, saved tree/ensemble comparisons and distribution results |
| `rebuild` | Notebook inputs plus accepted historical source collections, captured college sources, source evidence, and captured current-season reports/snapshots |
| `archive` | Rebuild inputs plus the broader research and experiment history, provider cache and saved catalog references |

`archive` includes failed and interrupted runs for historical evidence. Downloading
or uploading one does not certify it as a valid model. The profiles are defined in
[`data/releases/profiles.json`](../../data/releases/profiles.json). They describe
inputs captured at snapshot time; refresh the profile/release when notebooks gain
new dependencies. The auxiliary source catalog is checked against the complete
gold → enriched → raw manifest chain before a snapshot can be published. Every
profile containing that catalog must include its declared dependencies.

The cache defaults to `~/.cache/armchair-labs`. Set `ARMCHAIR_DATA_CACHE` to another
SSD directory if desired. Objects are SHA-256 addressed and stored once per cache;
working copies are materialized at the existing repository paths. Allow space for
both cache and working copies. Copies isolate editable files from shared cache bytes.
Mac/Linux are supported; use WSL on the Windows GPU machine for the existing shell
launchers and build pipeline.

```sh
# Restore deleted working copies using cached bytes, without GCS/network access:
just source-fetch notebooks --offline
# Independently check the real notebook readers, without fitting models:
uv run python -m research.verify_shared_inputs
```

The reader pins GCS generations, downloads into temporary files, verifies sizes and
SHA-256, then atomically installs each file. A readiness record is written only
after the entire selected profile verifies. Fetch can resume after interruption.
If an existing local file differs, fetch stops; use a fresh checkout/destination
rather than overwriting local research. A cache lock coordinates concurrent fetches.
An offline fetch requires the previously cached release manifest as well as objects.

Google application-default credentials are supported. If absent, the adapter uses
the existing `gcloud auth login` session and refreshes tokens through gcloud. No
service-account key or bearer token is stored in the repository or data release.

## Publisher: create and upload a release

First clean/check source notebooks. Marimo `.py` notebooks already contain source
only. The cleaner removes outputs, execution counts/timing and widget state from
versionable `.ipynb` files, preserving code and Markdown. It does not delete saved
research predictions or touch a running marimo session.

```sh
just notebooks-clean
just notebooks-check
```

The initial bucket is already provisioned. To provision a separate environment
later, create a dedicated bucket in its project (one-time operation):

```sh
gcloud storage buckets create gs://BUCKET \
  --project=PROJECT --location=us-east1 --default-storage-class=STANDARD \
  --uniform-bucket-level-access --public-access-prevention
```

Use a globally unique bucket name. Bucket IAM grants teammates access; notebook
hosting and application login are outside this workflow. Uniform bucket-level
access and public access prevention keep access controlled at the bucket level.
[Google documentation](https://docs.cloud.google.com/storage/docs/public-access-prevention).

Stop writers to the selected inputs before freezing a release:

```sh
just source-snapshot RELEASE
uv run python -m engine.data.shared inventory \
  --manifest data/.shared/staging/RELEASE/manifest.json
just source-upload RELEASE gs://armchair-labs-data/armchair-labs
```

Snapshot copies selected bytes into the content-addressed cache, detects files
changed during capture, and emits an immutable local staging manifest. It excludes
sessions, temporary files, environment/credential files, and fresh bytecode caches.
A few historical seals explicitly include bytecode: those exact declared files are
retained to preserve old manifest verification. Existing sealed Parquet, CSV, JSON,
NumPy, source captures and source/environment snapshots keep their original bytes.
New tables should use Parquet; no legacy conversion is performed during transport.

Uploads invoke `gcloud storage cp --no-clobber` in up to four parallel workers.
The CLI applies a generation-zero precondition for no-clobber; an additional
`--if-generation-match` flag must not be combined with it. Each upload retains
SHA-256 metadata, then the adapter checks remote size, CRC32C, SHA-256 metadata,
and generation. Matching existing objects are reused; collisions are rejected.
Composite uploads are disabled to avoid temporary cloud components; ordinary
resumable uploads retain retry support. The SDK handles metadata checks and
generation-pinned downloads. All data objects are
uploaded before the completed remote manifest; an interrupted upload has no new
published release. Retry the same command to resume. GCS generations and the final
manifest hash are recorded in `data/releases/RELEASE.json`.
[Google CLI copy documentation](https://docs.cloud.google.com/sdk/gcloud/reference/storage/cp).

After verifying a cloud download in a clean destination, select the default release:

```sh
cp data/releases/RELEASE.json data/releases/current.json
```

Commit the small references, profile changes, source, tests and docs. Never add
`data/.shared`, raw/gold data, local caches, or experiment `runs/` directories to Git.
Remote publication is immutable; changing the default Git reference selects a new
release without changing any existing one. No automated remote deletion or cache
eviction is configured. Source data is not deleted after upload.

The [2026-10-03 GCS audit](../../data/releases/gcs-audit-20261003.json) checked all
six published releases against the bucket inventory, including object sizes,
SHA-256 metadata and pinned generations. All references passed. One unreferenced
1,729-byte source catalog from the unpublished `bootstrap_20261002_r3` attempt was
removed with a generation precondition; published historical releases were kept.
The selected 31-table release matches the local query catalog. This metadata audit
complements the independent download verification below; it did not redownload
every payload.

## Rebuilding and refreshing

Downloading a preserved release reproduces captured inputs. Re-fetching provider
APIs collects their current revisions and is a different operation.

After `just source-fetch rebuild`, inspect the preserved current report and create a
new output release (the IDs below are existing accepted source IDs):

```sh
just data-build --version NEW_RELEASE \
  --history historical_backfill_20260923_r2 \
  --college-source college_source_20260922_r1 \
  --current-report data/outputs/rookie_watch_2026.json
just data-verify NEW_RELEASE
```

The preserved report identifies the snapshot it used. Source registration and
replay retain the distinction between original captures, retrospective provider
cache snapshots, annotations, and legacy enriched imports. Transport does not
invent publication timestamps or make retrospective data cutoff-safe.

Prepared model inputs can be rebuilt separately without fitting:

```sh
uv run python -m experiments.future_player_lab.deep.prepare \
  --config experiments/future_player_lab/deep/config.json --run-id NEW_PREPARED_RUN
```

That config remains pinned to its original gold/inventory/benchmark inputs. To use
a newly built gold release, create a new config with its verified hashes; do not
edit old sealed manifests. Publish new data under a new transport release ID.

## Saved models and new experiments

The transport accepts native model files, preprocessing artifacts, Parquet
predictions/metrics, configuration and manifests under `experiments/*/runs/`.
It never deserializes a model to upload or inspect it. Keep preprocessing, feature
order, package versions, training configuration and input hashes alongside a fitted
model so another machine can use it correctly.

The initial inventory found saved predictions/draws/configurations, but no fitted
estimator binaries in recognized native formats. Existing in-memory estimators
cannot be recovered from prediction files. This change does not rerun training or
add estimator persistence to each training routine.

For a later completed model run, use a small custom profile instead of republishing
the entire archive. For example:

```json
{
  "profiles": {
    "model": {
      "include": ["experiments/future_player_lab/runs/MY_COMPLETED_RUN"]
    }
  }
}
```

Pass it with `python -m engine.data.shared snapshot --profiles PROFILE.json
--release MODEL_RELEASE --output data/.shared/staging/MODEL_RELEASE/manifest.json`,
then use the normal upload command. Include required data/preprocessing paths in
the profile if they are not already independently pinned and available. Keep
unfinished runs local unless explicitly archiving them as failed/interrupted evidence.

## Bootstrap release (October 2, 2026)

Published default: `bootstrap_20261002_r4`, selected by
[`data/releases/current.json`](../../data/releases/current.json). The full archive
was independently restored from GCS and verified on October 2, 2026; see the
[verification receipt](../../data/releases/bootstrap_20261002_r4.verification.json).

| Profile | Logical files | Unique objects | Unique bytes |
|---|---:|---:|---:|
| notebooks | 31,369 | 30,400 | 6,365,523,549 |
| rebuild | 44,778 | 35,584 | 6,697,128,870 |
| archive | 82,496 | 43,290 | 8,791,064,040 |

The full archive represents 17,851,769,437 logical bytes, stored as approximately
8.19 GiB of distinct objects. A fresh archive restore needs approximately 25 GiB
for both its cache and working copies (about 13 GiB for the notebooks profile).
It includes ten auxiliary source packages: NFLverse
players, rosters, NGS, participation and FTN; Big Data Bowl 2019 sample, 2020,
2026 prediction, 2026 analytics, and the 2024 SumerSports mirror. Their raw captures,
normalized Parquet tables, quality reports and manifest dependencies are preserved.
Uploading these observations does not automatically add features to the RB model.

The earlier `bootstrap_20261002_r2` release was independently restored from GCS:
all 81,859 archive files passed SHA-256 checks, and the notebook readers passed
with networking disabled. Local rebuild verification also registered 4,222 source
assets and rebuilt 25 gold tables. No models were trained during these checks.

The latest release passed these checks using an independent cache populated from GCS:

- All 82,496 files passed size and SHA-256 verification after restoration.
- All ten auxiliary sources passed manifest identity checks and Parquet row-count checks.
- With networking disabled, the readers loaded 5,038 RB examples with 2,929 features,
  423,704 weekly target rows, 92,985 injury rows and 49,139 saved distribution rows.
- Saved tree/ensemble comparisons and all 14 prepared-input hashes verified.
- The local notebooks profile also passed against the published manifest.
- All 24 transport tests passed, including parallel gcloud no-clobber uploads,
  interrupted transfers, corruption, concurrent creation and offline recovery.

Notebook output checks, lint and type checks passed. No model training was run.
