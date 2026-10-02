# Shared research data

Notebooks and training run locally. Shared data uses a private GCS bucket, immutable
release manifests, and a verified local SSD cache. Git stores source, cleaned
notebooks, profile definitions, and small release references. It does not store
research tables, saved model binaries, cached sessions, or generated notebook exports.

The transport is implemented in `patron.data.shared`. Bucket creation/publication
requires a working Google Cloud login and a chosen project. A release is available
only after its reference has been published to `data/releases/`; the presence of
local staging files does not mean an upload completed.

## Another machine: download once

Install Python 3.12+, uv, and Google Cloud CLI, then from the checkout:

```sh
uv sync --extra shared-data
gcloud auth login
just data-fetch notebooks
just data-cache-verify notebooks
bash experiments/future_player_lab/notebooks/launch.sh
```

The commands use `data/releases/current.json`, a small Git-versioned reference to
one immutable remote manifest with its SHA-256 and GCS generation. Until that file
exists, the initial cloud publication is still pending. To use a particular release:

```sh
just data-fetch notebooks --reference data/releases/RELEASE.json
```

Available download profiles (each larger profile includes the previous one):

| Profile | Contents |
|---|---|
| `notebooks` | Preserved raw objects, gold/enriched releases, prepared feature matrices, injury archive, original benchmark inputs, saved tree/ensemble comparisons and distribution results |
| `rebuild` | Notebook inputs plus accepted historical source collections, captured college sources, source evidence, and captured current-season reports/snapshots |
| `archive` | Rebuild inputs plus the broader research and experiment history, provider cache and saved catalog references |

`archive` includes failed and interrupted runs for historical evidence. Downloading
or uploading one does not certify it as a valid model. The profiles are defined in
[`data/releases/profiles.json`](../../data/releases/profiles.json). They describe
inputs captured at snapshot time; refresh the profile/release when notebooks gain
new dependencies.

The cache defaults to `~/.cache/armchair-labs`. Set `ARMCHAIR_DATA_CACHE` to another
SSD directory if desired. Objects are SHA-256 addressed and stored once per cache;
working copies are materialized at the existing repository paths. Allow space for
both cache and working copies. Copies isolate editable files from shared cache bytes.
Mac/Linux are supported; use WSL on the Windows GPU machine for the existing shell
launchers and build pipeline.

```sh
# Restore deleted working copies using cached bytes, without GCS/network access:
just data-fetch notebooks --offline
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

Create a dedicated bucket in the selected project (one-time operation):

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
just data-snapshot RELEASE
uv run python -m patron.data.shared inventory \
  --manifest data/.shared/staging/RELEASE/manifest.json
just data-upload RELEASE gs://BUCKET/armchair-labs
```

Snapshot copies selected bytes into the content-addressed cache, detects files
changed during capture, and emits an immutable local staging manifest. It excludes
sessions, temporary files, environment/credential files, and fresh bytecode caches.
A few historical seals explicitly include bytecode: those exact declared files are
retained to preserve old manifest verification. Existing sealed Parquet, CSV, JSON,
NumPy, source captures and source/environment snapshots keep their original bytes.
New tables should use Parquet; no legacy conversion is performed during transport.

Upload uses create-only writes and checks remote size, CRC32C and SHA-256 metadata.
It reuses matching existing objects and rejects collisions. All data objects are
uploaded before the completed remote manifest; an interrupted upload has no new
published release. Retry the same command to resume. GCS generations and the final
manifest hash are recorded in `data/releases/RELEASE.json`.
[Google upload/precondition documentation](https://docs.cloud.google.com/storage/docs/samples/storage-upload-file).

After verifying a cloud download in a clean destination, select the default release:

```sh
cp data/releases/RELEASE.json data/releases/current.json
```

Commit the small references, profile changes, source, tests and docs. Never add
`data/.shared`, raw/gold data, local caches, or experiment `runs/` directories to Git.
Remote publication is immutable; changing the default Git reference selects a new
release without changing any existing one. No automated remote deletion or cache
eviction is configured. Source data is not deleted after upload.

## Rebuilding and refreshing

Downloading a preserved release reproduces captured inputs. Re-fetching provider
APIs collects their current revisions and is a different operation.

After `just data-fetch rebuild`, inspect the preserved current report and create a
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

Pass it with `python -m patron.data.shared snapshot --profiles PROFILE.json
--release MODEL_RELEASE --output data/.shared/staging/MODEL_RELEASE/manifest.json`,
then use the normal upload command. Include required data/preprocessing paths in
the profile if they are not already independently pinned and available. Keep
unfinished runs local unless explicitly archiving them as failed/interrupted evidence.

## Initial bootstrap verification (October 2, 2026)

Prepared local snapshot: `bootstrap_20261002_r2`. Its staging manifest and verification
report are under `data/.shared/staging/bootstrap_20261002_r2/` (ignored by Git).
Cloud publication is pending a refreshed Google Cloud login and project selection.
No remote release or default `current.json` has been claimed or created.

| Profile | Logical files | Unique objects | Unique bytes |
|---|---:|---:|---:|
| notebooks | 30,732 | 30,084 | 2,736,797,818 |
| rebuild | 44,141 | 35,268 | 3,068,403,139 |
| archive | 81,859 | 42,975 | 5,164,491,056 |

The complete archive represents 13,826,449,129 logical bytes before content
deduplication. The initial snapshot contains no recognized fitted-model binaries.

Verified locally:

- Restored and hash-checked all 44,141 rebuild-profile files in a separate directory.
- Disabled networking and loaded 5,038 RB examples with 2,929 features, 423,704
  weekly target rows, 107 reviewed absence events, and 92,985 injury rows.
- Loaded saved tree/ensemble comparisons, original-model inputs, all 4,222 raw
  asset descriptors, and 49,139 saved distribution result rows.
- Checked all 14 hashes in the prepared dataset's direct input inventory.
- Registered 4,222 source assets and rebuilt 25 gold tables from restored inputs.
- Ran transport tests for corruption, interrupted uploads, retry, concurrent fetch,
  offline recovery, local-file conflicts, path traversal, model-file round trips,
  notebook output cleaning, legacy seal preservation, and GCS preconditions.

These checks did not train models. Local transport and GCS adapter tests do not
substitute for a real GCS upload/download verification, which remains outstanding.
