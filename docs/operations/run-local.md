# Run the system locally

Run commands from the repository root. The project uses Python through `uv`, Node/npm
for the React frontend, and `just` for recipes. The package and CLI are named `engine`.
See the [justfile](../../justfile) for the underlying commands.

## Install and start

```bash
uv run engine dev          # prepare everything, then API :8000 and frontend :5173
# Equivalent: just dev
```

Install `uv`, Node.js/npm, and Chrome or Chromium first (Flatpak Chrome/Chromium is
supported on Linux). The first download needs Google Cloud access to the private
bucket through ADC or `gcloud`; startup opens `gcloud auth login` when credentials
are unavailable. ESPN sign-in opens a browser only when the saved session cannot
be verified. Complete sign-in there and select your league if prompted.

Startup reuses valid local releases. Missing application artifacts are restored
through the verified archive/cache transport; missing query tables use the pinned
table release. It reconciles tables against the restored application selection,
prewarms DuckDB, and loads ESPN observations and their local tables. Existing files
with differing checksums are never overwritten. Setup failures stop startup with
their cause; readiness checks include `/api/status` and Team analysis.

```bash
uv run engine setup          # preparation without servers (also: just setup)
uv run engine dev --restart  # stop existing services, prepare, and restart
uv run engine dev --no-auth  # analysis only; skip ESPN login/sync
uv run engine dev --no-setup # services only, for an already prepared workspace
```

Startup fetches the published table catalog when it is missing and lists the campaign
bundles under `data/campaigns/`; it does not capture a newer NFL week or train
models. Use the publisher refresh workflow for new observations and the campaign
commands for new models.

Open <http://localhost:5173>. The frontend proxies `/api` to FastAPI. To run each
service in a separate terminal, use `just api` and `just web`.

The API reloads on Python changes under `src/engine/`; Vite handles frontend code
changes. Data downloads, generated artifacts and archived source copies do not
restart the API. Restart the dev command once after changing its watcher settings.

Git includes source, configuration, tests, documentation, and curated reference
fixtures. Generated datasets, research archives, model runs, and local environments
are excluded. A fresh clone does not include the historical datasets or a populated
analytical catalog; startup restores the published application inputs. See
[refreshing data](refresh-data.md) for new source captures and builds, and the
[data pipeline](tables.md) for the published reference and consumer contract.

## Stop or restart

```bash
just dev-status
just stop
just restart
```

`just dev` records its API and Vite process groups under `data/.runtime/` and forwards
shutdown signals to them. Ctrl+C stops the supervised services. `just restart`
can also reclaim old listeners on ports 8000 and 5173 after verifying their command
paths belong to this repository.

## ESPN sign-in

```bash
uv run engine auth login
uv run engine auth status
uv run engine auth logout
```

Login opens installed Chrome; sign in manually and select your league. It retains
a browser profile under `data/.browser-profile/` and writes the selected league and
session cookies to `.env` with mode 600. Keep `.env` untracked. On machines without
a browser, supply `ESPN_S2` and `ESPN_SWID` as environment variables; these take
precedence over the file. League configuration lives in `src/engine/config/`.

ESPN supplies league ownership, results, and recorded statuses. The app's **Refresh
league** action is separate from the football observation and forecast refresh.

## Validate changes

```bash
just test                  # offline Python suite
just check                 # lint, type checking, and offline tests
just test-all              # includes live network tests
cd web
npm run build
```

See the [frontend guide](../../web/README.md) for query tests and browser checks.
Scoring and metric functions operate on dataframes; adapters and pipeline commands
own network and disk I/O. Network tests are excluded from the default Python suite.
