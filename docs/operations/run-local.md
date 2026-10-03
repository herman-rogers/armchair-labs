# Run the system locally

Run commands from the repository root. The project uses Python through `uv`, Node/npm
for the React frontend, and `just` for recipes. The package and CLI are named `patron`.
See the [justfile](../../justfile) for the underlying commands.

## Install and start

```bash
just setup                 # uv sync, then npm install in web/
just dev                   # API :8000 and frontend :5173
```

Open <http://localhost:5173>. The frontend proxies `/api` to FastAPI. To run each
service in a separate terminal, use `just api` and `just web`.

Git includes source, configuration, tests, documentation, and curated reference
fixtures. Generated datasets, research archives, model runs, and local environments
are excluded. A fresh clone does not include the historical datasets or a populated
analytical catalog. Existing captured inputs or a data build are needed for data-backed
pages; see [refreshing data](refresh-data.md). Shared data distribution remains
[planned work](../plans/research_compute_and_shared_data_plan_2026-10-02.md).

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
uv run patron auth login
uv run patron auth status
uv run patron auth logout
```

Login opens installed Chrome; sign in manually and select your league. It retains
a browser profile under `data/.browser-profile/` and writes the selected league and
session cookies to `.env` with mode 600. Keep `.env` untracked. On machines without
a browser, supply `ESPN_S2` and `ESPN_SWID` as environment variables; these take
precedence over the file. League configuration lives in `src/patron/config/`.

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
