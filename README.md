# Armchair Labs

NFL player research, forecasting, and fantasy analysis, with local marimo notebooks
and the Armchair Labs analytics application. The Python package and CLI are `engine` (distribution: `armchair-engine`).
The league configuration supports **Sweaty Plays** (ESPN, 10-team full PPR with
big-play TD bonuses).

## Start locally

```bash
uv run engine dev
```

Open <http://localhost:5173>. This one command installs frontend dependencies,
restores missing published data, reconciles and verifies query tables, opens ESPN
sign-in when needed, loads league observations, and starts both servers. It checks
the data-backed API before announcing readiness. `just dev` is an alias.
Use `just stop`, `just restart`, and `just dev-status` to manage the services.
See [local setup](docs/operations/run-local.md) for prerequisites and options.

Git contains source, configuration, tests, documentation, and curated reference
fixtures. Generated datasets and research runs are excluded; a fresh clone does
not include the historical training data or a populated analytical catalog.

## Documentation

Start at the [documentation index](docs/README.md).

| Purpose | Guide |
|---|---|
| Run and maintain the system | [Operations](docs/operations/README.md) |
| Update observations, forecasts, and precomputed responses | [Refresh data](docs/operations/refresh-data.md) |
| Generate, build, publish, and consume queryable data | [Analytical tables](docs/operations/tables.md) |
| Understand storage, API, and frontend caching | [Architecture](docs/architecture/README.md) |
| See proposed work | [Plans](docs/plans/README.md) |
| Browse current methods, saved models and research runs | [Model library](docs/operations/model-library.md) |
| Read dated findings and evaluations | [Research](docs/research/README.md) |
| Review earlier implementations and migrations | [History](docs/history/README.md) |
| Interpret archived metrics and boards | [Reference](docs/reference/README.md) |
| Launch local model notebooks | [Notebook guide](experiments/future_player_lab/notebooks/README.md) |
| Work on the frontend | [Frontend guide](web/README.md) |

## System overview

Armchair Labs separates observed production, opportunity, efficiency, and participation
from forecasts and league decisions. nflverse supplies football observations; ESPN
supplies league ownership, results, and recorded statuses. Forecasts come from
**ChairStats** campaign bundles: sealed model outputs under `data/campaigns/`.

Three inputs, each with its own refresh, meet only in the API:

| Input | Where it lives | How the API reads it |
|---|---|---|
| Observations | DuckDB table catalog (`data/tables/current.json`) | `/api/tables/*`, `/api/team-analysis`, league identity joins |
| Models | Every campaign bundle under `data/campaigns/<id>/` | `/api/chairstats/*`; the request names the campaign |
| League | ESPN captures in a local league catalog | `/api/league/*`, forecasts joined from the named campaign |

There is no selected or promoted model. Every published bundle is queryable at all
times; pages choose one in the URL. The frontend uses TanStack Query for server
state. See [architecture](docs/architecture/README.md) for the contracts and
[plans](docs/plans/README.md) for further data/training work.

## Data: build, upload, and consume

Registered observation datasets use the same table pipeline: captured sources →
validated batches → versioned Parquet tables → GCS → local DuckDB tables.
Model outputs never enter this catalog; they stay in campaign bundles.

Analysis and league management read observations through DuckDB, and ESPN
observations use a separate local league catalog with the same build/query
machinery. See the [serving coverage](docs/operations/tables.md#frontend-serving-coverage).

```sh
uv sync --extra shared-data
just data-fetch                    # fetch the published query catalog
just tables list
just tables query "SELECT count(*) FROM analytics.team_player_games"

# On the publisher, capture the latest completed week, build the batch, upload tables:
just data-refresh --upload
# Rebuild table recipes from an already built batch:
just tables refresh --due --upload
# Inspect and verify the campaign bundles the API serves:
just campaigns list
```

[The data pipeline guide](docs/operations/tables.md) is the authoritative contract
for generation, validation, adding tables, retrying uploads, and consumption by
other systems. Distribute `data/releases/tables/current.json` to consumers; it pins
immutable Parquet objects and a language-neutral GCS catalog with schemas and
checksums. `just source-fetch` is only for larger research/rebuild input archives.

## Source layout

| Path | Purpose |
|---|---|
| `src/engine/api/` | FastAPI backend |
| `src/engine/data/` | Source capture, DuckDB enrichment (`data/enrich/sql`), observation batches, verification |
| `src/engine/tables/` | DuckDB query sessions, table recipes, atomic refreshes, and sharing |
| `src/engine/config/` | League scoring and settings |
| `src/engine/chairstats/` | Campaign bundle store and the manifest-driven views the API serves |
| `src/engine/scoring/`, `src/engine/metrics/` | League scoring rules and team/league calculations |
| `web/` | React frontend and browser checks |
| `research/`, `experiments/` | Build scripts, protocols, notebooks, and experiments |
| `tests/` | Python validation |
| `data/static/` | Curated reference fixtures |
| `data/` | Local generated datasets, catalogs, and outputs |
| `docs/` | Operations, architecture, plans, research, history, and reference |
