# Armchair Labs

NFL player research, forecasting, and fantasy analysis, with local marimo notebooks
and the Armchair Labs analytics application. The Python package and CLI are `engine` (distribution: `armchair-engine`).
The league configuration supports **Sweaty Plays** (ESPN, 10-team full PPR with
big-play TD bonuses).

## Start locally

```bash
just setup
just dev
```

Open <http://localhost:5173>. `just dev` runs the FastAPI backend on port 8000 and the
React frontend on port 5173. Use `just stop`, `just restart`, and `just dev-status`
to manage them. See [local setup](docs/operations/run-local.md) for ESPN sign-in,
configuration, testing, and the underlying commands.

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
| Read dated findings and evaluations | [Research](docs/research/README.md) |
| Review earlier implementations and migrations | [History](docs/history/README.md) |
| Interpret archived metrics and boards | [Reference](docs/reference/README.md) |
| Launch local model notebooks | [Notebook guide](experiments/future_player_lab/notebooks/README.md) |
| Work on the frontend | [Frontend guide](web/README.md) |

## System overview

NextGen separates observed production, opportunity, efficiency, and participation
from forecasts and league decisions. nflverse supplies football observations; ESPN
supplies league ownership, results, and recorded statuses. Older V1/V2/Adaptive
boards and experiments remain research archives.

The backend queries native DuckDB tables for analysis and league management,
verifies the products selected by `data/current.json`, and serves an API. Research
retains its artifact readers. Precomputed JSON responses and bounded in-process
caches accelerate reads. The frontend uses
TanStack Query for server state. See [architecture](docs/architecture/README.md) for
the contracts and [plans](docs/plans/README.md) for further data/training work.

## Data: build, upload, and consume

Registered analytical datasets use the same table pipeline: captured sources →
validated batches → versioned Parquet tables → GCS → local DuckDB tables.
Research archives are inputs for explicit promotion, not a competing query store.

Analysis and league management read through DuckDB: published products join the
analytical catalog as `app_*` tables, and ESPN observations use a separate local
league catalog with the same build/query machinery. Research-scoped requests keep
their existing artifact readers. Response caches sit above these query paths. See the
[current serving coverage](docs/operations/tables.md#frontend-serving-coverage).

```sh
uv sync --extra shared-data
just data-fetch                    # fetch the published query catalog
just tables list
just tables query "SELECT count(*) FROM analytics.team_player_games"

# On the publisher, generate sources/products, build tables and upload:
just data-refresh --upload
# Rebuild table recipes from captured inputs without retraining products:
just tables refresh --due --upload
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
| `src/engine/data/` | Source loading, versioned data, verification, and caches |
| `src/engine/tables/` | DuckDB query sessions, table recipes, atomic refreshes, and sharing |
| `src/engine/config/` | League scoring, model configuration, and overrides |
| `src/engine/scoring/`, `src/engine/metrics/`, `src/engine/board/` | Scoring, metrics, and board construction |
| `web/` | React frontend and browser checks |
| `research/`, `experiments/` | Build scripts, protocols, notebooks, and experiments |
| `tests/` | Python validation |
| `data/static/` | Curated reference fixtures |
| `data/` | Local generated datasets, catalogs, and outputs |
| `docs/` | Operations, architecture, plans, research, history, and reference |
