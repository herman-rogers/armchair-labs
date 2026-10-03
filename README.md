# Armchair Labs

NFL player research, forecasting, and fantasy analysis, with local marimo notebooks
and the Patron analytics application. The Python package and CLI remain `patron`.
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

The backend reads versioned Parquet and analysis products selected by
`data/current.json`, verifies their dependencies, and serves an API. Precomputed
JSON responses and bounded in-process caches accelerate reads. The frontend uses
TanStack Query for server state. See [architecture](docs/architecture/README.md) for
the contracts and [plans](docs/plans/README.md) for proposed shared storage and
further data/training work.

## Source layout

| Path | Purpose |
|---|---|
| `src/patron/api/` | FastAPI backend |
| `src/patron/data/` | Source loading, versioned data, verification, and caches |
| `src/patron/config/` | League scoring, model configuration, and overrides |
| `src/patron/scoring/`, `src/patron/metrics/`, `src/patron/board/` | Scoring, metrics, and board construction |
| `web/` | React frontend and browser checks |
| `research/`, `experiments/` | Build scripts, protocols, notebooks, and experiments |
| `tests/` | Python validation |
| `data/static/` | Curated reference fixtures |
| `data/` | Local generated datasets, catalogs, and outputs |
| `docs/` | Operations, architecture, plans, research, history, and reference |
