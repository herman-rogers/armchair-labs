# Architecture

The app has a Python FastAPI backend and a React frontend. The backend already
provides the `/api` boundary; the browser does not need to load the analytical
Parquet datasets. See [`src/engine/api/`](../../src/engine/api/) and the
[frontend guide](../../web/README.md).

## Data and serving layers

Preserved raw bytes → enriched observations → validated tables (Parquet storage, native DuckDB queries) →
versioned research/analysis products → API responses → frontend queries.

`data/current.json` atomically selects a consistent analytical release. Builds and
research readers use Polars; analysis and league data reads use native DuckDB
tables. Statistical calculations may still use Python/Polars after SQL selection. Manifests and
dependency verification establish identity and integrity. The system does not require a database server or Redis service.
Optional precomputed API JSON bundles and bounded in-process dataframe/response
caches reduce repeated loading and serialization. They remain subordinate to source
verification. [Pipeline contracts](data_pipeline.md) include the original migration
record and the current serving-product contract.

The frontend uses **TanStack Query** for server state, with centralized query options
in [`web/src/api/queries/`](../../web/src/api/queries/). Analytical query keys include
the catalog identity; verified release data can remain fresh until that identity
changes. Catalog polling, request cancellation, and prefetching coordinate loads.
Mutable league data uses a shorter freshness policy. Directory filters operate on
the compact directory response; college results use server pagination. See the
[frontend guide](../../web/README.md) for the precise cache and test behavior.

## Named analytical tables

Cleaned inputs and derived analytical tables share one embedded DuckDB catalog, with recipes in `src/engine/tables/registry.yaml` and an independent
`data/tables/current.json` selection. Research archives retain original studies;
promoted tables have explicit keys, dependencies, cadence, and provenance.
Native DuckDB snapshots are the default query store and remain disposable caches. GCS sharing reuses the
existing immutable transport. Production orchestration lives in `engine.data.pipeline` and `engine.data.refresh`.
Both source refreshes and recipe-only refreshes use `engine.tables.workflow`.
See [the data pipeline](../operations/tables.md)
for querying, promotion, refreshes, and sharing.

Team analysis uses `analytics.team_player_games` through DuckDB for its team/year
sample, retaining the existing QB filters and complete-case statistical functions.
Its browser cache pins the table manifest as well as the application release.
Normal weekly publication refreshes this table and its dependent summaries.

The application HTTP boundary enables `engine.tables.application.query_scope` for
analysis and league endpoints. Published product outputs become `app_*` tables
through `engine.tables.products`; source and product selections must match or the
request fails closed. Research scope keeps the existing artifact readers.
`engine.tables.league` captures ESPN observations, reviewed news and pregame logs
as local Parquet tables using the same build/verification/native-cache machinery.
League observations are queried before roster, matchup and forecast calculations.
Their separate catalog avoids rebuilding the analytical database on every ESPN
poll and is not included in the shared GCS analytics publication.

## Evidence and release policy

The [NextGen system design](nextgen_system_design_2026-09-23.md) records the design
implemented in September 2026: descriptive measurements, forecasts, and league
decisions have separate evidence requirements. Its release-specific details are
dated, not a promise that every proposed model now serves.

For current commands use [operations](../operations/README.md). For the remaining
source-data, training, and shared-storage work use [plans](../plans/README.md).

## Auxiliary tracking sources

The [NFL source inventory](data-sources.md) covers tracking archives, video, nflverse
and commercial sources. Their verified research releases reuse raw/enriched storage and validated table batches and are selected through `data/source_catalog.json`, independently of
serving. Use the [tracking runbook](../operations/tracking-data.md) for acquisition.
