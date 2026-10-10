# Architecture

The app has a Python FastAPI backend and a React frontend. The backend already
provides the `/api` boundary; the browser does not need to load the analytical
Parquet datasets. See [`src/engine/api/`](../../src/engine/api/) and the
[frontend guide](../../web/README.md).

## Data and serving layers

Observations: preserved raw bytes → enriched observations → validated tables
(Parquet storage, native DuckDB queries) → `/api/tables`, `/api/team-analysis` and
league joins. Models: campaign bundles under `data/campaigns/` → `/api/chairstats`.
League: ESPN captures → local league catalog → `/api/league`.

Nothing selects a model. Every published campaign bundle is queryable; the request
names the campaign, and the frontend keeps that choice in the URL. Builds and
research readers use Polars; observation reads use native DuckDB tables, pinned to
one catalog per request. Manifests and dependency verification establish identity
and integrity. The system does not require a database server or Redis service.
[Pipeline contracts](data_pipeline.md) record the original migration.

The frontend uses **TanStack Query** for server state, with centralized query options
in [`web/src/api/queries/`](../../web/src/api/queries/). Observation query keys include
the table catalog checksum; campaign query keys include the campaign id and
horizon, and responses carry the bundle's manifest checksum. Catalog polling,
request cancellation, and prefetching coordinate loads. Mutable league data uses a
shorter freshness policy. See the [frontend guide](../../web/README.md).

## Named analytical tables

Cleaned inputs and derived analytical tables share one embedded DuckDB catalog, with recipes in `src/engine/tables/registry.yaml` and an independent
`data/tables/current.json` selection. Research archives retain original studies;
registered tables have explicit keys, dependencies, cadence, and provenance.
Native DuckDB snapshots are the default query store and remain disposable caches. GCS sharing reuses the
existing immutable transport. Production orchestration lives in `engine.data.pipeline` and `engine.data.refresh`.
Both source refreshes and recipe-only refreshes use `engine.tables.workflow`.
See [the data pipeline](../operations/tables.md)
for querying, refreshes, and sharing.

Team analysis uses `analytics.team_player_games` through DuckDB for its team/year
sample, retaining the existing QB filters and complete-case statistical functions.
Its browser cache pins the table manifest.
Normal weekly publication refreshes this table and its dependent summaries.

The application HTTP boundary enables `engine.tables.application.query_scope` for
table-reading endpoints; a catalog change mid-request fails closed.
`engine.tables.league` captures ESPN observations, reviewed news and pregame logs
as local Parquet tables using the same build/verification/native-cache machinery.
League observations are queried before roster, matchup and forecast calculations.
Their separate catalog avoids rebuilding the analytical database on every ESPN
poll and is not included in the shared GCS analytics publication.

## Evidence and release policy

The [NextGen system design](nextgen_system_design_2026-09-23.md) records the design
implemented in September 2026 and retired in October 2026 in favour of ChairStats
campaign bundles: descriptive measurements, forecasts, and league decisions have
separate evidence requirements. Its release-specific details are
dated, not a promise that every proposed model now serves.

For current commands use [operations](../operations/README.md). For the remaining
source-data, training, and shared-storage work use [plans](../plans/README.md).

## Auxiliary tracking sources

The [NFL source inventory](data-sources.md) covers tracking archives, video, nflverse
and commercial sources. Their verified research releases reuse raw/enriched storage and validated table batches and are selected through `data/source_catalog.json`, independently of
serving. Use the [tracking runbook](../operations/tracking-data.md) for acquisition.
