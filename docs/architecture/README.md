# Architecture

The app has a Python FastAPI backend and a React frontend. The backend already
provides the `/api` boundary; the browser does not need to load the analytical
Parquet datasets. See [`src/patron/api/`](../../src/patron/api/) and the
[frontend guide](../../web/README.md).

## Data and serving layers

Preserved raw bytes → enriched observations → validated gold Parquet tables →
versioned research/analysis products → API responses → frontend queries.

`data/current.json` atomically selects a consistent analytical release. Polars reads
the versioned tables; manifests and dependency verification establish identity and
integrity. The system does not currently require a database or Redis service.
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

## Evidence and release policy

The [NextGen system design](nextgen_system_design_2026-09-23.md) records the design
implemented in September 2026: descriptive measurements, forecasts, and league
decisions have separate evidence requirements. Its release-specific details are
dated, not a promise that every proposed model now serves.

For current commands use [operations](../operations/README.md). For the remaining
source-data, training, and shared-storage work use [plans](../plans/README.md).

## Auxiliary tracking sources

The [NFL source inventory](data-sources.md) covers tracking archives, video, nflverse
and commercial sources. Their verified research releases reuse raw/enriched/gold
storage and are selected through `data/source_catalog.json`, independently of
serving. Use the [tracking runbook](../operations/tracking-data.md) for acquisition.
