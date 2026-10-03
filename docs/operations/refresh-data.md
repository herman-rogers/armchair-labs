# Refresh and publish application data

The [data pipeline guide](tables.md) defines the supported generation → tables →
GCS workflow. This page covers application/forecast eligibility and serving details.
Use `just data-refresh --upload` on the publisher; omit `--upload` for local work.

The backend serves a verified release selected by `data/current.json`. It reads
native DuckDB tables built from versioned Parquet and published products; research
scope retains its artifact readers. There is no database service to start.
The [pipeline contract](../architecture/data_pipeline.md) describes the storage layers.

## Normal weekly refresh

```bash
just data-refresh
just data-refresh --status
```

The refresh checks for a newly completed regular-season week. Every scheduled game
must be final and every team must have weekly statistics before the week advances.
It captures league-scored production and snaps, then rebuilds profiles, RB/WR/TE/QB
rankings, QB passing policies, and the dedicated weekly points product together.
Forecasts advance through Week 17;
Week 18 captures final observations without forecasting an already completed season.
The separate rookie-analog experiment's Week 13 limit does not constrain this workflow.

An unchanged week is a no-op. To incorporate same-week scoring corrections or newly
reviewed injury evidence:

```bash
just data-refresh --force
```

Injury inputs still require dated reviewed evidence; refresh does not infer return
dates from a status tag. ESPN ownership/status refresh is a separate action in the app.

The full rebuild includes historical QB validation and can take tens of minutes.
The unchanged-week check only fetches schedules and weekly statistics. Publication
validates the complete immutable release before atomically replacing `data/current.json`;
a failed build retains the previous selection. `--status` reports progress and the
log path. Use `--version <run-id>` to resume completed stages; a partially written
stage requires a new run ID. The command does not install a background schedule.

After source/product publication, the refresh builds every refreshable registered
table and its dependents. With `--upload`, it then publishes the query catalog and
consumer inventory to GCS through the same workflow as `tables refresh --upload`. Unchanged-week runs
check those recipes too. Selected application products and published ranking history
are also promoted into `app_*` query tables. Analysis routes require the table catalog
to match the application/weekly product selection. If the table step fails after
source publication, affected analysis reports unavailable until a retry succeeds;
resume the run or run `just tables refresh`.
Previously published table versions remain available for explicit historical SQL.

The weekly forecaster fits QB/RB/WR/TE separately using earlier seasons, with weekly
scoring, workload, snaps, and prior opponent results as inputs. It evaluates ridge
and boosting against weekly references under the fixed
[weekly protocol](../../research/weekly_points_protocol.md). Failed promotion scopes
serve the weekly reference selected from prior seasons; K/DST use prior recorded
scores. ESPN supplies observed lineups and outcomes, never these point forecasts.

For an independent weekly build against the current analysis:

```bash
uv run python research/build_weekly_points.py --version <unique-weekly-version> --publish
```

This saves immutable panels, chronological fold predictions, evaluation, and source
snapshots in `data/research/<unique-weekly-version>`. The selected product lives at
`data/weekly/current.json` and must match the exact current analysis. The normal
refresh activates it after catalog publication; an interrupted activation can be
resumed with the same run ID. The API withholds forecasts during a release mismatch.
After a standalone weekly build, run `just tables refresh --upload` to update the
application query catalog and shared tables.
League overview shows historical player errors and this season's conditional lineup
replay separately from actual timestamped pregame captures. Week 1 replay totals
may be unavailable because there is no prior captured K/DST history.

## Precompute frontend responses

The normal weekly workflow updates Team analysis tables automatically. After a
manual source publication, run `just tables refresh`. Other
scheduled SQL tables use `just tables refresh --due`. Add `--upload` to share the
resulting catalog through GCS. This does not replace application catalog
publication or the serving bundle step below.
See [analytical tables](tables.md) for daily/weekly scheduling and frozen studies.

After publishing a catalog or changing backend Python code:

```bash
just data-serving
```

This builds disposable JSON responses for the profile directory, both ranking
horizons, all four measurement periods, current candidate profiles, career comparisons,
and observed completed-season profiles. `just data-serving --no-history` produces
a smaller bundle. It does not train models or alter table/research artifacts.

Bundles are immutable under `data/serving/releases/`; `data/serving/current.json`
selects one. A bundle is bound to the analytical catalog and backend implementation.
When either changes, the API uses verified source reads until a matching bundle is
built. Other players and research scopes also use bounded lazy response caching.
A matching bundle with corrupted bytes fails validation rather than being served.
See [serving contracts](../architecture/data_pipeline.md#application-serving-products).

## Advanced builds and publication

Use the CLI help before a manual build; required inputs depend on retained captures
and the products being published:

```bash
uv run engine data build --help
uv run engine data products --help
uv run engine data publish --help
uv run engine data verify --help
uv run python research/build_read_models.py --help
```

Build into new version IDs, verify tables and their raw/enriched dependencies, and publish
only a consistent product/analysis set. Dated migration documents retain example
release IDs for reproduction; those are not fresh-clone bootstrap instructions.
Use the weekly refresh for normal advancement of an existing NextGen catalog.

## Diagnose stale or unavailable data

Check refresh progress and the selected catalog first. A completed build is not
necessarily the published release. The frontend checks catalog freshness every
minute and on focus; catalog changes invalidate release-bound queries. A catalog
mismatch or dependency mutation can return HTTP 409 so the client can reload the
catalog. Broken catalog/dependency validation should be repaired at the source;
clearing a browser cache does not make an invalid release valid.

For serving-cache and pagination checks:

```bash
uv run pytest tests/test_serving_cache.py tests/test_server_pagination.py
```
