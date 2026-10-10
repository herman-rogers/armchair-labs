# Refresh observation data

The [observation tables guide](tables.md) defines the capture → batch → tables → GCS
workflow. This page covers the weekly cadence. Use `just data-refresh --upload` on
the publisher; omit `--upload` for local work.

The backend reads native DuckDB tables built from versioned Parquet, plus every
campaign bundle under `data/campaigns/`. There is no database service to start and
no product or model selection to publish.

## Normal weekly refresh

```bash
just data-refresh
just data-refresh --status
```

The refresh checks for a newly completed regular-season week. Every scheduled game
must be final and every team must have weekly statistics before the week advances.
It captures league-scored production and snaps, registers them with the accepted
historical archive as a raw snapshot, enriches offline into observation tables, and
builds a new table batch. Then it rebuilds every refreshable registered table and,
with `--upload`, publishes the query catalog to GCS through the same workflow as
`tables refresh --upload`.

An unchanged week is a no-op that still checks recipes and retries an unfinished
upload. To incorporate same-week scoring corrections:

```bash
just data-refresh --force
```

`--status` reports progress and the log path. Use `--version <run-id>` to resume a
run whose batch was built but whose tables were not published. The command does not
install a background schedule.

## Model forecasts are separate

Campaign bundles do not refresh with the tables. Train and score a campaign with the
[model campaigns](model-campaigns.md) commands, publish its bundle, and it is
queryable at once through `/api/chairstats`. A campaign's saved forecasts carry
their own season and observation cutoff; the API reports them on every response.

## Diagnose stale or unavailable data

Check `just data-refresh --status` and `just tables list` first. The frontend polls
the table catalog every minute; a catalog change invalidates observation queries.
A catalog mismatch mid-request returns HTTP 409 so the client reloads. Broken
catalog validation should be repaired at the source; clearing a browser cache does
not make an invalid catalog valid.
