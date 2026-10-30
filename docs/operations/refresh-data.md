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

The refresh downloads only the schedule and current player stats to find the latest
completed regular-season week. Every scheduled game must be final and every team
must have weekly statistics before the week advances. When a new week is complete
it captures a raw snapshot straight from the providers (completed seasons are
carried forward from the previous snapshot by hash), enriches it with DuckDB into
observation tables, builds a new table batch, rebuilds every refreshable registered
table and, with `--upload`, publishes the query catalog to GCS through the same
workflow as `tables refresh --upload`. Finally it asks the campaign scorers to add
rolling-origin forecasts for the new week (`--no-rescore` skips this).

An unchanged week still checks recipes, retries an unfinished upload, and generates
any missing campaign forecasts. Forecast failures are reported so a later refresh
can retry them without rebuilding the observation batch. To incorporate same-week scoring corrections:

```bash
just data-refresh --force
```

`--status` reports progress and the log path. Use `--version <run-id>` to resume a
run whose batch was built but whose tables were not published.

## Automatic refresh and freshness checks

`deploy/armchair-refresh.timer` runs the complete observation/table/forecast
pipeline hourly, catches up after downtime, and retries failures after five
minutes (at most three starts per hour). This is separate from the 15-minute
injury timer. Install the units on the machine that owns the local data:

```sh
mkdir -p ~/.config/systemd/user
cp deploy/armchair-refresh.service deploy/armchair-refresh.timer ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now armchair-refresh.timer
systemctl --user start armchair-refresh.service
systemctl --user status armchair-refresh.timer armchair-refresh.service
journalctl --user -u armchair-refresh.service -n 80
```

The unit assumes the checkout is `~/dev/armchair-labs`; adjust WorkingDirectory
and ExecStart for another location. Installing the unit is a separate host action;
having these files in the repository does not install it. The timer operates while
the user service manager is running. Enable user lingering separately if unattended
refresh after logout is required. Local refresh does not upload the catalog; add
`--upload` only on the configured publisher.

`GET /api/freshness` reports the served observation cutoff, provider check age,
all four positions' weekly/remaining/season forecast origins, injury collection
age, and the latest refresh failure independently. Supply `?campaign=...` to
check a comparison campaign. The dashboard polls this endpoint every minute and
shows warnings. A recent injury capture cannot clear an old game-result warning.
The endpoint does not fetch providers or rebuild tables.

Provider checks older than six hours and injury captures older than two hours
are flagged; missing timestamps remain unknown. The schedule also flags results
still missing two days after a week's final scheduled date, without assuming
that provider stats have arrived. Refresh receipts now record the provider
check separately from the accepted cutoff, preserve the last successful run time,
and record preflight failures as well as forecast failures. An old receipt without
the new provider timestamp is flagged until the next full refresh succeeds in
checking the source. Failed runs remain visible in the API/UI and systemd journal.

## Weekly campaign re-scoring

`just start`, `just dev`, `just restart` and `engine setup` check provider data and run
the observation refresh before loading ESPN. The dashboard’s **Refresh data**
button runs the same data and forecast pipeline after updating ESPN, then reloads
observation and campaign queries. Both paths run
`experiments.future_player_lab.campaigns.rescore --due` after the tables are ready
(`--no-rescore` skips forecast generation at startup). For every campaign whose
config declares `"rescore": {"method": "rolling_origin"}` and whose remaining-season
models carry saved weights, it builds the same feature rows at the latest completed week,
applies the weights, and replaces the bundle with the new forecast added; the previous
revision is archived under `data/campaign_revisions/`. Re-scored forecasts are
labelled `rolling_origin` because the weights were trained at an earlier cutoff.
Campaigns with dated-availability features also refresh the current season's ESPN
transaction pages and reuse the newest supplement capture. `just campaign-rescore`
runs the same step by hand.

Training a new campaign is still a separate step; see [model campaigns](model-campaigns.md).

Campaign trackers follow every refresh, including no-op weekly refreshes. Which
campaigns are tracked, in what order and with which arguments is declared by the
`tracker` block in `experiments/future_player_lab/campaigns/configs/<campaign>.json`
(see [model campaigns](model-campaigns.md#tracker-configs)); dropping in a config
adds a tracker. Each tracker is isolated, so a failure is logged and never blocks
other trackers or fails the refresh. V3.1 tracking stays paused (`enabled: false`).

The v3.2 tracker collects provider observations and official NFL weekly reports
separately, retains immutable paired forecasts, and updates the prospective
scorecard; the v4 tracker runs after it when v4's fitted integration is available.
The v4.1 tracker (`opening_drive_v41`, the application default) publishes the new
origin's early-week forecast (skipped when already published). Late updates require
fresh game-week evidence before the per-game cutoff; otherwise the early forecast
stays in effect. See the [v3.2 contract](../../experiments/future_player_lab/campaigns/weekly_v32/README.md).

A local user timer runs the configs' `pregame` arms every 15 minutes
(`campaigns.monitor`; inspect with `systemctl --user status opening-drive-monitor.timer`):
v3.2 and v4 first, then v4.1 with pregame updates, which reuse the official report
captures v3.2 just collected (`data/research/opening_drive_v32_official_captures`).
Missed forecast deadlines are recorded rather than reconstructed after kickoff.
Latest v4.1 outputs are kept under
`data/outputs/opening_drive_v41/forecasts/<season>_<week>/<early_week|pregame_60m>/`.

## Diagnose stale or unavailable data

Check `just data-refresh --status` and `just tables list` first. The frontend polls
the table catalog every minute; a catalog change invalidates observation queries.
A catalog mismatch mid-request returns HTTP 409 so the client reloads. Broken
catalog validation should be repaired at the source; clearing a browser cache does
not make an invalid catalog valid.
