# Armchair Labs

NFL player research, forecasting, and fantasy analysis, with local marimo notebooks
for model experiments and the existing Patron analytics application.

Start the research notebooks with
`bash experiments/future_player_lab/notebooks/launch.sh`. See the
[notebook guide](experiments/future_player_lab/notebooks/README.md) and
[local compute and shared data plan](docs/research_compute_and_shared_data_plan_2026-10-02.md).

Git contains the source, configurations, tests, documentation, and curated reference
fixtures. Generated datasets, research archives, model runs, notebook sessions, and
local environments remain outside Git. Shared GCS/Parquet storage and a verified local
cache are planned; cloud deployment and authentication are deferred. A fresh clone
does not contain the historical datasets required to train the research models.
Existing Python package and CLI names remain `patron`.

Player analysis for **Sweaty Plays** (ESPN, 10-team, full PPR with big-play TD
bonuses). NextGen separates observed production, opportunity, efficiency and
participation from forecasts and league decisions. Fantasy points are one outcome,
not the test of whether every statistic is useful.

## Status

**NextGen is the default dashboard.** It serves corrected player measurements,
complete career profiles, outcome-specific preseason reference baselines, and their
chronological evaluations. [Current NextGen rankings](docs/nextgen_rankings_2026-09-23.md)
cover remaining-season and next-four-week points, including roster/free-agent views.
The current release serves labelled references; advanced challengers did not qualify.
The [September 24 revalidation review](docs/model_revalidation_review_2026-09-24.md)
cleared all 213 pending registry entries: four data dependencies were revalidated
and 209 obsolete or unsupported entries were archived, including 51 legacy recipes.
Training retains all earlier candidate seasons; modern
evaluation is a separate slice. See the [system design](docs/nextgen_system_design_2026-09-23.md)
and [implementation/runbook](docs/nextgen_system_implementation_2026-09-23.md).

Profiles include position-specific NFL Next Gen Stats and forecasts from the
published analysis policy. See the [profile release notes](docs/profile_gold_migration_2026-09-23.md)
for corrected data, coverage, and cutoff behavior.

The [October 2 data reliability and training review](docs/nextgen_data_training_review_2026-10-02.md)
records the published-data audit and the next data work: retain complete source
observations and sample counts, repair transformations and cache invalidation,
and expose the full eligible dataset to tree and representation experiments.
Research input eligibility depends on data validity and forecast timing, not
whether an earlier individual-stat screen or production model approved the input.

**Older models and data are research archives.** V1/V2/Adaptive boards, old Next-gen
blends, four-week outlooks and waiver/trade/matchup experiments do not drive normal
analysis. The archive preserves their original files and records why each use is
excluded. Known-bad target counts and affected derivatives are unknown in the
corrected data, with players and independent outcomes retained. Unsupported
predictive claims do not remove reliable descriptive statistics.

nflverse supplies football observations. ESPN supplies league ownership, results
and recorded statuses. The normal League view reads these observations without a
legacy model board. No current decision-value model has earned serving approval.

The older [metric catalog](docs/metrics.md), [V2 review](docs/v2_metrics_review.md)
and original implementation documents describe archived systems. They are retained
for provenance, not as current serving instructions.

## Quickstart

```bash
uv sync                  # install deps into .venv
just test                # fast offline unit suite
just board               # rebuild an archived board experiment -> data/outputs/
just metric-report       # rebuild archived V2 research; does not publish NextGen
just feature-discovery   # automated nonlinear/temporal/latent feature discovery backtest
just api                 # serve it at :8000
just web                 # React dev server at :5173
just dev                 # run both under one Ctrl+C-safe supervisor
just restart             # reclaim project orphans, then restart both
just stop                # stop both from another terminal
```

Use `just dev` for normal local work. It records the exact API and Vite process groups
under `data/.runtime/`, forwards INT/TERM/HUP into a graceful shutdown, and force-kills
only those recorded groups if they ignore the grace period. `just restart` also
reclaims listeners on ports 8000 and 5173 left by the former shell launcher, but only
after verifying their command paths belong to this repository. `just dev-status`
reports the supervisor and child state.

Without `just`, every recipe is a plain command — see the `Justfile`.

`just nextgen-refresh` checks fresh NFL data for a newly completed regular-season
week, captures league-scored production and snaps, and rebuilds profiles, RB/WR/TE/QB
rankings and QB passing policies together. A week advances only when every scheduled
game is final and every team has weekly statistics. Forecasts advance through Week
17; Week 18's final observations are captured without issuing forecasts for an
already completed season. The separate rookie-analog experiment's Week 13 limit
does not limit these updates.
ESPN's **Refresh league** still updates ownership/statuses separately.

An unchanged week is a no-op. Use `just nextgen-refresh --force` to rebuild after
same-week scoring corrections or newly reviewed injury news. Injuries still require
dated reviewed evidence; the refresh does not infer return dates from a status tag.
The full rebuild includes historical QB validation and can take tens of minutes;
the unchanged-week check only fetches schedules and weekly statistics.
Publication validates the complete immutable release before atomically replacing
`data/current.json`; a failed build keeps the prior forecast available.
`just nextgen-refresh --status` shows progress and the log path. Resume completed
stages with `--version <run-id>`; a partially written stage requires a new run id.
Daily scheduling is optional; this command alone does not install a background job.

## Layout

| Path | What |
|---|---|
| `src/patron/config/` | League scoring rules, VOR baselines, manual overrides |
| `src/patron/data/` | nflverse statistical loading and the derived-artifact cache |
| `src/patron/scoring/` | League scorer, big-play bonuses, kickers, DST |
| `src/patron/metrics/` | PPG, VOR, opportunity shares, age, floor/volatility, projections, fitted rankers |
| `src/patron/config/metric_report.yaml` | Backtest seasons, outcomes, and swappable metric catalog |
| `src/patron/board/` | Board assembly, flags, override application |
| `src/patron/api/` | FastAPI read API |
| `web/` | React frontend |
| `data/static/` | Committed reference data, including the draft-board fixture |
| `data/cache/`, `data/outputs/` | Generated, gitignored |

The **Metric Report** frontend tab reads the last generated
`data/outputs/metric_report.json`. Run `just metric-report` whenever the metric catalog,
v2 formulas, or historical data changes. The same report carries the walk-forward
fitted ranker's per-position weights, which `just board` applies to the live v2 board
(`league.yaml` → `projection_rank_key` / `projection_overall_key`). The board applies a
stored model only when its artifact matches the current fit config, season, and
depth-chart snapshot, so refit before rebuilding: `just metric-report && just board`
(`just metric-report --reanalyze` re-scores from the retained folds without a rebuild). The same run retains detailed player/fold
predictions in Parquet and writes a compact Markdown summary for offline review.

`just feature-discovery` is a separate, report-only research lab. It consumes the
retained metric folds, derives generic weekly time-series and random-convolution
features, and builds a richer source-aware weekly panel of snaps, routes, target and
carry share, expected opportunity, injury/practice state, roster state, team volume,
red-zone work, and EPA rates. It tests nonlinear, latent-archetype,
symbolic-residual, stacked, and market-correction challengers with nested
walk-forward validation. Use `--force-rich` to rebuild the cached rich panel and
descriptors. It writes
`feature_discovery_report.{json,md}` and `feature_discovery_predictions.parquet` but
does not change the production metric configuration or frozen 2026 forecast.

## Design note

The canonical analytics pipeline is **preserved raw sources → enriched tables →
gold tables → derived products**. `data/current.json` selects one validated gold
release and the profiles, outlooks, and college analysis built from it. The frontend
shows its coverage and distinguishes current data from archived model runs.
See [the data pipeline guide](docs/data_pipeline.md) for contracts, commands,
remaining gaps, and the Parquet storage decision.

Scoring and metrics functions are **pure**: they take Polars DataFrames and return
DataFrames, touching neither network nor disk. Source adapters live in `data/`;
pipeline and CLI entry points coordinate versioned writes. That lets the unit suite run offline.
Tests that hit live nflverse are marked `@pytest.mark.network` and excluded by default.

## ESPN sign-in

ESPN's fantasy API has no token flow; access rides on two session cookies. Rather than
copying them out of devtools:

```bash
uv run patron auth login     # opens Chrome, you sign in, it captures the session
uv run patron auth status    # check the stored session still works
uv run patron auth logout    # forget it
```

It drives the Chrome you already have (`channel="chrome"` — no 150MB browser download)
and keeps a browser profile in `data/.browser-profile/`, so signing in again later is
usually instant. Nothing is typed into the login form on your behalf: you sign in
normally, in a real browser, and the flow only watches for the resulting session.

Once signed in it lists the leagues on your account and writes the chosen one, plus
both cookies, into `.env` at mode 600. `.env` is gitignored and must stay that way —
those cookies grant full access to your league. On a machine with no browser, set
`ESPN_S2` / `ESPN_SWID` as environment variables instead; they take precedence over
the file.
