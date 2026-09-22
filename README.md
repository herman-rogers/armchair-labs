# Patron Saints Analytics Engine

League-exact fantasy valuation for **Sweaty Plays** (ESPN, 10-team, full PPR with
big-play TD bonuses). Computes every NFL player's value from raw nflverse data under
the league's own scoring rules, ranks by value over replacement, and joins that model
against live ESPN league state for ownership and waiver decisions.

See [`docs/metrics.md`](docs/metrics.md) for the reviewed v1 catalog and v2 forward
projection model, [`docs/v2_metrics_review.md`](docs/v2_metrics_review.md) for the
2026-08-29 backtest-driven review of v2 and its open issues,
`docs/fantasy_engine_implementation_plan.md` for the full system
design, and `docs/saints_metrics_example.py` for the original draft-prep script.

## Status

**Static boards v1 + v2 + Adaptive shadow.** V1 preserves the league-scored historical VOR board
validated against the Aug 2026 fixture. V2 blends a normalized multi-year PPG prior
with projected player stat lines driven by role, efficiency, age, team volume,
route opportunities/TPRR, official attempts, high-value usage, current depth charts,
and injury-based expected games. Walk-forward fitted rankers learned from 22 seasons
of backtest folds set the board order; everything the backtest disproved is recorded
in the metrics graveyard (`docs/metrics.md`, `docs/v2_metrics_review.md`).
The separate Adaptive board ranks 2026 from the immutable prospective selector
snapshot, without changing the production v2 default or its future grade. The API and
dashboard can switch among all three systems.

nflverse is the football-statistics source. ESPN is deliberately limited to private-
league state such as ownership, free agency, fantasy lineups, transactions, and a live
injury display; ESPN projections and lineup slots do not feed V2 player value.

## Quickstart

```bash
uv sync                  # install deps into .venv
just test                # fast offline unit suite
just board               # build the board -> data/outputs/
just metric-report       # rebuild rolling v2 backtests, metric evidence, and the fitted ranker
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

Scoring and metrics functions are **pure**: they take Polars DataFrames and return
DataFrames, touching neither network nor disk. All loading lives in `data/`, all
writing in `cli.py`. That is what lets the unit suite run offline in milliseconds.
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
