# Patron Saints Analytics Engine

League-exact fantasy valuation for **Sweaty Plays** (ESPN, 10-team, full PPR with
big-play TD bonuses). Computes every NFL player's value from raw nflverse data under
the league's own scoring rules, ranks by value over replacement, and joins that model
against live ESPN league state for ownership and waiver decisions.

See [`docs/metrics.md`](docs/metrics.md) for the reviewed v1 catalog and v2 forward
projection model, `docs/fantasy_engine_implementation_plan.md` for the full system
design, and `docs/saints_metrics_example.py` for the original draft-prep script.

## Status

**Static boards v1 + v2.** V1 preserves the league-scored historical VOR board
validated against the Aug 2026 fixture. V2 blends a normalized multi-year PPG prior
with projected player stat lines driven by role, efficiency, age, team/QB context,
teammate competition, route opportunities/TPRR, official attempts, high-value usage,
current depth charts, and injury-based expected games. The availability-aware score
balances expected/floor/ceiling VOR. The API and dashboard can switch between both
versions.

nflverse is the football-statistics source. ESPN is deliberately limited to private-
league state such as ownership, free agency, fantasy lineups, transactions, and a live
injury display; ESPN projections and lineup slots do not feed V2 player value.

## Quickstart

```bash
uv sync                  # install deps into .venv
just test                # fast offline unit suite
just board               # build the board -> data/outputs/
just api                 # serve it at :8000
just web                 # React dev server at :5173
```

Without `just`, every recipe is a plain command — see the `Justfile`.

## Layout

| Path | What |
|---|---|
| `src/patron/config/` | League scoring rules, VOR baselines, manual overrides |
| `src/patron/data/` | nflverse statistical loading and the derived-artifact cache |
| `src/patron/scoring/` | League scorer, big-play bonuses, kickers, DST |
| `src/patron/metrics/` | PPG, VOR, opportunity, TD-over-expectation, age, floor/volatility |
| `src/patron/board/` | Board assembly, flags, override application |
| `src/patron/api/` | FastAPI read API |
| `web/` | React frontend |
| `data/static/` | Committed reference data, including the draft-board fixture |
| `data/cache/`, `data/outputs/` | Generated, gitignored |

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
