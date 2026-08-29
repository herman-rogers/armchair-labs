# Patron Saints Analytics Engine

League-exact fantasy valuation for **Sweaty Plays** (ESPN, 10-team, full PPR with
big-play TD bonuses). Computes every NFL player's value from raw nflverse data under
the league's own scoring rules, ranks by value over replacement, and — in later
phases — joins that against live ESPN league state.

See `docs/fantasy_engine_implementation_plan.md` for the full design, and
`docs/saints_metrics_example.py` for the original draft-prep script this ports.

## Status

**Phase 1 — the static board.** Scoring engine, metrics catalog, and VOR board,
validated against the Aug 2026 draft board in `data/static/2026_draft_list.md`.

Phases 2–4 (ESPN sync, the polling loop, in-season intelligence) are not built yet;
`src/patron/{espn,store,alerts,scheduler}/` are placeholders.

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
| `src/patron/data/` | nflverse loading and the derived-artifact cache (**the only network I/O**) |
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

## Secrets

ESPN private-league access needs the `SWID` and `espn_s2` cookies from a logged-in
browser session. They go in `.env` (gitignored) — never in the repo. Copy
`.env.example` to get started. Phase 1 does not read them.
