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
