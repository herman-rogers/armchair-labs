# Forecast UI and research explorer

> Historical implementation record. Release IDs, results, commands, and interface descriptions reflect the recorded change; use the [operations guides](../operations/README.md) to run the system today.

The application separates production forecasts, historical research, and the
preserved draft reference. The Forecast area contains the league, player list,
rosters, comparisons, and waivers. Research contains evidence and the frozen
Adaptive board. Draft archive retains the original V1 ordering.

## Player and league views

- One shared column definition supplies active-game PPG, expected games, season
  points, source, overall VOR, and position rank. Sorting uses the exact same
  resolved value as the displayed cell. Explicitly unavailable canonical forecasts
  remain blank; older artifacts can use their fitted fields.
- Player Details explains rank basis, cutoff, market rank, sample support,
  historical production, and manual adjustments. Research inputs expand separately.
- Forecast context distinguishes current, stale, degraded, unverified, and frozen
  artifacts. Each board has its own publication timestamp and recorded cutoff(s).
  A publication timestamp is not presented as a refreshed forecast cutoff.
- Simulation breakdowns are optional on the league table. Adaptive season totals
  have no inferred PPG/games decomposition. Its reference view stays separate from
  the production league simulations.
- Matchup lineup/bench calculation and display behavior is preserved. Comparisons
  identify their value basis; roster comparisons no longer describe an unplayed week.

## Research interactions

- Choose evidence window, player population, position, outcome, model, benchmark,
  and hit rate or NDCG. Population options include rookies and saved roster-status
  sensitivity slices when available.
- Compare season curves, inspect a year, sort the leaderboard, expand exact fold
  values and coverage, or download the selected comparison as CSV. Paired deltas
  use shared seasons with equal weight. Missing values are not treated as zero.
- Inspect reported uncertainty separately from the selected benchmark. Older
  reports that compare different player pools are explicitly labeled as legacy.
- Expand common-player market disagreements to see model-only versus market-only
  hits, contrarian precision, recovered misses, and rank-gap bands.
- Inspect weights as feature rows, including variability across refits, training
  sample sizes, and ridge strength. Adaptive selectors instead show selected
  sources and historical selection counts.
- Filter and sort signal evidence; open a signal for its definition, interval,
  observed/eligible sample counts, and per-season correlations. Strongest signals
  are shown first, with all matching signals available on demand.

These views read saved reports and artifacts. They do not refit models, refresh
forecasts, or promote an experiment. Coverage sections identify absent saved data.

## Validation

- TypeScript and Vite production build pass. The installed Node 21 runtime emits
  Vite's existing supported-version warning (Vite requests 20.19+ or 22.12+).
- Browser smoke test covers canonical values, sorting, navigation, player details,
  roster/waiver consistency, missing values, selector models, research filters,
  CSV download, mobile overflow, disconnected ESPN, and empty evidence.
- Real saved-data browser checks: no JavaScript errors at desktop and mobile sizes;
  exercised rookie evidence, Adaptive details, and selector source histories.
- API metadata and matchup tests: 16 passed. Ruff passes on the changed API and
  browser-test files. Frontend lint has existing warnings and no errors.

Run the browser test against a local Vite dev or preview server:

```sh
uv run python web/tests/ui_smoke.py --url http://127.0.0.1:5175
```

It intercepts API requests with deterministic fixtures and uses installed Chrome.
`--channel chromium` uses a separately installed Playwright Chromium browser.
