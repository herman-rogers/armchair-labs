# Patron Saints — frontend

React + TypeScript NextGen dashboard, served by Vite. Reads the FastAPI read API in
`src/patron/api/`; it computes nothing itself.

```bash
npm install
npm run dev      # :5173, proxies /api to the backend on :8000
npm run build    # type-check and bundle to dist/
```

The backend must be running (`just api` from the repo root) or the page shows a
notice explaining what to start.

## Layout

| Path | What |
|---|---|
| `src/routes.tsx` | Every page as a React Router route (data router, lazy-loaded) |
| `src/navigation.ts` | The navigation conventions, path builders and URL-state hooks |
| `src/api/nextgen.ts` | Current measurements, forecasts and evidence contracts |
| `src/components/NextGenView.tsx` | Dashboard/section layouts, research pages and the Research archive |
| `src/components/NextGenRankings.tsx` | Published current-season ranks, filters, exports and evidence |
| `src/api/client.ts` | Fetch wrappers |
| `src/components/BoardTable.tsx` | Archived board display; never the default analysis |
| `src/components/Flags.tsx` | BUY / TD-luck / age / Ngms chips |
| `src/index.css` | Light and dark palettes, dense tables, and tabular figures |

The dashboard defaults to a soft light theme. The header toggle switches to dark
mode and remembers the choice in this browser.

## Note on Vite

Use Node 22.12 or later in the Node 22 series; the repository's `.nvmrc` selects
Node 22 (`nvm install && nvm use` from the repository root). The frontend currently
uses Vite 7. Node 21 is outside Vite 7's supported range even if a local build succeeds.

The current catalog pins data and analysis policy together. Archived requests use
explicit research scope; current analysis fails closed when verification fails.
Run `web/tests/nextgen_smoke.py` against local API and frontend servers for the
current desktop/mobile flow. Older smoke scripts describe archived layouts.

Navigation is organized into persistent page groups: **Player analysis** (rankings,
QB passing, rookies), **League management** (overview, matchups, rosters and
comparisons, free agents, transactions, draft recap), and **Research** (model
evidence, QB experiments, reference forecasts, archive). On small screens, Browse
pages opens the same navigation. It stays available on player profiles.
Rankings remain the entry page, with remaining-season/next-four-week
horizons and explicit reference/validated status. Roster/free-agent tables and current
profiles use the same published ranks. Dated preseason ECR remains a separate reference.
Sorting/filtering covers full result sets before display pagination. Run
`web/tests/rankings_smoke.py` for the ranking and league integration checks. See
[workflow map](../docs/history/navigation_restoration_2026-09-23.md).

## Navigation

React Router owns every place in the app; `src/navigation.ts` documents the one
pattern to follow:

- **Places are paths**, declared in `src/routes.tsx` and reached with `<Link>`/`<NavLink>`.
  Grouped page links use `<NavLink>` (active page: `aria-current="page"`).
  `src/pageNavigation.ts` defines page labels, descriptions and purpose groups.
- **View configuration is search params** (filters, sort, page, week, team), via
  `useUrlState` / `useUrlPage` / `useUrlParams`. Changes push history so Back undoes
  them; free-text search uses `replace` so keystrokes don't pile up. A page's primary
  `DataTable` keeps its sort in the URL with `sortParam`.
- **Component state** is only for ephemeral UI: expanded rows, open `<details>`,
  "show all" toggles, export errors.
- **Scroll** is handled by `<ScrollRestoration>`: new pages start at the top, Back and
  Forward restore the position, search-param changes keep it.

| Path | Page |
|---|---|
| `/intelligence/rankings` · `/qb-passing` · `/rookies` | Current analysis |
| `/league/overview` · `/matchups?week=` · `/rosters?team=` · `/free-agents` · `/transactions` · `/draft` | League |
| `/research/evidence` · `/qb-experiments` · `/forecasts` · `/archive` | Research |
| `/research/archive/intelligence/:view` · `/research/archive/league/:view` | Archived workspaces |
| `/players/:playerId/:section?` · `/college/:collegeId/:section?` | Profiles |

Profiles show every section on one page; a section path (`/players/:id/stats`,
`/history`, `/role`, `/similar`, `/tracking`, `/forecasts`, `/evidence`) scrolls to it.
Old `/?section=…` links redirect to their path.

Vite development and preview servers provide the SPA fallback. For a static
production host, serve `web/dist/index.html` for non-file frontend paths (for example,
`try_files $uri $uri/ /index.html` in nginx), while proxying `/api/` to FastAPI.
This is required for direct links and refreshing any page.

League overview combines standings, roster forecast summaries and recent activity,
with navigation to other league pages in the sidebar. Weekly scores and lineups live
on the Matchups schedule; the selected week remains in its URL. Each game opens
`/league/matchups/:week/:homeId/:awayId` with scores and complete lineups. Game
pages link back to their week, to adjacent matchups, and to current team rosters. See [league overview behavior](../docs/history/league_overview_2026-09-24.md).

## Data access and caching

`src/api/queries/` owns current-data query keys, cancellation, freshness, selectors,
and prefetching. Rankings are fetched once per release/horizon and shared by the
ranking page, league workspace, and profiles. Measurements are shared per period.
`/api/profiles/directory` supplies the compact player directory; search, population,
position, and scope filters run locally before table sorting and pagination.

Published queries stay fresh for their release and retain inactive results for
30 minutes. The data-catalog provider polls every minute, verifying the current
source/product closure; errors hide dependent views, and a changed catalog selects
new query entries. League observations have separate 30-second freshness and a
one-minute poll; refresh invalidates the `league-observations` family only.

Profile links prefetch on hover/focus. On navigation, independent profile sections
start in parallel. Fetches consume TanStack Query's abort signal. College research
uses server-side sorting/filtering and 100-row pages, with debounced search; the API
sorts the complete filtered result with stable identity tie-breakers before slicing.
Other small paginated responses use bounded concurrent loading with total/length
checks, so local sorting still sees the complete population.

Validation commands:

```bash
node web/tests/data_queries_test.mjs             # from the repository root
.venv/bin/python web/tests/serving_smoke.py       # API :8000 + Vite :5173
```

The browser check blocks live ESPN calls and verifies search/query reuse, profile
cutoffs, server pagination, mobile layout, and stale-catalog rejection.

## Weekly rank movement

Player rankings, roster/free-agent tables, and player profiles compare the current
published ranks with the latest prior completed-week publication for the same
season and horizon. `/api/nextgen/rankings/history` follows the verified catalog
publication chain, keeps the final published revision per cutoff, and excludes
unpublished experiments. Historical ranking eligibility is checked against that
release's policy. Week labels make gaps explicit; missing ranks are never zero.
Row details and profiles show the available weekly history. Recipe changes and
changing forecast windows can affect the movement.

League overview reconstructs standings after each completed week using win
percentage (ties count as half a win), then points scored. Equal totals share rank.
These are labeled reconstructed standings, not official provider playoff seeds.
An incomplete set of results withholds that week; the current partial week is
excluded. Team row details show the season's weekly standings.

Checks: `pytest tests/test_ranking_history.py tests/test_current_ranking_routes.py`,
`node web/tests/standings_history_test.mjs`, and browser checks
`web/tests/rank_movement_smoke.py` / `web/tests/league_overview_smoke.py`.

## Weekly model points and accuracy

League overview and matchup detail pages use `/api/nextgen/league/forecasts` for
point estimates. The explicitly labeled `nextgen-weekly-reference-v1` recipe
allocates each skill player's eligible published next-four forecast evenly across
scheduled NFL games. K/DST estimates average up to four earlier recorded scores.
No ESPN projections enter this recipe. ESPN remains the source of lineup selections,
ownership and actual results. Missing starter forecasts withhold team totals and
winner picks. This is a weekly reference, not a separately validated weekly model,
opponent adjustment or calibrated win probability.

Inputs must match the season and the immediately preceding completed week. Current
or later scores are excluded from K/DST predictors; bench forecasts never enter
starting totals. Details expose individual model points and coverage.

Forecast requests preserve immutable captures under
`data/outputs/weekly_forecasts/<league>/<season>/`. Accuracy starts at Week 1 and
counts only saved forecasts generated, published and lineup-captured before 00:00
UTC on the week's earliest NFL game date (a conservative cutoff because this
schedule lacks kickoff times). The latest qualifying capture supplies each pick.
Missing captures, incomplete forecasts and actual ties do not inflate the winner
accuracy denominator. Point MAE is measured on verified complete team forecasts.
Historical results remain visible without invented historical predictions. Capture
happens while the forecast page is in use; this does not schedule an unattended job.

Checks: `pytest tests/test_weekly_lineup.py tests/test_league_observations.py`,
`node web/tests/league_outlook_test.mjs`, and
`python web/tests/weekly_forecasts_smoke.py` with the local API and Vite running.
