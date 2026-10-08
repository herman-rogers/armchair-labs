# Armchair Labs — frontend

React + TypeScript NextGen dashboard, served by Vite. Reads the FastAPI read API in
`src/engine/api/`; it computes nothing itself.

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

Navigation is organized into persistent page groups: **Analysis** (rankings,
team analysis, QB passing, rookies), **League management** (overview, matchups, team
strength, free agents, transactions, draft recap), and **Research** (model
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
- **View configuration is search params** (filters, sort, page, week), via
  `useUrlState` / `useUrlPage` / `useUrlParams`. Changes push history so Back undoes
  them; free-text search uses `replace` so keystrokes don't pile up. A page's primary
  `DataTable` keeps its sort in the URL with `sortParam`.
- **Component state** is only for ephemeral UI: expanded rows, open `<details>`,
  "show all" toggles, export errors.
- **Scroll** is handled by `<ScrollRestoration>`: new pages start at the top, Back and
  Forward restore the position, search-param changes keep it.

| Path | Page |
|---|---|
| `/intelligence/rankings` · `/teams` · `/qb-passing` · `/rookies` | Current analysis |
| `/league/overview?week=` · `/matchups/:week/:homeId/:awayId` · `/teams/:teamId/:teamSlug?` · `/free-agents` · `/transactions` · `/draft` | League |
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

## Team analysis

Variance decomposition and observed pair scoring use Recharts through the shared
`ChartFrame`, axes and tooltip theme. Negative covariance plots left of zero;
missing variance remains unavailable. The correlation matrix remains an interactive
HTML data table: Recharts has no native heatmap, and the table preserves cell values,
sample-size labels, keyboard selection and row/column headers without custom chart
rendering. Player selectors, KPIs and contribution tables remain standard UI.

`/intelligence/teams` uses `/api/nextgen/team-analysis` for NFL QB/WR/TE
relationships and selected-starter scoring variance. Team, season range,
quarterback, participation, variance basis and selected player IDs live in the URL.
The default quarterback is the most-used starter in the latest selected season.

The endpoint reads the formal `analytics.team_player_games` table through DuckDB,
with coverage from 2013. Each analysis response includes its table version and
observation cutoff; the page needs no preliminary catalog request. Results are
fresh for 30 seconds and refetched every minute, including table-only publications.
The API loads and validates each database snapshot and its coverage once, then
reuses that setup while file revisions remain unchanged. Missing, corrupt, or source-mismatched tables
produce an unavailable state rather than a fallback. See the
[table guide](../docs/operations/tables.md) for refresh and SQL examples.

The matrix uses each pair's shared games. Selected-starter risk instead uses one
common sample for every selected player, preserving
`Var(sum) = sum(Var) + 2 sum(Cov)` and a positive semidefinite covariance matrix.
The page displays standard deviation, variance decomposition, covariance's
percentage effect, observed score quantiles, sample sizes and exploratory paired
bootstrap intervals. Missing overlap is unavailable, never zero risk. These are
historical scoring statistics conditional on participation, not future projections
or estimates of injury/availability risk. The within-season basis removes season
means and divides by residual degrees of freedom; raw score means and quantiles
remain separately labeled observations.

Checks: `pytest tests/test_team_correlations.py` and
`python web/tests/team_analysis_smoke.py` with the local API and frontend running.

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

## Shared presentation components and Team Strength

- `components/AnalysisPanel.tsx`: shared section surfaces for Team Analysis and Team
  Strength, with optional headings, source labels and a compact padding variant.
- `components/Controls.tsx`: `SelectField` uses the standard filter-control styles.
- `components/TeamSelect.tsx`: one team selector for header and footer; callers own navigation.
- `components/StickyFooter.tsx`: reusable, labeled page-footer region that sticks to the
  viewport bottom while scrolling. Place it after the page content and context;
  it remains in normal flow, supports mobile safe areas, and accepts shared controls.
- `components/StatCards.tsx`: overview cards, an opt-in compact variant, and `StatList`
  for dense secondary metrics without individual boxes.
- `components/Charts.tsx` and `chartTheme.ts`: responsive Recharts frames, captions,
  legends, axes, tooltip styling and palette. Colors follow the existing light/dark
  theme. Chart animations are disabled on Team Strength; exact values remain available.

`/league/teams` starts with an empty chooser. Selecting a team navigates to its stable
ID and readable name, e.g. `/league/teams/3/just-say-no`. Header and sticky-footer
selectors push the same routes; Back/Forward restores team and scroll position.
Old `/league/rosters?team=` links redirect. Recharts and the team diagnostics are
loaded only when opening a selected team's breakdown.

The first chart row pairs projected player contributions with actual weekly scoring
composition. The second pairs replacement impact with actual position benchmarks.
Results show each completed week's score, opponent and league average. Secondary
metrics are compact; detailed scoring, position and replacement tables expand on demand.
Recorded weekly contributions come from reconciled historical starters, including
players no longer on the current roster. Missing breakdowns stay unavailable and
negative scores remain negative. Offensive charts exclude K/DST; the results chart
includes them. No ESPN predictions enter the charts.

Historical correlation risk connects recorded QB/WR/TE starters sharing an NFL
team, for every fantasy team. The API queries the pinned DuckDB catalog through
`team_session`: `analytics.players` supplies unambiguous ESPN/GSIS identities and
`analytics.team_player_games` supplies shared starts. Previous-five-season samples
exclude the current season and other NFL clubs; within-season centering removes
changes in season means. Each pair displays covariance, its contribution to the
pair's variance, combined/zero-covariance standard deviations, approximate Fisher
intervals, bootstrap variance-change intervals, and season breakdowns. Pairwise
historical samples are never pooled into a roster covariance matrix.

The separate current-season contribution uses reconciled recorded starters and
includes K/DST in the team-score denominator. It requires both players in every
completed week (at least three); missing coverage is unavailable, not zero. The
same-team covariance terms add only on this common observed sample. Neither branch
feeds forecasts or alters the existing results standard deviation. Connections use
recorded lineup choices rather than forecast-driven substitutions.

Checks: `pytest tests/test_roster_correlations.py tests/test_team_strength.py` and
`python web/tests/team_correlation_smoke.py`.

Checks: `pytest tests/test_team_strength.py` and
`python web/tests/team_strength_smoke.py` against the API and Vite servers. Browser
checks cover named routes, history, chart coverage, compact desktop layout, mobile
resizing, theme switching, sticky-footer navigation, missing forecasts and stale data.
