# Armchair Labs — frontend

React + TypeScript ChairStats dashboard, served by Vite. Reads the FastAPI read API in
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
| `src/api/chairstats.ts` | Campaign bundles, rankings, per-player forecasts and per-model targets |
| `src/api/tables.ts` | Observation tables: catalog, players, rookies, college history |
| `src/api/league.ts` | ESPN observations, matchups and campaign-joined forecasts |
| `src/api/queries/hooks.ts` | `useCampaignChoice`: the campaign and horizon a page looks at, kept in the URL |
| `src/components/Workspace.tsx` | Dashboard/section layouts, rookies and the campaign inventory |
| `src/components/ChairStatsRankings.tsx` | Cross-position rankings of one campaign, filters and CSV export |
| `src/components/CampaignSelect.tsx` | Campaign and horizon pickers shared by forecast-bearing pages |
| `src/api/client.ts` | Fetch wrappers |
| `src/index.css` | Light and dark palettes, dense tables, and tabular figures |

The dashboard defaults to a soft light theme. The header toggle switches to dark
mode and remembers the choice in this browser.

## Note on Vite

Use Node 22.12 or later in the Node 22 series; the repository's `.nvmrc` selects
Node 22 (`nvm install && nvm use` from the repository root). The frontend currently
uses Vite 7. Node 21 is outside Vite 7's supported range even if a local build succeeds.

Observation queries are keyed by the published table catalog checksum; campaign
queries are keyed by campaign id and horizon. `useCampaignChoice` reads
`?campaign=` and `?horizon=` from the URL and otherwise uses the API-declared
`default_campaign_id` (Opening Drive v3). V2 remains available for comparison and
as a fallback when the default bundle has not been published locally. Refresh data
updates ESPN, checks NFL observations, and generates missing campaign forecasts.
The mutation does not pin the old data catalog; when it settles, it reloads the
catalog and invalidates league, campaign, scoring, and status queries.

Forecast periods follow each page's purpose:

- Draft Recap pins full-season forecasts to the league season and preseason
  cutoff (`origin_week=0`), even if a URL contains another horizon. Missing
  preseason forecasts stay unranked. These are reconstructed preseason inputs,
  not evidence of forecasts actually published before the draft.
- Player Rankings, Free Agents, QB Production and roster comparisons default to
  remaining season; Rookies defaults to full season. Their horizon selector can
  still request a weekly view.
- Predictions, matchup player ranks and Team Strength's lineup/depth analysis
  use weekly forecasts. V2 lineup estimates retain the remaining-season per-game
  fallback. Matchup player ranks require the selected season/week's weekly
  forecast; unavailable weekly ranks stay blank.
- Rank columns identify their forecast period. Weekly zero-game forecasts show
  `BYE`, including in player profiles, instead of a misleading bottom-of-list rank.
  The CSV export leaves bye ranks blank and includes a `forecast_status` column.

Navigation is organized into persistent page groups: **Analysis** (ChairStats
rankings, team analysis, QB production, rookies), **League** (predictions, matchups,
team strength, free agents, transactions, draft recap), and **Models** (campaign
inventory with saved metrics). On small screens, Browse pages opens the same
navigation. It stays available on player profiles. Browser smoke scripts for the
previous NextGen layout were removed with it; `npm run build` type-checks every page.

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
| `/league/predictions?week=` · `/matchups/:week/:homeId/:awayId` · `/teams/:teamId/:teamSlug?` · `/free-agents` · `/transactions` · `/draft` | League |
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

Predictions is the main league page. Its existing week buttons also select a compact
rank snapshot above the stats: projected rank, change from the previous week,
projected points, and final actual points. Desktop shows two rows of team tiles;
small screens scroll the snapshot horizontally. It uses the same
team-week values as the matchup cards. A league rank requires complete coverage;
ties share rank. Historical model outputs are identified as reconstructions.

Needs attention and the forecast-model selector are compact disclosures above the
Predictions content. Attention reflects the latest roster. The global **Refresh data**
button sits beside the theme control and updates ESPN, NFL observations, forecasts,
and the dashboard caches. Former `/league/overview` links redirect to Predictions
and retain their query parameters. Matchup detail pages remain available at
`/league/matchups/:week/:homeId/:awayId` and link back to Predictions for that week.

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

Validation: `npm run build` type-checks every page; the backend suite covers the
API contracts the pages consume (`uv run pytest`).

## Team analysis

Variance decomposition and observed pair scoring use Recharts through the shared
`ChartFrame`, axes and tooltip theme. Negative covariance plots left of zero;
missing variance remains unavailable. The correlation matrix remains an interactive
HTML data table: Recharts has no native heatmap, and the table preserves cell values,
sample-size labels, keyboard selection and row/column headers without custom chart
rendering. Player selectors, KPIs and contribution tables remain standard UI.

`/intelligence/teams` uses `/api/team-analysis` for NFL QB/WR/TE
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

Checks: `pytest tests/test_team_correlations.py`.

## Campaign ranks in league pages

Roster, free-agent, draft and matchup tables join the chosen campaign's horizon
rankings (`/api/chairstats/rankings`) to ESPN players by NFL identity. The campaign
and horizon come from `?campaign=` and `?horizon=`; changing them is a query, not a
publication. Overview orders teams by projected weekly offensive lineup points
(excluding K/DST), with depth and actual results alongside. This is a weekly points
rank, not a composite strength score. The roster rank sums horizon forecasts.

Checks: `pytest tests/test_weekly_lineup.py tests/test_league_observations.py tests/test_team_strength.py`.

## Weekly per-game points and accuracy

Matchup detail and Team Strength pages use `/api/league/forecasts?campaign=…`
for point estimates. The `campaign_per_game` source allocates each skill player's
horizon forecast evenly across scheduled NFL games. K/DST estimates average up to
four earlier recorded scores. No ESPN projections enter this recipe. ESPN remains the
source of lineup selections, ownership and actual results. Missing starter forecasts
withhold team totals and winner picks. This is a per-game allocation, not a
separately validated weekly model, opponent adjustment or calibrated win probability.

Forecast requests preserve immutable captures under
`data/outputs/weekly_forecasts/<league>/<season>/`. Accuracy counts only saved
forecasts generated, published and lineup-captured before 00:00 UTC on the week's
earliest NFL game date. Missing captures, incomplete forecasts and actual ties do not
inflate the winner accuracy denominator.

## Actual versus predicted scoring

`/intelligence/scoring` is the **Scoring Trends** analysis page. Search and position
filters, pagination, and the actual-scoring selector live in the URL. Cards are
grouped by position and share axes; they show the full NFL season, actual weekly
production, saved Opening Drive pace, and the actual season-to-date average.
The default actuals reproduce the model's scoring contract; league scoring is an
explicit alternative. Player averages include zero production in completed
scheduled games and exclude byes. Unknown coverage stays blank.

Predictions offers the Recharts card grid under “Explore scoring trends by team,”
for all fantasy teams. Historical weeks use their recorded starters. Full-team predictions
include captured ESPN K/DST estimates, with prior-score fallback, matching the
current matchup recipe. Future weeks hold the current selected lineup, apply known
byes, and use recent K/DST averages; no automatic substitutions are implied. Weeks
after the fantasy regular season are lineup scenarios, not scheduled playoff picks.
Only completed results enter averages and errors.

`/api/scoring/players` and `/api/scoring/league` join saved campaign origins with
the existing DuckDB observation catalog at request time. `scoring_weeks` in
`tables/observations.py` provides the deduplicated scoring transform; current
captures take priority over historical overlap. No extra table publication or
manual research export is required. The normal weekly data refresh and campaign
rescore make the new week available, and the UI polls once per minute. League
refresh also invalidates the team series. Historical weeks select only an earlier
observation cutoff, so a new rescore does not overwrite their origin with current
knowledge. These are explicitly reconstructed curves, not verified pregame captures.

Checks: `pytest tests/test_scoring_trends.py tests/test_weekly_lineup.py`,
`npm run build`, and `npm run lint`.

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

Checks: `pytest tests/test_roster_correlations.py tests/test_team_strength.py`.
