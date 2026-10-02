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

Current navigation: **Intelligence** (NextGen rankings, Players, Rookies), **League**
(League overview, Rosters, Free agents, Transactions, Draft recap), and **Research**.
Players opens shared profiles, including Similar careers statistical comparisons.
All-careers browsing is hidden; historical data remains available for comparisons.
NextGen rankings are the default Intelligence tab, with remaining-season/next-four-week
horizons and explicit reference/validated status. Roster/free-agent tables and current
profiles use the same published ranks. Dated preseason ECR remains a separate reference.
Sorting/filtering covers full result sets before display pagination. Run
`web/tests/rankings_smoke.py` for the ranking and league integration checks. See
[workflow map](../docs/navigation_restoration_2026-09-23.md).

## Navigation

React Router owns every place in the app; `src/navigation.ts` documents the one
pattern to follow:

- **Places are paths**, declared in `src/routes.tsx` and reached with `<Link>`/`<NavLink>`.
  Tab bars are `<NavLink>`s (active tab: `aria-current="page"`), not buttons with state.
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
| `/league/overview` · `/rosters?team=` · `/free-agents` · `/transactions` · `/draft` | League |
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

League overview combines standings, weekly matchups, roster forecast summaries and
recent activity. See [league overview behavior](../docs/league_overview_2026-09-24.md).

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
