/**
 * Every place in the app is a route here. Conventions for paths, search params and
 * scroll live in `navigation.ts`.
 */
import { createBrowserRouter, redirect, type LoaderFunctionArgs } from 'react-router'
import App, { NotFound, RouteError } from './App'
import { ARCHIVED_VIEWS, collegePath, isProfileSection, playerPath } from './navigation'

const workspace = () => import('./components/NextGenView')
const league = () => import('./components/LeagueWorkspace')
const archivedIntelligence = () => import('./components/IntelligenceView')
const archivedLeague = () => import('./components/LeagueView')
const profile = () => import('./components/PlayerPage')

/** Section routes carry their heading; layouts read it with `useMatches()`. */
export type RouteHandle = { title?: string }

const go = (path: string) => () => redirect(path)

/** Links and bookmarks from the old single-page `/?section=…` scheme. */
const legacyRankingParams: Record<string, string> = { rank_horizon: 'horizon', rank_position: 'position', rank_search: 'q', rank_ownership: 'pool' }
const legacyResearchViews: Record<string, string> = { evidence: 'evidence', 'qb-variations': 'qb-experiments', forecasts: 'forecasts', archive: 'archive' }
function dashboardIndex({ request }: LoaderFunctionArgs) {
  const old = new URL(request.url).searchParams
  const section = old.get('section')
  if (section === 'league') return redirect('/league/overview')
  if (section === 'research') return redirect(`/research/${legacyResearchViews[old.get('research') ?? ''] ?? 'evidence'}`)
  const view = old.get('intelligence')
  if (view === 'qb-passing' || view === 'rookies') return redirect(`/intelligence/${view}`)
  const params = new URLSearchParams()
  for (const [from, name] of Object.entries(legacyRankingParams)) if (old.get(from)) params.set(name, old.get(from)!)
  const offset = Number(old.get('rank_offset'))
  if (offset > 0) params.set('page', String(Math.floor(offset / 50) + 1))
  return redirect(`/intelligence/rankings${params.size ? `?${params}` : ''}`)
}

/** Unknown archived views fall back to the workspace's first view. */
function archivedView(workspace: keyof typeof ARCHIVED_VIEWS) {
  const views: readonly string[] = ARCHIVED_VIEWS[workspace]
  return ({ params }: LoaderFunctionArgs) => views.includes(params.view!) ? null : redirect(`/research/archive/${workspace}/${views[0]}`)
}

/** Unknown profile sections fall back to the top of the profile, keeping view params. */
function profileSection(base: (id: string) => string, idParam: 'playerId' | 'collegeId') {
  return ({ params, request }: LoaderFunctionArgs) =>
    params.section && !isProfileSection(params.section) ? redirect(base(params[idParam]!) + new URL(request.url).search) : null
}

export const router = createBrowserRouter([{
  path: '/',
  Component: App,
  hydrateFallbackElement: <p role="status">Loading page…</p>,
  // Pathless wrapper so a page error renders inside the masthead layout.
  children: [{ ErrorBoundary: RouteError, children: [
    { index: true, loader: dashboardIndex },
    {
      lazy: () => workspace().then(m => ({ Component: m.DashboardLayout })),
      children: [
        {
          path: 'intelligence',
          handle: { title: 'Intelligence' } satisfies RouteHandle,
          lazy: () => workspace().then(m => ({ Component: m.IntelligenceSection })),
          children: [
            { index: true, loader: go('/intelligence/rankings') },
            { path: 'rankings', lazy: () => import('./components/NextGenRankings').then(m => ({ Component: m.NextGenRankings })) },
            { path: 'qb-passing', lazy: () => import('./components/QBPassing').then(m => ({ Component: m.QBPassing })) },
            { path: 'rookies', lazy: () => workspace().then(m => ({ Component: m.IntelligenceRookies })) },
          ],
        },
        {
          path: 'league',
          handle: { title: 'League' } satisfies RouteHandle,
          lazy: () => league().then(m => ({ Component: m.LeagueWorkspace })),
          children: [
            { index: true, loader: go('/league/overview') },
            { path: 'overview', lazy: () => league().then(m => ({ Component: m.LeagueOverviewPage })) },
            { path: 'rosters', lazy: () => league().then(m => ({ Component: m.LeagueRosters })) },
            { path: 'free-agents', lazy: () => league().then(m => ({ Component: m.LeagueFreeAgents })) },
            { path: 'transactions', lazy: () => league().then(m => ({ Component: m.LeagueTransactions })) },
            { path: 'draft', lazy: () => league().then(m => ({ Component: m.LeagueDraft })) },
          ],
        },
        {
          path: 'research',
          handle: { title: 'Research' } satisfies RouteHandle,
          lazy: () => workspace().then(m => ({ Component: m.ResearchSection })),
          children: [
            { index: true, loader: go('/research/evidence') },
            { path: 'evidence', lazy: () => workspace().then(m => ({ Component: m.ResearchEvidence })) },
            { path: 'qb-experiments', lazy: () => workspace().then(m => ({ Component: m.ResearchQBExperiments })) },
            { path: 'forecasts', lazy: () => workspace().then(m => ({ Component: m.ReferenceForecasts })) },
            {
              path: 'archive',
              children: [
                { index: true, lazy: () => workspace().then(m => ({ Component: m.ResearchArchive })) },
                {
                  path: 'intelligence',
                  lazy: () => archivedIntelligence().then(m => ({ Component: m.ArchivedIntelligence })),
                  children: [
                    { index: true, loader: go('/research/archive/intelligence/players') },
                    { path: ':view', loader: archivedView('intelligence'), lazy: () => archivedIntelligence().then(m => ({ Component: m.ArchivedIntelligenceView })) },
                  ],
                },
                {
                  path: 'league',
                  lazy: () => archivedLeague().then(m => ({ Component: m.LeagueView })),
                  children: [
                    { index: true, loader: go('/research/archive/league/overview') },
                    { path: ':view', loader: archivedView('league'), lazy: () => archivedLeague().then(m => ({ Component: m.LeagueViewTab })) },
                  ],
                },
              ],
            },
          ],
        },
      ],
    },
    {
      path: 'players/:playerId/:section?',
      loader: profileSection(playerPath, 'playerId'),
      lazy: () => profile().then(m => ({ Component: m.PlayerPage })),
    },
    {
      path: 'college/:collegeId/:section?',
      loader: profileSection(collegePath, 'collegeId'),
      lazy: () => profile().then(m => ({ Component: m.PlayerPage })),
    },
    { path: '*', Component: NotFound },
  ] }],
}])
