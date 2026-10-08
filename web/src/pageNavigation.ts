import { useLocation } from 'react-router'

/** Pages grouped by the job they help with, shared by navigation and page headings. */
export const PAGE_GROUPS = [
  { label: 'Analysis', pages: [
    { path: '/intelligence/rankings', relatedPaths: ['/players'], title: 'Player rankings', description: 'Compare expected production and find players to investigate.' },
    { path: '/intelligence/teams', title: 'Team analysis', description: 'Player relationships and combined scoring variance.' },
    { path: '/intelligence/qb-passing', title: 'QB passing', description: 'Explore quarterback production, efficiency and opportunity.' },
    { path: '/intelligence/rookies', relatedPaths: ['/college'], title: 'Rookies', description: 'Evaluate incoming players and their college evidence.' },
  ] },
  { label: 'Leauge', pages: [
    { path: '/league/overview', relatedPaths: ['/league/matchups'], title: 'Overview', description: 'Rankings, weekly matchups and scoring trends in one place.' },
    { path: '/league/teams', title: 'Team Strength', description: 'Explore each team’s starters, scoring balance, usable depth and results.' },
    { path: '/league/free-agents', title: 'Free agents', description: 'Find available players and compare their remaining-season forecasts.' },
    { path: '/league/transactions', title: 'Transactions', description: 'Follow captured roster moves and acquisition bids.' },
    { path: '/league/draft', title: 'Draft recap', description: 'Review draft selections alongside current player forecasts.' },
  ] },
  { label: 'Research', pages: [
    { path: '/research/evidence', title: 'Model evidence', description: 'Inspect historical evaluations and the evidence behind forecasts.' },
    { path: '/research/qb-experiments', title: 'QB experiments', description: 'Explore quarterback model variations and their historical results.' },
    { path: '/research/forecasts', title: 'Reference forecasts', description: 'Explore reference estimates and their limitations.' },
    { path: '/research/archive', title: 'Research archive', description: 'Browse preserved models, reports and earlier league analyses.' },
  ] },
] as const

/** Resolve detail routes to the same section and parent page used by the sidebar. */
export function navigationPage(pathname: string) {
  const path = pathname.replace(/\/$/, '')
  for (const group of PAGE_GROUPS) {
    const page = group.pages.find(page => [page.path, ...('relatedPaths' in page ? page.relatedPaths : [])]
      .some(parent => path === parent || path.startsWith(`${parent}/`)))
    if (page) return { group, page }
  }
  return undefined
}

export function useCurrentPage() {
  const { pathname } = useLocation()
  const match = navigationPage(pathname)
  if (!match) return undefined
  if (pathname.startsWith('/league/matchups/')) return { ...match.page, title: 'Matchup details', description: 'Scores and lineups for the selected game.', group: match.group.label }
  return { ...match.page, group: match.group.label }
}
