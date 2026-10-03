import { useLocation } from 'react-router'

/** Pages grouped by the job they help with, shared by navigation and page headings. */
export const PAGE_GROUPS = [
  { label: 'Analysis', pages: [
    { path: '/intelligence/rankings', title: 'Player rankings', description: 'Compare expected production and find players to investigate.' },
    { path: '/intelligence/teams', title: 'Team analysis', description: 'Player relationships and combined scoring variance.' },
    { path: '/intelligence/qb-passing', title: 'QB passing', description: 'Explore quarterback production, efficiency and opportunity.' },
    { path: '/intelligence/rookies', title: 'Rookies', description: 'Evaluate incoming players and their college evidence.' },
  ] },
  { label: 'League management', pages: [
    { path: '/league/overview', title: 'League overview', description: 'Rankings, weekly matchups and scoring trends in one place.' },
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

export function useCurrentPage() {
  const { pathname } = useLocation()
  if (pathname.startsWith('/league/matchups/')) return { path: '/league/overview', title: 'Matchup details', description: 'Scores and lineups for the selected game.', group: 'League management' }
  for (const group of PAGE_GROUPS) {
    const page = group.pages.find(page => pathname === page.path || pathname.startsWith(`${page.path}/`))
    if (page) return { ...page, group: group.label }
  }
  return undefined
}
