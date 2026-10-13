import { useLocation } from 'react-router'

/** Pages grouped by the job they help with, shared by navigation and page headings. */
export const PAGE_GROUPS = [
  { label: 'Analysis', pages: [
    { path: '/intelligence/rankings', relatedPaths: ['/players', '/college'], title: 'Player Rankings', description: 'Compare campaign forecasts across positions and find players to investigate.' },
    { path: '/intelligence/teams', title: 'Team Analysis', description: 'Player relationships and combined scoring variance.' },
    { path: '/intelligence/scoring', title: 'Scoring Trends', description: 'Weekly player production, forecast pace and season averages.' },
    { path: '/intelligence/qb-passing', title: 'QB Production', description: 'Every quantity the quarterback models predict.' },
    { path: '/intelligence/rookies', relatedPaths: ['/college'], title: 'Rookies', description: 'Evaluate incoming players and their college evidence.' },
  ] },
  { label: 'League', pages: [
    { path: '/league/predictions', relatedPaths: ['/league/matchups'], title: 'Predictions', description: 'How the model’s weekly forecasts and picks have fared, game by game.' },
    { path: '/league/teams', title: 'Team Strength', description: 'Explore each team’s starters, scoring balance, usable depth and results.' },
    { path: '/league/free-agents', title: 'Free Agents', description: 'Find available players and compare their remaining-season forecasts.' },
    { path: '/league/transactions', title: 'Transactions', description: 'Follow captured roster moves and acquisition bids.' },
    { path: '/league/draft', title: 'Draft Recap', description: 'Review draft selections alongside preseason full-season forecasts.' },
  ] },
  { label: 'Models', pages: [
    { path: '/research/models', title: 'Campaigns', description: 'Every published campaign bundle, its models and saved metrics.' },
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
