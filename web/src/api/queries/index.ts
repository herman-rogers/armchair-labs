import { queryOptions, type QueryClient } from '@tanstack/react-query'
import { fetchPlayerProfile, get } from '../client'
import type { ProfileDirectory } from '../profiles'
import { nextgenCatalog, nextgenLeague, nextgenMatchups, nextgenPlayers, nextgenRankings, nextgenRankingEvidence, nextgenForecasts, nextgenRegistry, nextgenRookies, nextgenEvidence } from '../nextgen'
import { passingForecasts, passingEvidence, passingVariations } from '../qbPassing'

// The catalog provider polls and verifies the release; changing its token selects
// new cache entries. Live observations use their own freshness and invalidation.
const published = { staleTime: Infinity, gcTime: 30 * 60_000, retry: false as const }
export const canonicalParams = (params: URLSearchParams) => {
  const copy = new URLSearchParams(params); copy.sort(); return copy.toString()
}
const paramsQuery = <T,>(name: string, params: URLSearchParams, token: string | undefined, fetcher: (params: URLSearchParams, token?: string, signal?: AbortSignal) => Promise<T>) => {
  const key = canonicalParams(params)
  return queryOptions({ ...published, queryKey: [name, token, key], queryFn: ({ signal }) => fetcher(new URLSearchParams(key), token, signal) })
}
export const catalogQuery = (token?: string) => queryOptions({ ...published, queryKey: ['nextgen-catalog', token], queryFn: ({ signal }) => nextgenCatalog(token, signal) })
export const profileQuery = (params: URLSearchParams, token?: string) => paramsQuery('player-profile', params, token, fetchPlayerProfile)
export const directoryQuery = (token?: string) => queryOptions({ ...published, queryKey: ['profile-directory', token], queryFn: ({ signal }) => get<ProfileDirectory>('/api/profiles/directory', token, undefined, signal) })
export const rankingsQuery = (horizon = 'rest_of_season', token?: string) => paramsQuery('nextgen-rankings', new URLSearchParams({ horizon }), token, nextgenRankings)
export const measurementsQuery = (period: string, token?: string) => paramsQuery('nextgen-players', new URLSearchParams({ period }), token, nextgenPlayers)
export const rankingEvidenceQuery = (params: URLSearchParams, token?: string) => paramsQuery('ranking-evidence', params, token, nextgenRankingEvidence)
export const forecastsQuery = (params: URLSearchParams, token?: string) => paramsQuery('nextgen-forecasts', params, token, nextgenForecasts)
export const registryQuery = (params: URLSearchParams, token?: string) => paramsQuery('nextgen-registry', params, token, nextgenRegistry)
export const rookiesQuery = (params: URLSearchParams, token?: string) => paramsQuery('nextgen-rookies', params, token, nextgenRookies)
export const evidenceQuery = (params: URLSearchParams, token?: string) => paramsQuery('nextgen-evidence', params, token, nextgenEvidence)
export const passingQuery = (params: URLSearchParams, token?: string) => paramsQuery('qb-passing', params, token, passingForecasts)
export const passingEvidenceQuery = (params: URLSearchParams, token?: string) => paramsQuery('qb-passing-evidence', params, token, passingEvidence)
export const variationsQuery = (params: URLSearchParams, token?: string) => paramsQuery('qb-variations', params, token, passingVariations)
export const similarityQuery = <T,>(params: URLSearchParams, token?: string) => paramsQuery('similar-careers', params, token,
  (p, t, signal) => get<T>(`/api/profiles/similar?${p}`, t, undefined, signal))
export const leagueQuery = (token?: string) => queryOptions({ queryKey: ['league-observations', token], queryFn: ({ signal }) => nextgenLeague(token, signal), staleTime: 30_000, gcTime: 10 * 60_000, refetchInterval: 60_000, retry: false })
export const matchupsQuery = (week: number | null, token?: string) => queryOptions({ queryKey: ['league-observations', token, 'matchups', week], queryFn: ({ signal }) => nextgenMatchups(week, token, signal), staleTime: 30_000, refetchInterval: 60_000, retry: false })

export function filterDirectory(data: ProfileDirectory, filters: { search: string; position: string; population: string; scope: string }): ProfileDirectory {
  const players = data.players.filter(p =>
    (filters.scope !== 'current' || p.current_candidate) &&
    (filters.scope !== 'college_linked' || p.college_linked) &&
    (filters.position === 'ALL' || p.position === filters.position) &&
    (filters.population === 'all' || (p.rookie_season != null && (filters.population === 'rookie' ? p.rookie_season === data.report.season : p.rookie_season < data.report.season))) &&
    p.player_display_name.toLowerCase().includes(filters.search.toLowerCase()))
  return { ...data, total: players.length, players }
}

/** Start independent requests before the profile's child sections mount. */
export function prefetchProfile(client: QueryClient, params: URLSearchParams, token?: string, period = 'prior', horizon = 'rest_of_season') {
  void client.prefetchQuery(profileQuery(params, token))
  if (!params.has('player_id')) return
  void client.prefetchQuery(catalogQuery(token))
  void client.prefetchQuery(measurementsQuery(period, token))
  const similar = new URLSearchParams(params)
  void client.prefetchQuery(similarityQuery(similar, token))
  if (!params.has('season')) void client.prefetchQuery(rankingsQuery(horizon, token))
}

export const collegeQuery = <T,>(resource: string, params: URLSearchParams, token?: string) =>
  paramsQuery(`college-${resource}`, params, token, (p, t, signal) => get<T>(`/api/research/college${resource ? '/' + resource : ''}?${p}`, t, 'research', signal))
