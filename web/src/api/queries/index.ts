import { queryOptions, type QueryClient } from '@tanstack/react-query'
import { fetchDataCatalog, fetchPlayerProfile, fetchStatus, get } from '../client'
import { allPages } from '../pagination'
import type { ProfileDirectory } from '../profiles'
import { nextgenWeeklyEvidence, nextgenWeeklyForecasts, nextgenForecastAccuracy, nextgenRankingHistory, nextgenCatalog, nextgenLeague, nextgenMatchups, nextgenPlayers, nextgenRankings, nextgenRankingEvidence, nextgenForecasts, nextgenRegistry, nextgenRookies, nextgenEvidence } from '../nextgen'
import { passingForecasts, passingEvidence, passingVariations } from '../qbPassing'
import { nextgenTeamStrength } from '../teamStrength'
import { fetchTeamAnalysis, fetchTeamAnalysisCatalog } from '../teamAnalysis'

// The catalog provider polls and verifies the release; changing its token selects
// new cache entries. Live observations use their own freshness and invalidation.
const published = { staleTime: Infinity, gcTime: 30 * 60_000, retry: false as const }
/** Browsing lists keep showing the previous page while the next loads, but never across releases. */
const keepWithinRelease = (token: string | undefined) =>
  <T,>(previous: T | undefined, query?: { queryKey: readonly unknown[] }) => query?.queryKey[1] === token ? previous : undefined
const canonicalParams = (params: URLSearchParams) => {
  const copy = new URLSearchParams(params); copy.sort(); return copy.toString()
}
const paramsQuery = <T,>(name: string, params: URLSearchParams, token: string | undefined, fetcher: (params: URLSearchParams, token?: string, signal?: AbortSignal) => Promise<T>) => {
  const key = canonicalParams(params)
  return queryOptions({ ...published, queryKey: [name, token, key], queryFn: ({ signal }) => fetcher(new URLSearchParams(key), token, signal) })
}
export const statusQuery = () => queryOptions({ queryKey: ['status'], queryFn: fetchStatus })
/** The canonical data release; polled so a newly published release replaces cached analysis. */
export const dataCatalogQuery = () => queryOptions({ queryKey: ['data-catalog'], queryFn: fetchDataCatalog, retry: false, refetchInterval: 60_000, staleTime: 30_000 })
export const catalogQuery = (token?: string) => queryOptions({ ...published, queryKey: ['nextgen-catalog', token], queryFn: ({ signal }) => nextgenCatalog(token, signal) })
export const profileQuery = (params: URLSearchParams, token?: string) => paramsQuery('player-profile', params, token, fetchPlayerProfile)
export const directoryQuery = (token?: string) => queryOptions({ ...published, queryKey: ['profile-directory', token], queryFn: ({ signal }) => get<ProfileDirectory>('/api/profiles/directory', token, undefined, signal) })
export const rankingsQuery = (horizon = 'rest_of_season', token?: string) => paramsQuery('nextgen-rankings', new URLSearchParams({ horizon }), token, nextgenRankings)
export const rankingHistoryQuery = (horizon = 'rest_of_season', token?: string) => paramsQuery('ranking-history', new URLSearchParams({ horizon }), token, nextgenRankingHistory)
export const measurementsQuery = (period: string, token?: string) => paramsQuery('nextgen-players', new URLSearchParams({ period }), token, nextgenPlayers)
export const rankingEvidenceQuery = (params: URLSearchParams, token?: string) => paramsQuery('ranking-evidence', params, token, nextgenRankingEvidence)
export const forecastsQuery = (params: URLSearchParams, token?: string) => ({ ...paramsQuery('nextgen-forecasts', params, token, nextgenForecasts), placeholderData: keepWithinRelease(token) })
export const registryQuery = (params: URLSearchParams, token?: string) => ({ ...paramsQuery('nextgen-registry', params, token, nextgenRegistry), placeholderData: keepWithinRelease(token) })
export const rookiesQuery = (params: URLSearchParams, token?: string) => ({ ...paramsQuery('nextgen-rookies', params, token, nextgenRookies), placeholderData: keepWithinRelease(token) })
export const evidenceQuery = (params: URLSearchParams, token?: string) => paramsQuery('nextgen-evidence', params, token, nextgenEvidence)
export const passingQuery = (params: URLSearchParams, token?: string) => paramsQuery('qb-passing', params, token, passingForecasts)
export const teamAnalysisCatalogQuery = (token?: string) => queryOptions({
  queryKey: ['team-analysis-catalog', token],
  queryFn: ({ signal }) => fetchTeamAnalysisCatalog(token, signal),
  refetchInterval: 60_000, staleTime: 30_000, retry: false,
})
export const teamAnalysisQuery = (params: URLSearchParams, token?: string, tableVersion?: string) => ({
  ...published,
  queryKey: ['team-analysis', token, canonicalParams(params), tableVersion],
  enabled: Boolean(tableVersion),
  queryFn: ({ signal }: { signal: AbortSignal }) => {
    const request = new URLSearchParams(params)
    if (tableVersion) request.set('table_version', tableVersion)
    return fetchTeamAnalysis(request, token, signal)
  },
  placeholderData: (previous: Awaited<ReturnType<typeof fetchTeamAnalysis>> | undefined, query?: { queryKey: readonly unknown[] }) => {
    if (!query || query.queryKey[1] !== token || query.queryKey[3] !== tableVersion || typeof query.queryKey[2] !== 'string') return undefined
    const before = new URLSearchParams(query.queryKey[2]); const after = new URLSearchParams(params)
    before.delete('players'); after.delete('players')
    return canonicalParams(before) === canonicalParams(after) ? previous : undefined
  },
})
export const passingEvidenceQuery = (params: URLSearchParams, token?: string) => paramsQuery('qb-passing-evidence', params, token, passingEvidence)
export const variationsQuery = (params: URLSearchParams, token?: string) => paramsQuery('qb-variations', params, token, passingVariations)
export const similarityQuery = <T,>(params: URLSearchParams, token?: string) => paramsQuery('similar-careers', params, token,
  (p, t, signal) => get<T>(`/api/profiles/similar?${p}`, t, undefined, signal))
export const leagueQuery = (token?: string) => queryOptions({ queryKey: ['league-observations', token], queryFn: ({ signal }) => nextgenLeague(token, signal), staleTime: 30_000, gcTime: 10 * 60_000, refetchInterval: 60_000, retry: false })
export const matchupsQuery = (week: number | null, token?: string) => queryOptions({ queryKey: ['league-observations', token, 'matchups', week], queryFn: ({ signal }) => nextgenMatchups(week, token, signal), staleTime: 30_000, refetchInterval: 60_000, retry: false })
export const teamStrengthQuery = (token?: string) => queryOptions({ queryKey: ['league-observations', token, 'team-strength'], queryFn: ({ signal }) => nextgenTeamStrength(token, signal), staleTime: 30_000, refetchInterval: 60_000, retry: false })

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

type IndividualEvidence = { total: number; comparisons: Record<string, string | number | boolean | null>[] }
/** Every screening comparison recorded for one research stat. */
export const individualEvidenceQuery = (stat: string, token?: string) => queryOptions({ ...published, queryKey: ['individual-evidence', token, stat],
  queryFn: () => allPages<IndividualEvidence>(new URLSearchParams({ scope: 'research', search: stat, limit: '100' }),
    page => get<IndividualEvidence>(`/api/nextgen/individual-evidence?${page}`, token), 'comparisons') })

export const weeklyForecastsQuery = (token?: string) => queryOptions({ queryKey: ['league-observations', token, 'weekly-forecasts'], queryFn: ({ signal }) => nextgenWeeklyForecasts(token, signal), staleTime: 30_000, refetchInterval: 60_000, retry: false })
export const forecastAccuracyQuery = (token?: string) => queryOptions({ queryKey: ['league-observations', token, 'forecast-accuracy'], queryFn: ({ signal }) => nextgenForecastAccuracy(token, signal), staleTime: 30_000, refetchInterval: 60_000, retry: false })

export const weeklyEvidenceQuery = (token?: string) => queryOptions({ queryKey: ['league-observations', token, 'weekly-evidence'], queryFn: ({ signal }) => nextgenWeeklyEvidence(token, signal), staleTime: 60_000, retry: false })
