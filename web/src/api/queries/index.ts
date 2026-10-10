import { queryOptions } from '@tanstack/react-query'
import { fetchDataCatalog, fetchStatus } from '../client'
import { fetchCampaigns, fetchMetrics, fetchPlayerForecasts, fetchRankings, fetchTargets } from '../chairstats'
import { fetchForecastAccuracy, fetchLeague, fetchMatchups, fetchWeeklyForecasts } from '../league'
import { fetchCollege, fetchPlayer, fetchPlayers, fetchRookies } from '../tables'
import { fetchTeamStrength } from '../teamStrength'
import { fetchTeamAnalysis } from '../teamAnalysis'

// The catalog provider polls and verifies the published table catalog; changing its
// token selects new cache entries. Campaign responses carry their manifest checksum
// and are keyed by campaign id. Live league observations use their own freshness.
const published = { staleTime: Infinity, gcTime: 30 * 60_000, retry: false as const }
const canonicalParams = (params: URLSearchParams) => {
  const copy = new URLSearchParams(params); copy.sort(); return copy.toString()
}

export const statusQuery = () => queryOptions({ queryKey: ['status'], queryFn: fetchStatus })
/** The published table catalog; polled so a newly published catalog replaces cached observations. */
export const dataCatalogQuery = () => queryOptions({ queryKey: ['data-catalog'], queryFn: fetchDataCatalog, retry: false, refetchInterval: 60_000, staleTime: 30_000 })

// ChairStats: every campaign bundle is available; the request names the campaign.
export const campaignsQuery = () => queryOptions({ queryKey: ['chairstats', 'campaigns'], queryFn: ({ signal }) => fetchCampaigns(signal), staleTime: 60_000, refetchInterval: 120_000, retry: false })
export const rankingsQuery = (campaign: string, horizon: string, extra: Record<string, string> = {}) => {
  const params = new URLSearchParams({ campaign, horizon, limit: '2000', ...extra })
  const key = canonicalParams(params)
  return queryOptions({ ...published, queryKey: ['chairstats', 'rankings', key], queryFn: ({ signal }) => fetchRankings(new URLSearchParams(key), signal), enabled: !!campaign && !!horizon })
}
export const playerForecastsQuery = (playerId: string) =>
  queryOptions({ ...published, staleTime: 60_000, queryKey: ['chairstats', 'player', playerId], queryFn: ({ signal }) => fetchPlayerForecasts(playerId, signal) })
export const targetsQuery = (campaign: string, model: string) =>
  queryOptions({ ...published, queryKey: ['chairstats', 'targets', campaign, model], queryFn: ({ signal }) => fetchTargets(campaign, model, signal), enabled: !!campaign && !!model })
export const metricsQuery = (campaign: string, model: string) =>
  queryOptions({ ...published, queryKey: ['chairstats', 'metrics', campaign, model], queryFn: ({ signal }) => fetchMetrics(campaign, model, signal), enabled: !!campaign && !!model })

// Observation tables, keyed by the published catalog token.
export const playersQuery = (params: URLSearchParams, token?: string) => {
  const key = canonicalParams(params)
  return queryOptions({ ...published, queryKey: ['tables', 'players', token, key], queryFn: ({ signal }) => fetchPlayers(new URLSearchParams(key), token, signal) })
}
export const playerQuery = (playerId: string, season = '', token?: string) =>
  queryOptions({ ...published, queryKey: ['tables', 'player', token, playerId, season], queryFn: ({ signal }) => fetchPlayer(playerId, season, token, signal) })
export const rookiesQuery = (token?: string) =>
  queryOptions({ ...published, queryKey: ['tables', 'rookies', token], queryFn: ({ signal }) => fetchRookies(token, signal) })
export const collegeQuery = (collegeId: string, token?: string) =>
  queryOptions({ ...published, queryKey: ['tables', 'college', token, collegeId], queryFn: ({ signal }) => fetchCollege(collegeId, token, signal) })
// Each response contains its own table version and coverage. Poll the data itself
// so table-only publications refresh without a blocking catalog round trip.
export const teamAnalysisQuery = (params: URLSearchParams, token?: string) => ({
  ...queryOptions({ ...published, queryKey: ['team-analysis', token, canonicalParams(params)], queryFn: ({ signal }) => fetchTeamAnalysis(new URLSearchParams(canonicalParams(params)), token, signal) }),
  staleTime: 30_000,
  refetchInterval: 60_000,
  placeholderData: (previous: Awaited<ReturnType<typeof fetchTeamAnalysis>> | undefined, query?: { queryKey: readonly unknown[] }) => {
    if (!query || query.queryKey[1] !== token || typeof query.queryKey[2] !== 'string') return undefined
    const before = new URLSearchParams(query.queryKey[2]); const after = new URLSearchParams(params)
    before.delete('players'); after.delete('players')
    return canonicalParams(before) === canonicalParams(after) ? previous : undefined
  },
})

// League: ESPN observations, plus forecasts from whichever campaign is asked for.
export const leagueQuery = (token?: string) => queryOptions({ queryKey: ['league-observations', token], queryFn: ({ signal }) => fetchLeague(token, signal), staleTime: 30_000, gcTime: 10 * 60_000, refetchInterval: 60_000, retry: false })
export const matchupsQuery = (week: number | null, token?: string) => queryOptions({ queryKey: ['league-observations', token, 'matchups', week], queryFn: ({ signal }) => fetchMatchups(week, token, signal), staleTime: 30_000, refetchInterval: 60_000, retry: false })
export const weeklyForecastsQuery = (campaign: string, horizon: string, token?: string) => queryOptions({ queryKey: ['league-observations', token, 'forecasts', campaign, horizon], queryFn: ({ signal }) => fetchWeeklyForecasts(campaign, horizon, token, signal), staleTime: 30_000, refetchInterval: 60_000, retry: false, enabled: !!campaign })
export const teamStrengthQuery = (campaign: string, horizon: string, token?: string) => queryOptions({ queryKey: ['league-observations', token, 'team-strength', campaign, horizon], queryFn: ({ signal }) => fetchTeamStrength(campaign, horizon, token, signal), staleTime: 30_000, refetchInterval: 60_000, retry: false, enabled: !!campaign })
export const forecastAccuracyQuery = (token?: string) => queryOptions({ queryKey: ['league-observations', token, 'accuracy'], queryFn: ({ signal }) => fetchForecastAccuracy(token, signal), staleTime: 30_000, retry: false })
