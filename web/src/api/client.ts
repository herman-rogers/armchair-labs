import type {
  DraftResponse,
  BoardResponse,
  CompareResponse,
  LeaguePlayersResponse,
  LeagueResponse,
  LeagueStatus,
  MatchupsResponse,
  MetricVersion,
  MetricReport,
  OpponentsResponse,
  RosterResponse,
  Status,
  TransactionsResponse,
  UnrankableResponse,
  WireResponse,
} from './types'
import { presentMetricReport } from '../metricPresentation'

type RequestOptions = { dataCatalog?: string; scope?: 'research'; signal?: AbortSignal; method?: 'GET' | 'POST' }

/**
 * Every API call goes through here: it pins the verified data release and analysis
 * scope, signals a stale catalog, and turns failures into the API's `detail` message.
 */
async function request(path: string, { dataCatalog, scope, signal, method = 'GET' }: RequestOptions = {}) {
  const headers: Record<string, string> = {}
  if (dataCatalog) headers['X-Data-Catalog'] = dataCatalog
  if (scope) headers['X-Analysis-Scope'] = scope
  const response = await fetch(path, { method, headers, signal })
  if (!response.ok) {
    if (response.headers.get('X-Data-Catalog-Stale') === 'true') {
      window.dispatchEvent(new Event('data-catalog-changed'))
    }
    const body = (await response.json().catch(() => ({}))) as { detail?: string }
    throw new Error(body.detail ?? `${response.status} ${response.statusText}`)
  }
  return response
}

export const getResponse = (path: string, dataCatalog?: string, scope?: 'research', signal?: AbortSignal) =>
  request(path, { dataCatalog, scope, signal })

export async function get<T>(path: string, dataCatalog?: string, scope?: 'research', signal?: AbortSignal): Promise<T> {
  return (await getResponse(path, dataCatalog, scope, signal)).json() as Promise<T>
}

export async function post<T>(path: string, options: Pick<RequestOptions, 'dataCatalog' | 'scope'> = {}): Promise<T> {
  return (await request(path, { ...options, method: 'POST' })).json() as Promise<T>
}

export const archiveGet = <T,>(path: string, dataCatalog?: string) => get<T>(path, dataCatalog, 'research')

export const fetchStatus = () => get<Status>('/api/status')

export const fetchPlayerProfile = (params: URLSearchParams, dataCatalog?: string, signal?: AbortSignal) =>
  get<import('./profiles').PlayerProfileData>(`/api/profiles/player?${params}`, dataCatalog, undefined, signal)
export const fetchDataCatalog = () => get<import('./types').DataCatalog>('/api/research/data-catalog')

export const fetchMetricReport = () => archiveGet<MetricReport>('/api/metric-report').then(presentMetricReport)

export const fetchBoard = (version: MetricVersion, limit = 400) =>
  archiveGet<BoardResponse>(`/api/board?version=${version}&limit=${limit}`)

// ---------------------------------------------------------------- league state

export const fetchLeague = (version: MetricVersion) =>
  archiveGet<LeagueResponse>(`/api/league?version=${version}`)

export const fetchLeaguePlayers = (version: MetricVersion, limit = 1000) =>
  archiveGet<LeaguePlayersResponse>(`/api/league/players?version=${version}&limit=${limit}`)

export const fetchRoster = (teamId: number, version: MetricVersion) =>
  archiveGet<RosterResponse>(`/api/league/roster/${teamId}?version=${version}`)

export const fetchWire = (version: MetricVersion, healthyOnly: boolean, limit = 120) =>
  archiveGet<WireResponse>(
    `/api/league/wire?version=${version}&healthy_only=${healthyOnly}&limit=${limit}`,
  )

export const fetchUnrankable = (version: MetricVersion) =>
  archiveGet<UnrankableResponse>(`/api/league/unrankable?version=${version}`)

export const fetchOpponents = (version: MetricVersion) =>
  archiveGet<OpponentsResponse>(`/api/league/opponents?version=${version}`)

export const fetchTransactions = () => archiveGet<TransactionsResponse>('/api/league/transactions')

export const fetchDraft = (version: MetricVersion) => archiveGet<DraftResponse>(`/api/league/draft?version=${version}`)

/** Force a pull from ESPN, ignoring the TTL. */
export const refreshLeague = (version: MetricVersion) =>
  post<LeagueStatus>(`/api/league/refresh?version=${version}`, { scope: 'research' })

export const fetchMatchups = (version: MetricVersion, week?: number) =>
  archiveGet<MatchupsResponse>(
    `/api/league/matchups?version=${version}${week ? `&week=${week}` : ''}`,
  )

export const fetchCompare = (left: number, right: number, version: MetricVersion) =>
  archiveGet<CompareResponse>(`/api/league/compare?left=${left}&right=${right}&version=${version}`)

export const fetchResearchSources = () => archiveGet<import('./types').ResearchSources>('/api/research/sources')
export const fetchResearchCatalog = (dataset = 'legacy') =>
  archiveGet<import('./types').ResearchCatalog>(`/api/research/catalog?${new URLSearchParams({ dataset })}`)
export const fetchResearchReport = (dataset: string) =>
  archiveGet<MetricReport>(`/api/research/report?${new URLSearchParams({ dataset })}`).then(presentMetricReport)
export const fetchResearchPlayers = (params: URLSearchParams, current = false) =>
  archiveGet<import('./types').ResearchPlayers>(`/api/research/${current ? 'current-players' : 'players'}?${params}`)
export const fetchLeagueImpact = (params: URLSearchParams) =>
  archiveGet<import('./types').LeagueImpact>(`/api/research/league-impact?${params}`)

export const fetchPlayerOutlook = (dataCatalog?: string) => archiveGet<import('./types').PlayerOutlookReport>('/api/research/outlook', dataCatalog)
