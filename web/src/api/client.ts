import type {
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

async function get<T>(path: string): Promise<T> {
  const response = await fetch(path)
  if (!response.ok) {
    const body = (await response.json().catch(() => ({}))) as { detail?: string }
    throw new Error(body.detail ?? `${response.status} ${response.statusText}`)
  }
  return response.json() as Promise<T>
}

export const fetchStatus = () => get<Status>('/api/status')

export const fetchMetricReport = () => get<MetricReport>('/api/metric-report')

export const fetchBoard = (version: MetricVersion, limit = 400) =>
  get<BoardResponse>(`/api/board?version=${version}&limit=${limit}`)

// ---------------------------------------------------------------- league state

async function post<T>(path: string): Promise<T> {
  const response = await fetch(path, { method: 'POST' })
  if (!response.ok) {
    const body = (await response.json().catch(() => ({}))) as { detail?: string }
    throw new Error(body.detail ?? `${response.status} ${response.statusText}`)
  }
  return response.json() as Promise<T>
}

export const fetchLeagueStatus = () => get<LeagueStatus>('/api/league/status')

export const fetchLeague = (version: MetricVersion) =>
  get<LeagueResponse>(`/api/league?version=${version}`)

export const fetchLeaguePlayers = (version: MetricVersion, limit = 1000) =>
  get<LeaguePlayersResponse>(`/api/league/players?version=${version}&limit=${limit}`)

export const fetchRoster = (teamId: number, version: MetricVersion) =>
  get<RosterResponse>(`/api/league/roster/${teamId}?version=${version}`)

export const fetchWire = (version: MetricVersion, healthyOnly: boolean, limit = 120) =>
  get<WireResponse>(
    `/api/league/wire?version=${version}&healthy_only=${healthyOnly}&limit=${limit}`,
  )

export const fetchUnrankable = (version: MetricVersion) =>
  get<UnrankableResponse>(`/api/league/unrankable?version=${version}`)

export const fetchOpponents = (version: MetricVersion) =>
  get<OpponentsResponse>(`/api/league/opponents?version=${version}`)

export const fetchTransactions = () => get<TransactionsResponse>('/api/league/transactions')

/** Force a pull from ESPN, ignoring the TTL. */
export const refreshLeague = (version: MetricVersion) =>
  post<LeagueStatus>(`/api/league/refresh?version=${version}`)

export const fetchMatchups = (version: MetricVersion, week?: number) =>
  get<MatchupsResponse>(
    `/api/league/matchups?version=${version}${week ? `&week=${week}` : ''}`,
  )

export const fetchCompare = (left: number, right: number, version: MetricVersion) =>
  get<CompareResponse>(`/api/league/compare?left=${left}&right=${right}&version=${version}`)
