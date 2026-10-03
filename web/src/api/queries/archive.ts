/**
 * Queries for the archived workspaces under `/research/archive/*`. They read the
 * research-scoped v1/v2 endpoints, never the current analysis release.
 */
import { queryOptions } from '@tanstack/react-query'
import {
  archiveGet, fetchBoard, fetchCompare, fetchDraft, fetchLeague, fetchLeagueImpact, fetchLeaguePlayers, fetchMatchups,
  fetchMetricReport, fetchOpponents, fetchPlayerOutlook, fetchResearchCatalog, fetchResearchPlayers, fetchResearchReport,
  fetchResearchSources, fetchRoster, fetchTransactions, fetchUnrankable, fetchWire,
} from '../client'
import type { MetricVersion, Status } from '../types'

export const archivedStatusQuery = () =>
  queryOptions({ queryKey: ['archived-status'], queryFn: () => archiveGet<Status>('/api/status?scope=research'), retry: false })

export const archivedLeagueQuery = (version: MetricVersion) =>
  queryOptions({ queryKey: ['league', version], queryFn: () => fetchLeague(version) })
export const leaguePlayersQuery = (version: MetricVersion) =>
  queryOptions({ queryKey: ['league-players', version], queryFn: () => fetchLeaguePlayers(version), retry: false })
export const rosterQuery = (teamId: number, version: MetricVersion) =>
  queryOptions({ queryKey: ['roster', teamId, version], queryFn: () => fetchRoster(teamId, version) })
export const compareQuery = (left: number, right: number, version: MetricVersion) =>
  queryOptions({ queryKey: ['compare', left, right, version], queryFn: () => fetchCompare(left, right, version) })
export const opponentsQuery = (version: MetricVersion) =>
  queryOptions({ queryKey: ['opponents', version], queryFn: () => fetchOpponents(version) })
export const transactionsQuery = () =>
  queryOptions({ queryKey: ['transactions'], queryFn: fetchTransactions })
export const archivedMatchupsQuery = (version: MetricVersion, week: number | null) =>
  queryOptions({ queryKey: ['matchups', version, week], queryFn: () => fetchMatchups(version, week ?? undefined) })
export const wireQuery = (version: MetricVersion, healthyOnly: boolean) =>
  queryOptions({ queryKey: ['wire', version, healthyOnly], queryFn: () => fetchWire(version, healthyOnly) })
export const unrankableQuery = (version: MetricVersion) =>
  queryOptions({ queryKey: ['unrankable', version], queryFn: () => fetchUnrankable(version) })
export const draftQuery = (version: MetricVersion) =>
  queryOptions({ queryKey: ['draft', version], queryFn: () => fetchDraft(version) })

/** The player board; `limit` distinguishes the full reference board from the league fallback. */
export const boardQuery = (version: MetricVersion, limit?: number) =>
  queryOptions({ queryKey: ['board', version, limit], queryFn: () => fetchBoard(version, limit) })

/** The published metric report, or a saved research dataset's report. */
export const metricReportQuery = (dataset?: string) =>
  queryOptions({ queryKey: ['metric-report', dataset ?? 'published'], queryFn: () => dataset ? fetchResearchReport(dataset) : fetchMetricReport() })

export const researchSourcesQuery = (token?: string) =>
  queryOptions({ queryKey: ['research-sources', token], queryFn: fetchResearchSources, retry: false })
export const researchCatalogQuery = (dataset: string) =>
  queryOptions({ queryKey: ['research-catalog', dataset], queryFn: () => fetchResearchCatalog(dataset), retry: false })
export const researchPlayersQuery = (params: URLSearchParams, current: boolean) =>
  queryOptions({ queryKey: ['research-players', current ? 'current' : 'saved', params.toString()], queryFn: () => fetchResearchPlayers(params, current), retry: false })
export const leagueImpactQuery = (params: URLSearchParams) =>
  queryOptions({ queryKey: ['league-impact', params.toString()], queryFn: () => fetchLeagueImpact(params), retry: false })
export const playerOutlookQuery = (token?: string) =>
  queryOptions({ queryKey: ['player-outlook', token], queryFn: () => fetchPlayerOutlook(token), retry: false })
