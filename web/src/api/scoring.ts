import { get } from './client'

export interface ScoringWeek {
  week: number; actual: number | null; predicted: number | null
  league_actual?: number | null; status: string; bye: boolean
  origin_week: number | null; opponent?: string | null
  origins?: number[]; lineup_basis?: string; reconciled?: boolean | null
  adjustments?: string[]; unfilled_slots?: string[]
  starters?: { name: string; position: string | null; slot: string; prediction: number | null; basis: string }[]
}
export interface ScoringSeries {
  name: string; average: number | null; average_games: number; weeks: ScoringWeek[]
}
export interface PlayerScoring extends ScoringSeries {
  player_id: string; position: string; team: string | null
  forecast_rate: number | null; league_average: number | null; league_average_games: number
}
export interface TeamScoring extends ScoringSeries { team_id: number; mae: number | null }
interface ScoringResponse {
  campaign_id: string; campaign_name: string; manifest_sha256: string; season: number
  through_week: number; last_week: number; forecast_basis: string; method: string; scoring: string
}
export interface PlayerScoringResponse extends ScoringResponse {
  players: PlayerScoring[]; total: number; limit: number; offset: number
}
export interface LeagueScoringResponse extends ScoringResponse {
  teams: TeamScoring[]; captured_at: string; current_week: number; regular_season_weeks: number
  future_basis: string; stale: boolean
}
export const fetchPlayerScoring = (params: URLSearchParams, token?: string, signal?: AbortSignal) =>
  get<PlayerScoringResponse>(`/api/scoring/players?${params}`, token, signal)
export const fetchLeagueScoring = (campaign: string, token?: string, signal?: AbortSignal) =>
  get<LeagueScoringResponse>(`/api/scoring/league?${new URLSearchParams({ campaign })}`, token, signal)
