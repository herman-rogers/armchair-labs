import { get, post } from './client'

export interface AttentionFlag { kind?: string; severity: string; label: string }
export interface InjuryNews { summary: string; known_on: string; source_url: string; evidence_status: string }
export interface LeaguePlayer {
  player_id: string | null; espn_id: number; player_display_name: string; position: string
  owner_team_name: string | null; owner_team_id: number | null; lineup_slot: string | null; espn_team: string | null
  availability: string; injury_status: string | null; is_mine: boolean; attention?: AttentionFlag[]; injury_news?: InjuryNews[]
}
export interface LeagueTeam {
  team_id: number; team_name: string; wins: number; losses: number; is_mine: boolean; faab_remaining: number | null
  division_name: string | null; schedule?: { week: number; opponent_team_id: number; score: number; outcome: string }[]
}
export interface LeagueObservations {
  attention_notices?: AttentionFlag[]; injury_news_warning?: string | null
  season: number; week: number; stale: boolean; league_name: string; captured_at: string; age_seconds: number
  my_team_id: number | null; regular_season_weeks: number
  teams: LeagueTeam[]; players: LeaguePlayer[]
  transactions: { date: string | null; kind: string | null; team_name: string | null; player_name: string | null; bid_amount: number | null }[]
  draft: { player_id?: string | null; overall: number; round: number; round_pick: number; team_id: number; team_name: string; espn_id: number; player_display_name: string; bid_amount: number | null; keeper: boolean }[]
}
export interface ObservedLineup {
  espn_id: number; player_display_name: string; position: string | null; slot: string
  points: number; pro_opponent: string | null; on_bye: boolean; started: boolean
}
export interface MatchupSide { team_id: number; team_name: string; score: number | null; lineup: ObservedLineup[]; lineup_available: boolean }
export interface ObservedMatchups {
  season: number; current_week: number; requested_week: number; available_weeks: number[]
  captured_at: string; stale: boolean; age_seconds: number; source: string
  matchups: { home: MatchupSide; away: MatchupSide; status: string; involves_me: boolean }[]
}
export interface WeeklyForecastSide {
  team_id: number; team_name: string; score: number | null; model_projection: number | null
  covered_starters: number; starter_count: number
  forecasts: { espn_id: number; player_display_name: string; position: string | null; slot: string; started: boolean; prediction: number | null; basis: string | null; unavailable_reason: string | null }[]
}
export interface WeeklyForecasts {
  source: string; campaign_id: string; campaign_name: string; manifest_sha256: string; horizon: string
  forecast_origins: { season: number; origin_week: number }[]
  season: number; current_week: number; requested_week: number; captured_at: string; stale: boolean
  pregame_deadline: string | null; limitations: string[]
  matchups: { home: WeeklyForecastSide; away: WeeklyForecastSide; involves_me: boolean; margin: number | null; favorite_team_id: number | null }[]
}
export interface ForecastAccuracy {
  season: number; through_week: number; correct: number; evaluated: number; total: number; accuracy: number | null
  point_mae: number | null; point_teams: number; method: string
  games: { week: number; home_team_id: number; away_team_id: number; forecast_margin?: number | null; actual_margin?: number | null; correct?: boolean | null }[]
}

export const fetchLeague = (token?: string, signal?: AbortSignal) => get<LeagueObservations>('/api/league', token, signal)
export const refreshObservations = (token?: string) => post<{ captured_at: string; stale: boolean; age_seconds: number }>('/api/league/refresh', { dataCatalog: token })
export const fetchMatchups = (week: number | null, token?: string, signal?: AbortSignal) =>
  get<ObservedMatchups>(`/api/league/matchups${week == null ? '' : `?week=${week}`}`, token, signal)
export const fetchWeeklyForecasts = (campaign: string, horizon: string, token?: string, signal?: AbortSignal) =>
  get<WeeklyForecasts>(`/api/league/forecasts?${new URLSearchParams({ campaign, horizon })}`, token, signal)
export const fetchForecastAccuracy = (token?: string, signal?: AbortSignal) => get<ForecastAccuracy>('/api/league/forecast-accuracy', token, signal)
