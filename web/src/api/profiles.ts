import type { OutlookPlayer } from './types'
import type { Forecast } from './nextgen'

export type TrackingSeason = {
  player_id: string; season: number; position: string | null
  ngs_cpoe: number | null; ngs_separation: number | null; ngs_yac_oe: number | null; ngs_ryoe_per_att: number | null
}
export type TrackingMetric = keyof Pick<TrackingSeason, 'ngs_cpoe' | 'ngs_separation' | 'ngs_yac_oe' | 'ngs_ryoe_per_att'>

export type ProfileIdentity = {
  player_id: string | null; player_display_name: string; position: string | null; team?: string | null
  rookie_season: number | null; draft_pick?: number | null; draft_round?: number | null; draft_team?: string | null
  birth_date?: string | null; first_nfl_season?: number | null; last_nfl_season?: number | null
  nfl_seasons?: number; college_linked: boolean; current_candidate?: boolean; history_left_truncated?: boolean
}
export type Production = {
  observed_weeks: number; snap_observations: number; offensive_weeks: number | null
  league_points: number | null; points_per_observed_week: number | null; snap_share: number | null
  targets: number | null; carries: number | null; attempts: number | null
  targets_per_observed_week: number | null; carries_per_observed_week: number | null
  passing_yards_per_attempt: number | null; receiving_yards_per_target: number | null; rushing_yards_per_carry: number | null
}
export type NFLSeason = Production & { season: number; teams: string[]; positions: string[]; through_week: number; partial: boolean }
export type CollegeSeason = {
  college_id: string; season: number; team_id: string; college_team: string | null; college_position: string | null
  coverage: number | null; complete_team_season: boolean; observed_stat_games: number
  passing_yards: number; rushing_yards: number; receiving_yards: number; receptions: number; receiving_tds: number
  share_receiving_yards: number | null; share_rushing_yards: number | null
}
export type MetricDefinition = { label: string; unit: string; definition: string }
export type ProfileReport = {
  version: string; season: number; through_week: number; generated_at: string; observations_saved_at: string
  players: number; nfl_weeks: number; college_linked_players: number
  metrics: Record<string, MetricDefinition>; limits: string[]
}
export type ProfileDirectory = { report: ProfileReport; total: number; players: ProfileIdentity[] }
export type PlayerProfileData = {
  version: string; identity: ProfileIdentity; cutoff: { season: number; week: number }
  available_through: { season: number; week: number }
  sources: { gold?: string | null; history: string; college: string; outlook: string; analysis?: string | null; observations_saved_at: string }
  links: { college_id: string; status: string; method: string | null; evidence: string | null }[]
  college_seasons: CollegeSeason[]; nfl_seasons: NFLSeason[]; career: Production
  features: Record<string, number | null>; role_history: (Production & { role: string; label: string })[]
  role_periods: (Production & { role: string; season: number; start_week: number; end_week: number; team: string | null })[]
  transitions: { season: number; kind: string; level: string; description: string }[]
  injury_reports: { season: number; week: number; team: string | null; report_status: string | null; report_primary_injury: string | null; practice_status: string | null }[]
  tracking_seasons: TrackingSeason[]
  tracking: { metrics: Record<TrackingMetric, MetricDefinition & { positions: string[] }>; note: string } | null
  approved_forecasts: (Forecast & { label: string; horizon: string; actual: number | null; complete: boolean; serving: string; reason: string; analysis_version: string })[]
  forecast_scope: 'approved_analysis' | 'archived_research'
  preseason_forecasts: {
    forecast_season: number; forecast_cutoff_date?: string | null; fitted_ppg?: number | null
    fitted_games?: number | null; fitted_season_points?: number | null; fitted_nextgen_season_points?: number | null
    cutoff_preseason_team?: string | null; cutoff_preseason_status?: string | null; cutoff_state_resolution?: string | null
    cutoff_source_url?: string | null; cutoff_evidence_clause?: string | null
  }[]
  college_forecasts: { forecast_year: number; target: string; draft_only?: number | null; college_only?: number | null; college_plus_draft?: number | null; actual?: number | null }[]
  college_development_forecasts: { college_id: string; forecast_year: number; position: string; college_model: number | null; last_season: number | null; actual: number | null }[]
  current_outlook: (OutlookPlayer & { cutoff_week: number }) | null; missing_evidence: string[]; metrics: Record<string, MetricDefinition>; limits: string[]
}
