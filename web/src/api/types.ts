/** Shapes returned by the Patron read API. Mirrors `patron.pipeline.BOARD_EXPORT_COLUMNS`. */

export type Position = 'QB' | 'RB' | 'WR' | 'TE'
export type MetricVersion = 'v1' | 'v2'

export interface Player {
  metric_version: MetricVersion
  rank: number
  player_id: string
  player_display_name: string
  position: Position
  team: string
  /** VOR after manual overrides — what the board actually sorts by. */
  adj_vor: number
  /** VOR as computed from the tape, before any manual judgement. */
  vor: number
  ppg: number
  flags: string
  games: number
  season_pts: number
  /** Big-play bonus points. This league's private edge, kept visible. */
  bonus_pts: number
  floor: number | null
  volatility: number | null
  wtd_opp: number
  target_share: number | null
  air_yards_share: number | null
  wopr: number | null
  td_over_exp: number
  age_at_season: number | null
  repl_ppg: number
  override_delta: number
  override_reason: string | null
  /** v2 forward-looking fields; absent on the historical v1 artifact. */
  projected_team?: string
  v2_score?: number
  adj_proj_vor?: number
  proj_vor?: number
  proj_ppg?: number
  proj_repl_ppg?: number
  historical_ppg_prior?: number
  individual_prior_ppg?: number
  prior_branch_ppg?: number
  component_proj_ppg?: number
  projected_floor?: number
  projected_ceiling?: number
  projected_volatility?: number
  floor_vor?: number
  ceiling_vor?: number
  projection_confidence?: number
  effective_games?: number
  age_factor?: number
  td_regression_adjustment?: number
  bonus_regression_adjustment?: number
  team_context_factor?: number
  qb_context?: number
  team_scoring_context?: number
  team_pass_volume?: number
  team_rush_volume?: number
  teammate_competition?: number
  season_target_share?: number | null
  season_carry_share?: number | null
  projected_target_share?: number
  projected_carry_share?: number
  projected_wopr?: number
  projected_air_yards_share?: number
  projected_targets_pg?: number
  projected_carries_pg?: number
  projected_receptions_pg?: number
  projected_receiving_yards_pg?: number
  projected_rushing_yards_pg?: number
  projected_receiving_tds_pg?: number
  projected_rushing_tds_pg?: number
  projected_passing_yards_pg?: number
  projected_passing_tds_pg?: number
  projected_interceptions_pg?: number
  projected_bonus_pg?: number
  air_yard_factor?: number
  opportunity_multiplier?: number
  projection_reason?: string | null
  historical_vor?: number
  historical_repl_ppg?: number
}

export interface BoardResponse {
  version: MetricVersion
  total: number
  offset: number
  limit: number
  players: Player[]
}

export interface Status {
  board_available: boolean
  built_at: string | null
  player_count: number
  metric_versions: Record<MetricVersion, { available: boolean; player_count: number }>
  league: {
    name: string
    team_count: number
    board_season: number
    draft_season: number
    seasons: number[]
    vor_baseline_rank: Record<string, number>
  }
  espn_connected: boolean
  phase: number
}
