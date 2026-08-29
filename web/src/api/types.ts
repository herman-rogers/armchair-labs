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
  /** Historical VOR after manual overrides — the v1 sort key. */
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
  depth_chart_rank?: number | null
  depth_chart_position?: string | null
  depth_chart_position_group?: string | null
  depth_chart_date?: string | null
  depth_role_factor?: number
  /** Per-position sort key chosen from the metric report (docs/v2_metrics_review.md §8). */
  /** Overall board order: fitted_ppg above its positional replacement (one scale for all positions). */
  v2_overall_vor?: number | null
  /** Order within the position by the position's own key. */
  v2_position_rank?: number
  v2_rank_key?: string
  /** The key's per-game equivalent value. */
  v2_rank_value?: number | null
  /** Per-game value above the positional replacement of the same key, plus overrides. Sorts within position. */
  v2_rank_vor?: number | null
  /** Walk-forward fitted ranker outputs; null when no fitted model exists for the season. */
  fitted_ppg?: number | null
  fitted_games?: number | null
  fitted_season_points?: number | null
  fitted_season_points_direct?: number | null
  fitted_return_prob?: number | null
  fitted_games_if_played?: number | null
  fitted_two_stage?: number | null
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
  projection_confidence?: number
  availability_confidence?: number
  projected_availability?: number
  expected_games?: number
  season_equivalent_ppg?: number
  historical_injury_report_weeks?: number
  injury_missed_equivalents?: number
  effective_games?: number
  age_factor?: number
  qb_context?: number
  team_scoring_context?: number
  team_pass_volume?: number
  team_dropbacks?: number
  team_rush_volume?: number
  teammate_competition?: number
  season_target_share?: number | null
  season_carry_share?: number | null
  projected_target_share?: number
  projected_carry_share?: number
  projected_wopr?: number
  projected_air_yards_share?: number
  projected_targets_pg?: number
  projected_route_opportunities_pg?: number
  projected_route_participation?: number
  projected_targets_per_route_opportunity?: number
  projected_carries_pg?: number
  projected_receptions_pg?: number
  projected_receiving_yards_pg?: number
  projected_rushing_yards_pg?: number
  projected_receiving_tds_pg?: number
  projected_rushing_tds_pg?: number
  projected_red_zone_targets_pg?: number
  projected_end_zone_targets_pg?: number
  projected_red_zone_carries_pg?: number
  projected_goal_line_carries_pg?: number
  projected_pass_attempts_pg?: number
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
  metric_report: { available: boolean; built_at: string | null }
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

// ---------------------------------------------------------------- metric report

export type MetricAssessment =
  | 'strong'
  | 'useful'
  | 'harmful'
  | 'redundant'
  | 'weak'
  | 'mixed'
  | 'insufficient'

export interface MetricCatalogEntry {
  key: string
  label: string
  group: string
  description: string
  targets: string[]
  positions: string[]
  prediction_target: string | null
  available_from_forecast: number | null
  expected_sign: 'positive' | 'negative' | 'either'
  available: boolean
  coverage: number | null
}

export interface MetricFoldResult {
  forecast_season: number
  n: number
  spearman: number | null
}

export interface MetricBacktestResult {
  metric: string
  label: string
  group: string
  window: string
  window_label: string
  target: string
  target_label: string
  position: string
  n: number
  eligible: number
  coverage: number | null
  folds: number
  spearman: number | null
  spearman_ci_low: number | null
  spearman_ci_high: number | null
  pearson: number | null
  partial_spearman: number | null
  direction_consistency: number | null
  assessment: MetricAssessment
  fold_results: MetricFoldResult[]
}

export interface MetricModelResult {
  metric: string
  label: string
  window: string
  window_label: string
  target: string
  target_label: string
  position: string
  n: number
  mae: number
  rmse: number
  bias: number
  spearman: number | null
}

export interface MetricRankingFoldResult {
  forecast_season: number
  n: number
  hit_rate: number
  ndcg: number | null
  pool_spearman: number | null
  top_k_actual_mean: number
  ideal_top_k_actual_mean: number
}

export interface MetricRankingResult {
  ranker: string
  role: 'baseline' | 'candidate'
  window: string
  window_label: string
  target: string
  position: Position
  k: number
  pool: number
  folds: number
  hit_rate: number
  ndcg: number | null
  pool_spearman: number | null
  top_k_actual_mean: number
  ideal_top_k_actual_mean: number
  best_baseline: string | null
  best_ndcg_baseline: string | null
  /** The best baseline's hit rate on the candidate's own folds. */
  baseline_hit_rate: number | null
  hit_rate_lift: number | null
  hit_rate_lift_se: number | null
  hit_rate_lift_ci_low: number | null
  hit_rate_lift_ci_high: number | null
  folds_won: number | null
  folds_tied: number | null
  folds_lost: number | null
  ndcg_lift: number | null
  folds_beating_baseline: number | null
  beats_baseline: boolean | null
  fold_results: MetricRankingFoldResult[]
}

export interface MetricReport {
  schema_version: number
  title: string
  generated_at: string
  configuration: {
    forecast_seasons: number[]
    input_seasons: number[]
    history_seasons: number
    depth_chart_cutoff: string
    depth_chart_start_season: number
    minimum_sample: number
    baseline_metric: string
    analysis_windows: Array<{ key: string; label: string; start: number; end: number }>
    ranking: {
      baselines: string[]
      baselines_by_target: Record<string, string[]>
      candidates: string[]
      targets: string[]
      top_k: Record<Position, number>
      pool: Record<Position, number>
    }
    fit?: {
      models: Array<Record<string, unknown>>
      outputs: string[]
      min_train_folds: number
      inner_validation_folds: number
    }
  }
  data_summary: {
    completed_forecasts: number[]
    pending_forecasts: number[]
    forecast_rows: number
    completed_rows: number
    returning_player_scope: boolean
    assessment_counts: Record<string, number>
    limitations: string[]
  }
  targets: Array<{
    key: string
    label: string
    description: string
    missing_as_zero: boolean
  }>
  metrics: MetricCatalogEntry[]
  results: MetricBacktestResult[]
  model_results: MetricModelResult[]
  ranking_results: MetricRankingResult[]
  fitted_models: FittedModelRecord[]
  fitted_model_summary: FittedModelSummary[]
  fitted_artifact: FittedArtifact | Record<string, never>
}

export interface FittedModelRecord {
  model: string
  forecast_season: number
  position: Position
  scope: 'position' | 'pooled'
  train_seasons: [number, number]
  n_train: number
  ridge_lambda: number
  clip: [number | null, number | null]
  intercept: number
  coefficients: Record<string, number>
  standardization: Record<string, { mean: number; scale: number }>
}

export interface FittedArtifact {
  schema_version: number
  model_version: string
  fingerprint: string
  fit_config: Record<string, unknown>
  created_at: string
  pending_forecast_season: number | null
  pending_rows: number
  completed_forecast_seasons: number[]
  depth_chart_cutoff: string
  depth_chart_latest: string | null
}

export interface FittedModelSummary {
  model: string
  position: Position
  latest_forecast_season: number
  scope: 'position' | 'pooled'
  n_train: number
  ridge_lambda: number | null
  lambdas_chosen: number[]
  intercept: number
  coefficients: Record<string, number>
  coefficient_sd_across_refits: Record<string, number>
  refits: number
}

// ---------------------------------------------------------------- league state

/** Whether ESPN says a player is owned, claimable, or was never mentioned. */
export type Availability = 'rostered' | 'free_agent' | 'unknown'

/** Freshness metadata on every league response.
 *
 * Attached to everything because data without its age is how a stale wire gets acted
 * on. `stale` means the last refresh failed and this is the previous good snapshot.
 */
export interface Freshness {
  age_seconds: number
  stale: boolean
  week: number
  season: number
}

/** A board row cross-referenced with live league state. */
export interface LeaguePlayer extends Player {
  availability: Availability
  is_free_agent: boolean
  is_mine: boolean
  owner_team_id: number | null
  owner_team_name: string | null
  espn_team: string | null
  injury_status: string | null
  percent_owned: number | null
  percent_started: number | null
  /** True when ESPN's current team disagrees with the team the metrics came from. */
  changed_team: boolean
  /** Value against the free-agent pool rather than the preseason board. Wire only. */
  wire_vor?: number
}

export interface LeagueTeam {
  team_id: number
  team_name: string
  owner: string | null
  wins: number
  losses: number
  faab_remaining: number | null
  is_mine: boolean
  division_id: number | null
  division_name: string | null
  /** League-relative roster strength: 5.0 is average, 1.5 points is one SD. */
  team_score: number
  team_rank: number
  scored_players: number
  /** Players without nflverse tape whose ESPN projection supplied the fallback. */
  fallback_players: number
  /** Availability-aware mean from the best legal active skill-position lineup. */
  expected_weekly_points: number
  /** Standard deviation of weekly skill-position lineup points. */
  weekly_risk: number
  /** Approximate 25th-percentile weekly skill-position score. */
  weekly_floor: number
  /** Probability the roster can fill every skill-position starting slot. */
  lineup_coverage: number
  /** Expected weekly points supplied by players outside the full-strength lineup. */
  bench_rescue_points: number
}

export interface LeagueResponse extends Freshness {
  league_id: number
  league_name: string
  my_team_id: number | null
  teams: LeagueTeam[]
}

export interface LeaguePlayersResponse extends Freshness {
  total: number
  offset: number
  limit: number
  players: LeaguePlayer[]
}

export interface RosterResponse extends Freshness {
  team_id: number
  team_name: string | null
  players: LeaguePlayer[]
}

export interface WireResponse extends Freshness {
  /** What replacement level actually is right now, given who is still free. */
  replacement_levels: Record<string, number>
  total: number
  players: LeaguePlayer[]
}

export interface UnrankablePlayer {
  player_display_name: string
  position: string
  espn_team: string | null
  percent_owned: number | null
  owner_team_name: string | null
  injury_status: string | null
}

export interface UnrankableResponse extends Freshness {
  total: number
  players: UnrankablePlayer[]
}

export interface OpponentRow {
  owner_team_name: string
  position: string
  best_vor: number
  best_player: string
  depth: number
}

export interface OpponentsResponse extends Freshness {
  teams: OpponentRow[]
}

export interface Transaction {
  date: string | null
  kind: string | null
  team_name: string | null
  player_name: string | null
  bid_amount: number | null
}

export interface TransactionsResponse extends Freshness {
  total: number
  transactions: Transaction[]
}

export interface LeagueStatus {
  authenticated: boolean
  synced: boolean
  age_seconds: number | null
  stale: boolean
  ttl_seconds: number
  league_name?: string
  season?: number
  week?: number
  my_team_id?: number | null
  join?: {
    matched: number
    total: number
    match_rate: number
    unmatched: number
    not_ranked: number
  }
}

// ---------------------------------------------------------------- matchups

export interface LineupSlot {
  slot: string
  player_id: string | null
  player_display_name: string | null
  position: string | null
  value: number
}

/** A team's best fieldable lineup and what it is worth.
 *
 * Not a weekly projection: it knows nothing about byes, this week's injury report, or
 * opponent. It answers "who has the better team" — the question a schedule scan and a
 * trade conversation both ask. Kicker and defense slots are excluded because this
 * league tiers those rather than ranking them.
 */
export interface TeamStrength {
  team_id: number
  team_name: string
  /** Which board value the comparison uses, e.g. v2_overall_vor. */
  metric: string
  total: number
  by_position: Record<string, number>
  /** Rostered skill players with no board value, so the total understates this team. */
  unranked_starters: number
  starters: LineupSlot[]
  bench: LineupSlot[]
}

export interface Matchup {
  home: TeamStrength
  away: TeamStrength
  involves_me: boolean
  margin: number
}

export interface MatchupsResponse extends Freshness {
  requested_week: number
  current_week: number
  regular_season_weeks: number
  my_team_id: number | null
  matchups: Matchup[]
}

export interface CompareResponse extends Freshness {
  left: TeamStrength
  right: TeamStrength
  margin: number
}
