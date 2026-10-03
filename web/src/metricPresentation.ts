import type { MetricReport, MetricVersion, Player } from './api/types'

const RANKER_LABELS: Record<string, string> = {
  historical_ppg_prior: 'multi-year PPG',
  ppg: 'last-season PPG',
  season_pts: 'last-season points',
  proj_ppg: 'V2 projected PPG',
  prior_branch_ppg: 'historical-branch PPG',
  component_proj_ppg: 'stat-line projected PPG',
  fitted_ppg: 'fitted projected PPG',
  fitted_season_points: 'fitted season points',
  fitted_season_points_direct: 'fitted season points (direct)',
  fitted_two_stage: 'fitted season points (two-stage)',
  adaptive_season_points: 'frozen Adaptive season points',
  market_ecr_score: 'Market ECR',
  market_overall_ecr_score: 'Overall market ECR',
  fitted_adaptive_ppg_hybrid: 'Adaptive PPG selector',
  fitted_adaptive_season_hybrid: 'Adaptive season selector',
  fitted_rich_weekly_combined: 'Rich weekly forecast',
  actual_availability_value: 'Availability-adjusted value',
  actual_ppg: 'next-season PPG',
  actual_season_points: 'next-season points',
}

export function rankerLabel(value: string) {
  return signalLabel(value, RANKER_LABELS[value] ?? value.replaceAll('_', ' '))
}

export function signalLabel(key: string, label: string) {
  if (/route_participation/.test(key)) return label.replace(/route(?:[- ]opportunity)? participation/ig, 'dropback on-field share (proxy)')
  if (/targets_per_route|tprr/.test(key)) return label.replace(/targets? (?:rate )?per route(?: opportunity)?|tprr/ig, 'targets per on-field dropback (proxy)')
  if (/route_opportunities/.test(key)) return label.replace(/route opportunit/ig, 'on-field dropback opportunit')
  return label
}

/** Display aliases only: frozen columns and report bytes retain their original IDs. */
export function presentMetricReport(report: MetricReport): MetricReport {
  return {
    ...report,
    metrics: report.metrics.map(entry => ({ ...entry, label: signalLabel(entry.key, entry.label),
      description: /route_participation|targets_per_route|route_opportunities|tprr/.test(entry.key)
        ? 'Passing-play on-field participation proxy, not verified routes run. Backs and tight ends may be blocking. Targets per opportunity use on-field dropbacks as the denominator. Missing or incomplete feed coverage is unknown; inspect the version policy.'
        : entry.description })),
    results: report.results.map(row => ({ ...row, label: signalLabel(row.metric, row.label) })),
  }
}

export function boardMetricLabel(value: string) {
  const labels: Record<string, string> = {
    v2_overall_vor: 'overall projection VOR',
    v2_rank_vor: 'position projection VOR',
    adj_proj_vor: 'adjusted projected VOR',
    adj_vor: 'adjusted historical VOR',
    vor: 'historical VOR',
    ppg: 'historical PPG',
    // Not a board metric: what a played week's lineup actually scored.
    points: 'fantasy points',
  }
  return labels[value] ?? value.replaceAll('_', ' ')
}

/** The actual configured input to a player's within-position rank. */
export function rankBasis(player: Player) {
  const key = player.v2_rank_key
  if (!key) return '—'
  if (key === 'market') {
    return player.market_ecr != null ? `ECR ${player.market_position ?? ''}${Math.round(player.market_ecr)}` : 'market'
  }
  if (key === 'espn_ppr_rank' && 'espn_draft_rank' in player) {
    const rank = player.espn_draft_rank
    return typeof rank === 'number' ? `ESPN #${rank}` : 'ESPN unranked'
  }

  const raw = player[key as keyof Player]
  if (typeof raw !== 'number') return '—'
  return key.endsWith('season_points') ? `${raw.toFixed(0)} pts` : `${raw.toFixed(1)}/g`
}

export const OVERALL_VOR_TITLE =
  'Overall board value: the selected forecast system’s season score (as a per-game rate) above its positional replacement, plus any manual override. This common scale drives the overall # rank.'

export const POSITION_RANK_TITLE =
  'Rank within the position using that position’s backtest-selected projection. See Rank basis for the model and value used on this row.'

export const SEASON_EQUIVALENT_TITLE =
  'Selected forecast season points converted to a per-scheduled-game rate. Unlike active-game PPG, this includes fitted availability and is the value used for waiver comparisons.'

export const SYSTEM_LABELS: Record<MetricVersion, string> = {
  v1: 'Draft reference', v2: 'Production forecast', adaptive: 'Frozen experiment',
}

/** Explicit nulls in the canonical contract stay null; older artifacts use fitted fields. */
export function forecastValues(player: Player, version: MetricVersion = player.metric_version) {
  return {
    activePPG: version === 'adaptive' ? null
      : player.forecast_active_ppg !== undefined ? player.forecast_active_ppg : player.fitted_ppg,
    expectedGames: version === 'adaptive' ? null
      : player.forecast_expected_games !== undefined ? player.forecast_expected_games : player.fitted_games,
    seasonPoints: player.forecast_season_points !== undefined ? player.forecast_season_points
      : version === 'adaptive' ? player.adaptive_season_points : player.fitted_season_points,
  }
}

export function forecastSource(player: Player, version: MetricVersion = player.metric_version) {
  if (player.rank_source === 'espn_ppr') return 'ESPN rank only'
  if (version === 'v1') return 'Draft reference'
  if (player.rank_source === 'market' || player.forecast_source === 'market_rank_match') return 'Market estimate'
  if (player.forecast_status === 'unavailable') return 'Unavailable'
  if (version === 'adaptive') return 'Frozen experiment'
  if (player.forecast_source === 'production_returner_v1') return 'Production'
  return player.forecast_source ? rankerLabel(player.forecast_source) : 'Unverified forecast'
}
