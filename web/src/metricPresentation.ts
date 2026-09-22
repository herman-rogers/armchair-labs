import type { Player } from './api/types'

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
  actual_ppg: 'next-season PPG',
  actual_season_points: 'next-season points',
}

export function rankerLabel(value: string) {
  return RANKER_LABELS[value] ?? value.replaceAll('_', ' ')
}

export function boardMetricLabel(value: string) {
  const labels: Record<string, string> = {
    v2_overall_vor: 'overall projection VOR',
    v2_rank_vor: 'position projection VOR',
    adj_proj_vor: 'adjusted projected VOR',
      adj_vor: 'adjusted historical VOR',
    vor: 'historical VOR',
    ppg: 'historical PPG',
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

export function rankBasisTitle(player: Player) {
  if (!player.v2_rank_key) return 'No position-specific rank key was available.'
  if (player.v2_rank_key === 'market') {
    return 'No usable tape for the model. Placed by rank-matching the FantasyPros consensus: the value shown is the median model value of the three rated players at this position whose market rank is nearest. Tagged market, not imputed as a model score.'
  }
  if (player.v2_rank_key === 'espn_ppr_rank') {
    return 'No prior NFL production is available to Patron. ESPN’s current PPR draft-room ordering supplies this explicitly flagged fallback; no model score was imputed.'
  }
  return `Position rank uses ${rankerLabel(player.v2_rank_key)}, expressed above that model’s positional replacement and then adjusted by any manual override. This cell shows the raw model basis.`
}

export const OVERALL_VOR_TITLE =
  'Overall board value: the selected forecast system’s season score (as a per-game rate) above its positional replacement, plus any manual override. This common scale drives the overall # rank.'

export const POSITION_RANK_TITLE =
  'Rank within the position using that position’s backtest-selected projection. See Rank basis for the model and value used on this row.'

export const SEASON_EQUIVALENT_TITLE =
  'Selected forecast season points converted to a per-scheduled-game rate. Unlike active-game PPG, this includes fitted availability and is the value used for waiver comparisons.'
