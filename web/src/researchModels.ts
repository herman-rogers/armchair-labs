import { rankerLabel } from './metricPresentation'

/** Shared display policy only. Never rename IDs in saved research artifacts. */
export const PLAYER_MODELS: Record<string, { label: string; description: string }> = {
  fitted_season_points: { label: 'Core season forecast · returners', description: 'Projected active-game PPG × expected games. The core model recomputed on this dataset; not a replacement for the published board.' },
  fitted_ppg: { label: 'Core active-game PPG · returners', description: 'Expected scoring when active. Availability is separate; this is not a season-total or cross-position draft-value ranking.' },
  fitted_nextgen_season_points: { label: 'Next-gen season forecast · rookies + returners', description: 'Combines separate rookie and returner forecasts into one season-points output. Exploratory, not a promoted model. The separate College → NFL study is not incorporated into this saved model.' },
  fitted_market_availability_season_points: { label: 'Market-informed season forecast · experimental', description: 'Uses dated consensus information for returner availability, plus the separate rookie forecast. Not independent of the market.' },
  fitted_adaptive_ppg_hybrid: { label: 'Adaptive selector · experimental', description: 'Selects a component per position using earlier seasons. Despite the internal PPG name, its output is season points. Not proven to beat consensus.' },
  frozen_adaptive_2026: { label: 'Frozen 2026 Adaptive experiment', description: 'The original locked prospective forecast, not recomputed with the repaired data. Available only under legacy research.' },
}

const COMPONENT_LABELS: Record<string, string> = {
  candidate_games_nonredundant: 'Expected games · reduced-input experiment',
  fitted_games: 'Core expected games · component',
  fitted_return_prob: 'Season appearance probability · component',
  fitted_games_if_played: 'Games conditional on appearing · component',
  fitted_two_stage: 'Two-stage season forecast · experimental',
  fitted_adaptive_season_hybrid: 'Adaptive season selector · experimental',
  fitted_market_availability_returner_season_points: 'Market-informed season forecast · returners only',
  fitted_market_availability_games: 'Market-informed expected games · component',
  fitted_nextgen_ppg: 'Next-gen active-game PPG · rookies + returners',
  fitted_rich_weekly_combined: 'Weekly-history combined season forecast · experimental',
}

export function researchModelLabel(id: string) {
  return PLAYER_MODELS[id]?.label ?? COMPONENT_LABELS[id] ?? rankerLabel(id)
}

export function researchModelDescription(id: string) {
  return PLAYER_MODELS[id]?.description
    ?? 'Experimental output or model component; inspect its units, population coverage, and evidence before comparing players.'
}
