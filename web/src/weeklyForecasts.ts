import type { LeagueObservations, ObservedMatchups, WeeklyForecasts } from './api/nextgen'

/** Never combine model points with another release, week, or observed lineup. */
export function currentWeeklyForecasts(data: LeagueObservations, forecasts?: WeeklyForecasts, matchups?: ObservedMatchups) {
  if (!forecasts || !['nextgen_weekly_reference', 'nextgen_weekly_model'].includes(forecasts.source)
    || forecasts.version !== data.version || forecasts.season !== data.season
    || forecasts.current_week !== data.week || forecasts.requested_week !== data.week
    || forecasts.through_week !== data.week - 1 || forecasts.captured_at !== data.captured_at) return undefined
  if (matchups && (matchups.season !== data.season || matchups.current_week !== data.week
    || matchups.requested_week !== data.week || matchups.captured_at !== forecasts.captured_at)) return undefined
  return forecasts
}

export const weeklyForecastLabel = (forecasts?: WeeklyForecasts) => forecasts?.source === 'nextgen_weekly_model' && forecasts.evidence_status !== 'reference' ? 'NextGen one-week model' : 'NextGen weekly reference'
