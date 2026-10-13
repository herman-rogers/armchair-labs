import type { LeagueObservations, ObservedMatchups, WeeklyForecasts } from './api/league'

/** Never combine campaign points with another week or observed lineup capture. */
export function currentWeeklyForecasts(data: LeagueObservations, forecasts?: WeeklyForecasts, matchups?: ObservedMatchups) {
  if (!forecasts || !['campaign_per_game', 'campaign_weekly_distribution'].includes(forecasts.source) || forecasts.season !== data.season
    || forecasts.current_week !== data.week || forecasts.requested_week !== data.week
    || forecasts.captured_at !== data.captured_at) return undefined
  if (matchups && (matchups.season !== data.season || matchups.current_week !== data.week
    || matchups.requested_week !== data.week || matchups.captured_at !== forecasts.captured_at)) return undefined
  return forecasts
}

export const weeklyForecastLabel = (forecasts?: WeeklyForecasts) => forecasts ? `${forecasts.campaign_name} + ESPN K/DST` : 'Lineup projection'
