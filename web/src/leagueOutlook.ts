import type { LeagueObservations, WeeklyForecasts, RankingHistory, RankingsResponse } from './api/nextgen'
import { leagueSummary } from './leagueSummary'
import { currentWeeklyForecasts } from './weeklyForecasts'

const finite = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value)
const difference = (current: number | null, previous: number | null) => current != null && previous != null ? current - previous : null

/** Keep forecast movement, observed scoring, and NextGen weekly reference forecasts distinct. */
export function leagueOutlook(data: LeagueObservations, rankings?: RankingsResponse, history?: RankingHistory, matchups?: WeeklyForecasts) {
  if (rankings?.report.season !== data.season || rankings?.horizon !== 'rest_of_season') rankings = undefined
  const summary = leagueSummary(data, rankings)
  // A previous-week publication, not an arbitrary older or same-week revision.
  const prior = rankings && history?.version === rankings.version && history.season === data.season && history.horizon === rankings.horizon
    ? history.snapshots.find(s => s.through_week === rankings.report.through_week - 1) : undefined
  const previousPoints = new Map(prior?.rankings.map(r => [r.player_id, r.prediction]))
  // Re-score today's rosters on both releases. Historical ownership is not inferred.
  const previousTotals = summary.map(t => {
    const players = t.roster.filter(p => ['QB', 'RB', 'WR', 'TE'].includes(p.position))
    const values = players.map(p => previousPoints.get(p.player_id ?? ''))
    return { team_id: t.team_id, total: values.length && values.every(finite) ? values.reduce((a, b) => a + b, 0) : null }
  })
  // Both sides must cover the same teams for an honest rank comparison.
  const comparable = !!prior && summary.every(t => t.forecast_points != null) && previousTotals.every(t => t.total != null)
  const ordered = previousTotals.flatMap(t => t.total == null ? [] : [t.total]).sort((a, b) => b - a)
  const seasonWeeks = data.regular_season_weeks
  const completedWeeks = Math.max(0, Math.min(data.week - 1, seasonWeeks))
  const validMatchups = currentWeeklyForecasts(data, matchups)
  const games = (validMatchups?.matchups ?? []).map(g => {
    const home = finite(g.home.model_projection) ? g.home.model_projection : null
    const away = finite(g.away.model_projection) ? g.away.model_projection : null
    const margin = home != null && away != null ? home - away : null
    return { ...g, home_projection: home, away_projection: away, margin,
      favorite: margin == null || margin === 0 ? null : margin > 0 ? g.home : g.away }
  })
  const teams = summary.map(t => {
    const observations = (t.schedule ?? []).filter(g => g.week >= 1 && g.week <= completedWeeks && ['W', 'L', 'T'].includes(g.outcome) && finite(g.score))
    const average = (through: number) => {
      const selected = observations.filter(g => g.week <= through)
      return through > 0 && selected.length === through && new Set(selected.map(g => g.week)).size === through
        ? selected.reduce((sum, g) => sum + g.score, 0) / through : null
    }
    const ppg = average(completedWeeks), previousPpg = average(completedWeeks - 1)
    const pace = ppg != null ? ppg * seasonWeeks : null
    const previousPace = previousPpg != null ? previousPpg * seasonWeeks : null
    const total = previousTotals.find(p => p.team_id === t.team_id)?.total
    const previousRank = comparable && total != null ? ordered.indexOf(total) + 1 : null
    const game = games.find(g => g.home.team_id === t.team_id || g.away.team_id === t.team_id)
    const side = game?.home.team_id === t.team_id ? game?.home : game?.away
    const opponent = game?.home.team_id === t.team_id ? game?.away : game?.home
    return { ...t, previous_forecast_rank: previousRank,
      forecast_change: previousRank != null && t.nextgen_team_rank != null ? previousRank - t.nextgen_team_rank : null,
      ppg, ppg_change: difference(ppg, previousPpg), season_pace: pace, season_pace_change: difference(pace, previousPace),
      completed_points: ppg != null ? ppg * completedWeeks : null,
      record: observations.length === completedWeeks && ppg != null ? `${observations.filter(g => g.outcome === 'W').length}–${observations.filter(g => g.outcome === 'L').length}${observations.some(g => g.outcome === 'T') ? `–${observations.filter(g => g.outcome === 'T').length}` : ''}` : '—',
      weekly_projection: finite(side?.model_projection) ? side.model_projection : null,
      opponent, game,
    }
  })
  return { teams, games, prior_week: comparable ? prior?.through_week : null, completed_weeks: completedWeeks }
}
