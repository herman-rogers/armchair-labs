import { SKILL_POSITIONS, isSkillPosition } from './positions'
import type { LeagueObservations, Ranking, RankingsResponse } from './api/nextgen'

const reserve = new Set(['BE', 'BN', 'IR', 'ER'])
export function forecastFor(playerId: string | null | undefined, rankings?: RankingsResponse) {
  return rankings?.rankings.find(r => r.player_id === playerId && Number.isFinite(r.prediction))
}
export function forecastBasis(r?: Ranking) {
  return !r ? 'Unavailable' : r.evidence_status === 'validated_forecast' ? 'Historically validated' : 'Reference'
}
export function leagueSummary(data: LeagueObservations, rankings?: RankingsResponse) {
  if (rankings?.report.season !== data.season) rankings = undefined
  const byId = new Map(rankings?.rankings.filter(r => Number.isFinite(r.prediction)).map(r => [r.player_id, r]))
  const teams = data.teams.map(team => {
    const roster = data.players.filter(p => p.owner_team_id === team.team_id)
    const skills = roster.filter(p => isSkillPosition(p.position))
    const rated = skills.map(p => ({player: p, forecast: byId.get(p.player_id ?? '')})).filter(p => p.forecast != null)
    const overallRanks = rated.flatMap(p => {
      const rank = p.forecast?.overall_rank
      return rank != null && Number.isFinite(rank) ? [rank] : []
    })
    const knownPoints = rated.reduce((sum, p) => sum + p.forecast!.prediction, 0)
    const complete = skills.length > 0 && rated.length === skills.length && rankings != null
    const schedule = (team.schedule ?? []).filter(g => g.week < data.week && ['W','L','T'].includes(g.outcome))
    const against = schedule.map(g => data.teams.find(t => t.team_id === g.opponent_team_id)?.schedule?.find(o => o.week === g.week && o.opponent_team_id === team.team_id && ['W','L','T'].includes(o.outcome))?.score)
    const last = [...schedule].sort((a,b) => b.week - a.week)[0]
    return {...team, roster, skill_count: skills.length, forecast_count: rated.length,
      average_overall_rank: overallRanks.length ? overallRanks.reduce((sum, rank) => sum + rank, 0) / overallRanks.length : null,
      ranked_count: overallRanks.length,
      forecast_points: complete ? knownPoints : null, known_points: rated.length ? knownPoints : null,
      nextgen_team_rank: null as number | null,
      reference_count: rated.filter(p => p.forecast!.evidence_status !== 'validated_forecast').length,
      constrained_count: rated.filter(p => p.forecast!.constraint).length,
      unmodeled_count: roster.length - skills.length,
      alerts: roster.filter(p => p.attention?.length).length,
      urgent: roster.filter(p => p.attention?.some(f => f.severity === 'urgent')).length,
      points_for: schedule.length ? schedule.reduce((sum,g) => sum + g.score, 0) : null,
      points_against: schedule.length && against.every(p => p != null) ? against.reduce<number>((sum,p) => sum + p!, 0) : null,
      results_captured: schedule.length,
      last_result: last ? `W${last.week} ${last.outcome} · ${last.score.toFixed(1)} vs ${data.teams.find(t => t.team_id === last.opponent_team_id)?.team_name ?? 'Unknown opponent'}` : 'No completed result captured',
      starter_points: rated.filter(p => p.player.lineup_slot && !reserve.has(p.player.lineup_slot)).reduce((sum,p) => sum+p.forecast!.prediction,0),
      positions: SKILL_POSITIONS.map(position => {
        const pool = skills.filter(p => p.position === position)
        const forecasts = rated.filter(p => p.player.position === position)
        return {position, players:pool.length, covered: forecasts.length,
          points: forecasts.length ? forecasts.reduce((sum,p) => sum+p.forecast!.prediction,0) : null}
      }),
    }
  })
  const scores = teams.flatMap(t => t.forecast_points == null ? [] : [t.forecast_points]).sort((a,b)=>b-a)
  return teams.map(t => ({...t, nextgen_team_rank: t.forecast_points == null ? null : scores.indexOf(t.forecast_points)+1}))
}
export function recentTransactions(data: LeagueObservations) {
  return [...(data.transactions ?? [])].sort((a,b) => (Date.parse(b.date ?? '') || 0) - (Date.parse(a.date ?? '') || 0))
}
