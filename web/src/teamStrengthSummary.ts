import type { LeagueObservations } from './api/league'
import type { TeamStrengthResponse } from './api/teamStrength'

/** Both league views must use the same snapshot and complete league comparison. */
export function teamStrengthSummary(data: LeagueObservations, response?: TeamStrengthResponse) {
  if (!response || response.season !== data.season || response.week !== data.week
    || response.captured_at !== data.captured_at
    || response.teams.length !== data.teams.length
    || new Set(response.teams.map(t => t.team_id)).size !== data.teams.length
    || data.teams.some(t => !response.teams.some(s => s.team_id === t.team_id))) return undefined
  const comparable = response.forecast_available && response.teams.every(t =>
    t.current.complete && t.current.total != null && Number.isFinite(t.current.total))
  return response.teams.map(team => ({
    ...team,
    is_mine: team.team_id === data.my_team_id,
    weekly_rank: comparable ? 1 + response.teams.filter(t => t.current.total! > team.current.total!).length : null,
    strengths: team.positions.filter(p => p.rank != null && p.rank <= Math.ceil(data.teams.length * .3)),
    weaknesses: team.positions.filter(p => p.rank != null && p.rank > Math.floor(data.teams.length * .7)),
  }))
}
