import type { LeagueObservations } from './api/nextgen'

/** Reconstructed standings, not provider playoff seeds. Missing games withhold a week. */
export function standingsHistory(data: LeagueObservations) {
  return Array.from({ length: Math.max(0, data.week - 1) }, (_, index) => {
    const week = index + 1
    const totals = data.teams.map(team => {
      const games = (team.schedule ?? []).filter(g => g.week >= 1 && g.week <= week && ['W', 'L', 'T'].includes(g.outcome))
      const complete = games.length === week && new Set(games.map(g => g.week)).size === week && games.every(g => Number.isFinite(g.score))
      return { team_id: team.team_id, complete, wins: games.reduce((n, g) => n + (g.outcome === 'W' ? 1 : g.outcome === 'T' ? .5 : 0), 0), points: games.reduce((n, g) => n + g.score, 0) }
    })
    const complete = totals.length > 0 && totals.every(t => t.complete)
    const sorted = [...totals].sort((a, b) => b.wins - a.wins || b.points - a.points)
    return { week, teams: totals.map(t => ({ team_id: t.team_id, rank: complete ? sorted.findIndex(s => s.wins === t.wins && s.points === t.points) + 1 : null })) }
  })
}
