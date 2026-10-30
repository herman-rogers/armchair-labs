import type { LeagueObservations } from './api/league'

const done = (outcome: string) => outcome === 'W' || outcome === 'L' || outcome === 'T'

/** ESPN results only: record, scoring and all-play from completed matchups. */
export function espnStandings(data: LeagueObservations) {
  const teams = data.teams.map(team => {
    const games = (team.schedule ?? []).filter(g => done(g.outcome))
    let beaten = 0, faced = 0
    for (const g of games) for (const other of data.teams) {
      const theirs = other.team_id === team.team_id ? undefined : other.schedule?.find(o => o.week === g.week && done(o.outcome))
      if (!theirs) continue
      faced++
      beaten += g.score > theirs.score ? 1 : g.score === theirs.score ? .5 : 0
    }
    const count = (outcome: string) => games.filter(g => g.outcome === outcome).length
    const points = games.reduce((sum, g) => sum + g.score, 0)
    return { team_id: team.team_id, wins: count('W'), losses: count('L'), ties: count('T'), points,
      ppg: games.length ? points / games.length : null, all_play: faced ? beaten / faced : null }
  })
  type Row = typeof teams[number]
  const rankBy = (value: (t: Row) => number | null) => (t: Row) => {
    const v = value(t)
    return v == null ? null : 1 + teams.filter(o => (value(o) ?? -Infinity) > v).length
  }
  // ESPN's default standings: record (ties count half), then points for.
  const standing = rankBy(t => t.wins + t.ties / 2 + t.points / 1e6)
  const ppgRank = rankBy(t => t.ppg), allPlayRank = rankBy(t => t.all_play)
  return new Map(teams.map(t => [t.team_id, { ...t, standing: standing(t), ppg_rank: ppgRank(t), all_play_rank: allPlayRank(t) }]))
}

export const weekRange = (weeks?: number[]) =>
  !weeks?.length ? '' : weeks.length === 1 ? `Week ${weeks[0]}` : `Weeks ${weeks[0]}–${weeks[weeks.length - 1]}`
