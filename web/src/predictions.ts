/**
 * The prediction ladder: every team-week as a rung from the model's forecast to the
 * ESPN score, paired by matchup so the chart can say who played whom, who lost and
 * whether the model's pick was right.
 *
 * Built client-side from two responses the league pages already hold: the league
 * scoring trends (predicted and actual points per team-week) and the league
 * observations (each team's schedule with opponent and W/L/T outcome). No week is
 * invented: a week appears only when both sources describe it.
 */
import type { LeagueObservations, LeagueTeam } from './api/league'
import type { LeagueScoringResponse } from './api/scoring'

export type PickStatus = 'correct' | 'wrong' | 'tie' | 'pending' | 'none'
export type Outcome = 'won' | 'lost' | 'tie' | null

export interface LadderRung {
  week: number; team_id: number; name: string; code: string; mine: boolean
  predicted: number | null; actual: number | null; outcome: Outcome; opponent_id: number
}
export interface LadderPair {
  week: number; home: LadderRung; away: LadderRung
  /** The team the model has ahead, or null when either forecast is missing or they tie. */
  pick_id: number | null; winner_id: number | null; status: PickStatus
}
export interface LadderWeek {
  week: number; graded: boolean; pairs: LadderPair[]
  correct: number; evaluated: number; ties: number
  predicted_avg: number | null; actual_avg: number | null
}
export interface LadderTeam { team_id: number; name: string; code: string; record: string; mine: boolean }
export interface PredictionLadderModel {
  season: number; through_week: number; current_week: number
  weeks: LadderWeek[]; teams: LadderTeam[]
  picks: { correct: number; evaluated: number; ties: number }
  /** Mean of actual − predicted over graded team-weeks; negative means the model ran high. */
  bias: number | null; mae: number | null; under: number; graded: number
}

const STOPWORDS = new Set(['the', 'of', 'a', 'an', 'and', 'is', 'my', 'so', 'it', 'da', 'in', 'to'])

/** A 2–3 letter code from a team name, for charts too dense to print names. */
export function deriveTeamCode(name: string) {
  const words = name.normalize('NFKD').replace(/[̀-ͯ]/g, '').replace(/['’]/g, '').replace(/[^A-Za-z0-9 ]+/g, ' ').trim().split(/\s+/).filter(Boolean)
  const kept = words.filter(w => !STOPWORDS.has(w.toLowerCase()))
  const initials = (kept.length ? kept : words).map(w => w[0]).join('')
  const code = initials.length >= 3 ? initials.slice(0, 3) : (kept.length ? kept : words).join('').slice(0, 3)
  return (code || '???').toUpperCase()
}

/** Owner-chosen ESPN abbreviations when the snapshot has them, else derived codes, all unique. */
export function teamCodes(teams: LeagueTeam[]) {
  const codes = new Map<number, string>()
  const used = new Set<string>()
  for (const team of teams) {
    const base = team.team_abbrev?.trim().toUpperCase() || deriveTeamCode(team.team_name)
    let code = base, n = 2
    while (used.has(code)) code = `${base.slice(0, 2)}${n++}`
    used.add(code); codes.set(team.team_id, code)
  }
  return codes
}

const mean = (values: number[]) => values.length ? values.reduce((s, v) => s + v, 0) / values.length : null

/** League-wide weekly ranks; incomplete fields stay unranked and ties share rank. */
export function weeklyRankings(week: LadderWeek, previous: LadderWeek | undefined, teamCount: number) {
  const rows = week.pairs.flatMap(pair => [pair.home, pair.away])
  const prior = previous?.pairs.flatMap(pair => [pair.home, pair.away]) ?? []
  const comparable = (pool: LadderRung[], key: 'predicted' | 'actual') => pool.length === teamCount
    && new Set(pool.map(row => row.team_id)).size === teamCount
    && pool.every(row => row[key] != null && Number.isFinite(row[key]) && (key !== 'actual' || row.outcome != null))
  const ranking = (pool: LadderRung[], row: LadderRung, key: 'predicted' | 'actual') =>
    comparable(pool, key) ? 1 + pool.filter(other => other[key]! > row[key]!).length : null
  return rows.map(row => {
    const projectedRank = ranking(rows, row, 'predicted')
    const before = prior.find(other => other.team_id === row.team_id)
    const previousRank = before ? ranking(prior, before, 'predicted') : null
    return { ...row, projectedRank, actualRank: ranking(rows, row, 'actual'),
      rankChange: projectedRank != null && previousRank != null ? previousRank - projectedRank : null,
      actual: row.outcome != null ? row.actual : null,
      error: row.outcome != null && row.actual != null && row.predicted != null ? row.actual - row.predicted : null,
    }
  })
}

function record(team: LeagueTeam, through: number) {
  const played = team.schedule?.filter(e => e.week <= through && 'WLT'.includes(e.outcome)) ?? []
  if (!played.length) return `${team.wins}-${team.losses}`
  const w = played.filter(e => e.outcome === 'W').length, l = played.filter(e => e.outcome === 'L').length, t = played.filter(e => e.outcome === 'T').length
  return `${w}-${l}${t ? `-${t}` : ''}`
}

export function buildPredictionLadder(league: LeagueObservations, scoring: LeagueScoringResponse): PredictionLadderModel {
  const codes = teamCodes(league.teams)
  const byId = new Map(league.teams.map(t => [t.team_id, t]))
  const series = new Map(scoring.teams.map(t => [t.team_id, t]))
  const lastWeek = Math.min(league.week, scoring.last_week)
  const weeks: LadderWeek[] = []
  const rungs: LadderRung[] = []

  for (let week = 1; week <= lastWeek; week++) {
    const seen = new Set<number>()
    const pairs: LadderPair[] = []
    for (const team of league.teams) {
      const entry = team.schedule?.find(e => e.week === week)
      const opponent = entry && byId.get(entry.opponent_team_id)
      if (!entry || !opponent || seen.has(team.team_id) || seen.has(opponent.team_id)) continue
      const theirs = opponent.schedule?.find(e => e.week === week)
      if (!theirs) continue
      seen.add(team.team_id); seen.add(opponent.team_id)
      const rung = (t: LeagueTeam, e: NonNullable<LeagueTeam['schedule']>[number], opp: number): LadderRung => {
        const played = 'WLT'.includes(e.outcome)
        const week_ = series.get(t.team_id)?.weeks.find(w => w.week === week)
        const actual = week_?.actual ?? (played ? e.score : null)
        const row: LadderRung = {
          week, team_id: t.team_id, name: t.team_name, code: codes.get(t.team_id)!, mine: t.team_id === league.my_team_id,
          predicted: week_?.predicted ?? null, actual, opponent_id: opp,
          outcome: !played ? null : e.outcome === 'W' ? 'won' : e.outcome === 'L' ? 'lost' : 'tie',
        }
        rungs.push(row)
        return row
      }
      const home = rung(team, entry, opponent.team_id), away = rung(opponent, theirs, team.team_id)
      const played = home.outcome != null
      const pick_id = home.predicted == null || away.predicted == null || home.predicted === away.predicted ? null
        : home.predicted > away.predicted ? home.team_id : away.team_id
      const winner_id = home.outcome === 'won' ? home.team_id : away.outcome === 'won' ? away.team_id : null
      const status: PickStatus = !played ? (pick_id == null ? 'none' : 'pending')
        : home.outcome === 'tie' ? 'tie' : pick_id == null ? 'none' : pick_id === winner_id ? 'correct' : 'wrong'
      pairs.push({ week, home, away, pick_id, winner_id, status })
    }
    if (!pairs.length) continue
    const weekRungs = pairs.flatMap(p => [p.home, p.away])
    weeks.push({
      week, pairs, graded: pairs.every(p => p.home.outcome != null),
      correct: pairs.filter(p => p.status === 'correct').length,
      evaluated: pairs.filter(p => p.status === 'correct' || p.status === 'wrong').length,
      ties: pairs.filter(p => p.status === 'tie').length,
      predicted_avg: mean(weekRungs.map(r => r.predicted).filter((v): v is number => v != null)),
      actual_avg: mean(weekRungs.map(r => r.actual).filter((v): v is number => v != null)),
    })
  }

  const through = weeks.filter(w => w.graded).reduce((m, w) => Math.max(m, w.week), 0)
  const graded = rungs.filter(r => r.actual != null && r.predicted != null)
  const misses = graded.map(r => r.actual! - r.predicted!)
  return {
    season: league.season, through_week: through, current_week: league.week, weeks,
    teams: league.teams.map(t => ({ team_id: t.team_id, name: t.team_name, code: codes.get(t.team_id)!, record: record(t, through), mine: t.team_id === league.my_team_id })),
    picks: { correct: weeks.reduce((s, w) => s + w.correct, 0), evaluated: weeks.reduce((s, w) => s + w.evaluated, 0), ties: weeks.reduce((s, w) => s + w.ties, 0) },
    bias: mean(misses), mae: mean(misses.map(Math.abs)), under: misses.filter(m => m > 0).length, graded: graded.length,
  }
}
