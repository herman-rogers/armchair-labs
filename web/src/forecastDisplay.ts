import { describeHorizon, type Ranking, type RankingsResponse } from './api/chairstats'
import { rank } from './format'

/** A weekly zero-game forecast is a bye, not a measure of player quality. */
export const isForecastBye = (row?: Ranking) => row?.horizon === 'weekly' && row.scheduled_games === 0
export const forecastRank = (row?: Ranking) => isForecastBye(row) ? 'BYE' : rank(row?.overall_rank)
export const forecastPositionRank = (row?: Ranking) => isForecastBye(row) ? 'BYE'
  : row?.position_rank == null ? '—' : `${row.position ?? ''}${row.position_rank}`

export const forecastPeriod = (row: Pick<Ranking, 'horizon' | 'origin_week'>) =>
  row.horizon === 'weekly' ? `Week ${row.origin_week + 1}` : describeHorizon(row.horizon)

/** Use the saved forecast's week, which may lag the live league week. */
export function rankingsPeriod(rankings?: RankingsResponse) {
  if (!rankings) return 'Forecast'
  const weeks = [...new Set(rankings.models.map(m => m.origin_week + 1))]
  return rankings.horizon === 'weekly'
    ? (weeks.length === 1 ? `Week ${weeks[0]}` : 'Weekly')
    : describeHorizon(rankings.horizon)
}
