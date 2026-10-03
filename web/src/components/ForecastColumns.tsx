import type { MetricVersion, Player } from '../api/types'
import {
  forecastSource,
  forecastValues,
  OVERALL_VOR_TITLE,
  POSITION_RANK_TITLE,
} from '../metricPresentation'
import type { Column } from './DataTable'
import { fixed } from '../format'

/** Every player surface displays and sorts the same selected forecast. */
export function forecastColumns<Row extends Player>(version: MetricVersion): Column<Row>[] {
  if (version === 'v1')
    return [
      {
        key: 'adj_vor',
        label: 'Draft VOR',
        title: 'Preserved draft value above positional replacement, including manual adjustments.',
        render: (p) => <b>{fixed(p.adj_vor)}</b>,
      },
      {
        key: 'ppg',
        label: 'Historical PPG',
        title: 'League points divided by production-row games in the preserved draft reference.',
        render: (p) => fixed(p.ppg),
      },
      {
        key: 'games',
        label: 'Games',
        title: 'Games counted in the preserved draft reference.',
        render: (p) => fixed(p.games, 0),
      },
      {
        key: 'season_pts',
        label: 'Season points',
        title: 'League points scored in the reference season.',
        render: (p) => fixed(p.season_pts, 0),
      },
    ]
  const columns: Column<Row>[] = [
    {
      key: 'v2_overall_vor',
      label: 'Overall VOR',
      title: OVERALL_VOR_TITLE,
      render: (p) => <b>{fixed(p.v2_overall_vor)}</b>,
    },
    {
      key: 'v2_position_rank',
      label: 'Pos rank',
      title: POSITION_RANK_TITLE,
      initial: 'asc',
      render: (p) => p.v2_position_rank ?? '—',
    },
  ]
  if (version === 'v2')
    columns.push(
      {
        key: 'forecast_active_ppg',
        label: 'Active PPG',
        title: 'Selected production forecast per active game.',
        value: (p) => forecastValues(p, version).activePPG,
        render: (p) => fixed(forecastValues(p, version).activePPG),
      },
      {
        key: 'forecast_expected_games',
        label: 'Expected games',
        title: 'Expected active games underlying the selected season forecast.',
        value: (p) => forecastValues(p, version).expectedGames,
        render: (p) => fixed(forecastValues(p, version).expectedGames),
      },
    )
  columns.push(
    {
      key: 'forecast_season_points',
      label: 'Season points',
      title: 'Selected forecast in league points for the season.',
      value: (p) => forecastValues(p, version).seasonPoints,
      render: (p) => <b>{fixed(forecastValues(p, version).seasonPoints, 0)}</b>,
    },
    {
      key: 'forecast_source',
      label: 'Source',
      title:
        'Where this row’s selected forecast or fallback comes from. Open Details for its cutoff and assumptions.',
      align: 'left',
      value: (p) => forecastSource(p, version),
      render: (p) => (
        <span
          className={`forecast-source ${p.rank_source === 'market' || p.rank_source === 'espn_ppr' ? 'fallback' : ''}`}
        >
          {forecastSource(p, version)}
        </span>
      ),
    },
  )
  return columns
}
