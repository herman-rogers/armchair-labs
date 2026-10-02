import type { MetricVersion, Player } from '../api/types'
import { DataTable, type Column as DataColumn } from './DataTable'
import { forecastColumns } from './ForecastColumns'
import { PlayerDetails } from './PlayerDetails'

type Column = DataColumn<Player>

const IDENTITY_COLUMNS: Column[] = [
  {
    key: 'rank',
    label: '#',
    title: 'Overall board rank, after manual overrides where present.',
    align: 'left',
    initial: 'asc',
    render: (p) => <span className="rank">{p.rank}</span>,
  },
  {
    key: 'player_display_name',
    label: 'Player',
    title: 'Player name.',
    align: 'left',
    initial: 'asc',
    render: (p) => (
      <span className="name">
        {p.player_display_name}
        {p.override_reason ? (
          <>
            {' '}
            <span className="override" title={p.override_reason}>
              *
            </span>
          </>
        ) : null}
        {p.projection_reason ? (
          <>
            {' '}
            <span className="override" title={p.projection_reason}>
              ◇
            </span>
          </>
        ) : null}
      </span>
    ),
  },
  {
    key: 'position',
    label: 'Pos',
    title: 'Position.',
    align: 'left',
    initial: 'asc',
    render: (p) => <span className={`pos ${p.position}`}>{p.position}</span>,
  },
  {
    key: 'team',
    label: 'Tm',
    title: 'Team as of the last game of the season.',
    align: 'left',
    initial: 'asc',
    render: (p) => <span className="team">{p.team}</span>,
  },
]

export function BoardTable({ players, version }: { players: Player[]; version: MetricVersion }) {
  return <DataTable key={version} rows={players}
    columns={[...IDENTITY_COLUMNS, ...forecastColumns<Player>(version)]}
    defaultSort="rank" rowKey={player => player.player_id} rowLabel={player => player.player_display_name}
    renderDetails={player => <PlayerDetails player={player} version={version} />}
    emptyMessage="No players match these filters." />
}
