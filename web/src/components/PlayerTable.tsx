import type { LeaguePlayer, MetricVersion } from '../api/types'
import { DataTable } from './DataTable'
import type { PlayerColumn } from './playerColumns'
import { PlayerDetails } from './PlayerDetails'

/**
 * League-aware player rows on the common DataTable. The columns differ per screen
 * (a wire needs value-against-the-pool, a roster needs injury and bye) and every
 * row carries ownership.
 */
export function PlayerTable({
  players,
  columns,
  defaultSort,
  emptyMessage = 'Nothing to show.',
  version,
}: {
  players: LeaguePlayer[]
  columns: PlayerColumn[]
  defaultSort: keyof LeaguePlayer
  emptyMessage?: string
  version: MetricVersion
}) {
  return (
    <DataTable
      key={version}
      rowLabel={player => player.player_display_name}
      renderDetails={player => <PlayerDetails player={player} version={version} />}
      rows={players}
      columns={columns}
      defaultSort={defaultSort}
      rowKey={(player) => player.player_id}
      rowClass={(player) => (player.is_mine ? 'mine-row' : undefined)}
      emptyMessage={emptyMessage}
    />
  )
}
