import { useQuery } from '@tanstack/react-query'
import { useUrlState } from '../navigation'
import { boardQuery } from '../api/queries/archive'
import { BoardTable } from './BoardTable'

/** Saved comparison artifacts, independent of current ESPN ownership and added players. */
export function ReferenceBoard({ version }: { version: 'v1' | 'adaptive' }) {
  const [search, setSearch] = useUrlState('board_q', '', { replace: true })
  const board = useQuery(boardQuery(version, 1000))

  if (board.isError) return <div className="notice">{board.error.message}</div>
  if (!board.data) return <div className="notice">Loading the saved board…</div>

  const needle = search.trim().toLowerCase()
  const players = board.data.players.filter(player =>
    player.player_display_name.toLowerCase().includes(needle))

  return <>
    <div className="controls">
      <input type="search" aria-label="Search reference players" placeholder="Search players…"
        value={search} onChange={event => setSearch(event.target.value)} />
      <span className="count">{players.length} saved players</span>
    </div>
    <BoardTable players={players} version={version} />
  </>
}
