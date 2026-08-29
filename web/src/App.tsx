import { useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { fetchBoard, fetchStatus } from './api/client'
import type { Position } from './api/types'
import { BoardTable } from './components/BoardTable'

const POSITIONS: Position[] = ['QB', 'RB', 'WR', 'TE']
const FLAGS = ['BUY', 'TD-luck', 'age'] as const

export default function App() {
  const [positions, setPositions] = useState<Set<Position>>(new Set())
  const [flag, setFlag] = useState<string | null>(null)
  const [search, setSearch] = useState('')

  const status = useQuery({ queryKey: ['status'], queryFn: fetchStatus })
  const board = useQuery({
    queryKey: ['board'],
    queryFn: () => fetchBoard(),
    enabled: status.data?.board_available === true,
  })

  const players = useMemo(() => {
    const rows = board.data?.players ?? []
    const needle = search.trim().toLowerCase()
    return rows.filter((player) => {
      if (positions.size && !positions.has(player.position)) return false
      if (flag && !player.flags.includes(flag)) return false
      if (needle && !player.player_display_name.toLowerCase().includes(needle)) return false
      return true
    })
  }, [board.data, positions, flag, search])

  const togglePosition = (position: Position) =>
    setPositions((current) => {
      const next = new Set(current)
      if (next.has(position)) next.delete(position)
      else next.add(position)
      return next
    })

  const league = status.data?.league

  return (
    <div className="app">
      <header className="masthead">
        <h1>{league?.name ?? 'Patron Saints'} — Draft Board</h1>
        <p>
          {league
            ? `${league.team_count}-team, full PPR with big-play bonuses. Built from ${league.seasons.join('/')} tape for the ${league.draft_season} season.`
            : 'Loading league configuration…'}
        </p>
      </header>

      {status.isError && (
        <div className="notice">
          <h2>Cannot reach the API</h2>
          Start it with <code>just api</code> (or{' '}
          <code>uv run uvicorn patron.api.app:app --port 8000</code>).
        </div>
      )}

      {status.data && !status.data.board_available && (
        <div className="notice">
          <h2>No board built yet</h2>
          Run <code>just board</code> to pull nflverse data and generate one, then reload
          this page.
        </div>
      )}

      {board.isError && (
        <div className="notice">
          <h2>Could not load the board</h2>
          {(board.error as Error).message}
        </div>
      )}

      {board.isLoading && status.data?.board_available && (
        <div className="notice">Loading the board…</div>
      )}

      {board.data && (
        <>
          <div className="controls">
            {POSITIONS.map((position) => (
              <button
                key={position}
                type="button"
                className="chip"
                aria-pressed={positions.has(position)}
                onClick={() => togglePosition(position)}
              >
                {position}
              </button>
            ))}

            <span style={{ width: 8 }} />

            {FLAGS.map((name) => (
              <button
                key={name}
                type="button"
                className="chip"
                aria-pressed={flag === name}
                onClick={() => setFlag((current) => (current === name ? null : name))}
              >
                {name}
              </button>
            ))}

            <input
              className="search"
              placeholder="Search players…"
              value={search}
              onChange={(event) => setSearch(event.target.value)}
            />

            <span className="spacer" />
            <span className="count">
              {players.length} of {board.data.total}
            </span>
          </div>

          <BoardTable players={players} />

          <p className="legend">
            <b>VOR</b> — points per game above the replacement-level player at the position
            {league
              ? ` (QB${league.vor_baseline_rank.QB} · RB${league.vor_baseline_rank.RB} · WR${league.vor_baseline_rank.WR} · TE${league.vor_baseline_rank.TE} in a ${league.team_count}-team league)`
              : ''}
            . This is what the board sorts by, and why a tight end can outrank a receiver
            who scores more.
            <br />
            <b>*</b> beside a name marks a manual override — hover it for the reason.
            Hover any column header for what the metric means.
          </p>
        </>
      )}
    </div>
  )
}
