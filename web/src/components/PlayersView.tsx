import { useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { fetchBoard, fetchLeaguePlayers } from '../api/client'
import type { LeaguePlayer, MetricVersion, Position } from '../api/types'
import { BoardTable } from './BoardTable'
import { Freshness } from './Freshness'
import {
  FLAGS_COLUMN,
  IDENTITY,
  OWNERSHIP,
  PlayerTable,
  ROSTERED_PERCENT,
  number,
  type PlayerColumn,
} from './PlayerTable'

const POSITIONS: Position[] = ['QB', 'RB', 'WR', 'TE']
const FLAGS = ['BUY', 'TD-luck', 'age'] as const

type Ownership = 'all' | 'free_agent' | 'rostered' | 'mine'

const OWNERSHIP_FILTERS: { id: Ownership; label: string; hint: string }[] = [
  { id: 'all', label: 'Everyone', hint: 'Every ranked player.' },
  { id: 'free_agent', label: 'Available', hint: 'Confirmed claimable right now.' },
  { id: 'rostered', label: 'Rostered', hint: 'Owned by a team in your league.' },
  { id: 'mine', label: 'Mine', hint: 'On your roster.' },
]

const LEAGUE_COLUMNS: PlayerColumn[] = [
  ...IDENTITY,
  OWNERSHIP,
  {
    key: 'adj_vor',
    label: 'VOR',
    title: 'Points per game above the replacement-level player at this position.',
    render: (p) => <span className="strong">{number(p.adj_vor, 2)}</span>,
  },
  { key: 'ppg', label: 'PPG', title: 'League-scored points per game.', render: (p) => number(p.ppg) },
  FLAGS_COLUMN,
  {
    key: 'floor',
    label: 'Floor',
    title: '25th-percentile weekly score.',
    render: (p) => <span className="dim">{number(p.floor)}</span>,
  },
  {
    key: 'td_over_exp',
    label: 'TDOE',
    title:
      'Touchdowns above or below what the volume implies. Positive is luck that will not repeat; negative with volume is a buy.',
    render: (p) => (
      <span
        style={{
          color:
            p.td_over_exp > 2 ? 'var(--sell)' : p.td_over_exp < -2 ? 'var(--buy)' : undefined,
        }}
      >
        {p.td_over_exp > 0 ? '+' : ''}
        {p.td_over_exp.toFixed(1)}
      </span>
    ),
  },
  ROSTERED_PERCENT,
]

/**
 * Every ranked player, cross-referenced with league status.
 *
 * Falls back to the pure metric board when ESPN is not connected, because the ranking
 * is useful on its own — it just cannot tell you who is available.
 */
export function PlayersView({ version }: { version: MetricVersion }) {
  const [positions, setPositions] = useState<Set<Position>>(new Set())
  const [flag, setFlag] = useState<string | null>(null)
  const [ownership, setOwnership] = useState<Ownership>('all')
  const [search, setSearch] = useState('')

  const league = useQuery({
    queryKey: ['league-players', version],
    queryFn: () => fetchLeaguePlayers(version),
    retry: false,
  })
  // Only fetched when the league join is unavailable, so an unauthenticated user still
  // gets the board rather than an error screen.
  const board = useQuery({
    queryKey: ['board', version],
    queryFn: () => fetchBoard(version),
    enabled: league.isError,
  })

  const connected = !league.isError && !!league.data
  const rows: LeaguePlayer[] = connected
    ? league.data.players
    : ((board.data?.players ?? []) as LeaguePlayer[])

  const filtered = useMemo(() => {
    const needle = search.trim().toLowerCase()
    return rows.filter((player) => {
      if (positions.size && !positions.has(player.position)) return false
      if (flag && !player.flags.includes(flag)) return false
      if (needle && !player.player_display_name.toLowerCase().includes(needle)) return false
      if (connected && ownership !== 'all') {
        if (ownership === 'mine' && !player.is_mine) return false
        if (ownership === 'free_agent' && player.availability !== 'free_agent') return false
        if (ownership === 'rostered' && player.availability !== 'rostered') return false
      }
      return true
    })
  }, [rows, positions, flag, search, ownership, connected])

  const togglePosition = (position: Position) =>
    setPositions((current) => {
      const next = new Set(current)
      if (next.has(position)) next.delete(position)
      else next.add(position)
      return next
    })

  if (league.isLoading) return <div className="notice">Loading players…</div>
  if (!connected && board.isLoading) return <div className="notice">Loading the board…</div>
  if (!connected && board.isError) {
    return <div className="notice">{(board.error as Error).message}</div>
  }

  return (
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
        {connected && (
          <>
            <span style={{ width: 8 }} />
            {OWNERSHIP_FILTERS.map((entry) => (
              <button
                key={entry.id}
                type="button"
                className="chip"
                aria-pressed={ownership === entry.id}
                title={entry.hint}
                onClick={() => setOwnership(entry.id)}
              >
                {entry.label}
              </button>
            ))}
          </>
        )}
        <input
          className="search"
          placeholder="Search players…"
          value={search}
          onChange={(event) => setSearch(event.target.value)}
        />
        <span className="spacer" />
        <span className="count">
          {filtered.length} of {rows.length}
        </span>
      </div>

      {!connected && (
        <p className="legend tight faint">
          Showing the metric board only — ESPN is not connected, so ownership is unknown.
          Sign in with <code>uv run patron auth login</code> to cross-reference against your
          league.
        </p>
      )}

      {connected ? (
        <PlayerTable players={filtered} columns={LEAGUE_COLUMNS} defaultSort="adj_vor" />
      ) : (
        <BoardTable players={filtered} version={version} />
      )}

      {connected && <Freshness data={league.data} />}
    </>
  )
}
