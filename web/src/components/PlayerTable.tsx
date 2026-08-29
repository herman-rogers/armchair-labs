import { useEffect, useMemo, useState } from 'react'
import type { LeaguePlayer } from '../api/types'
import { Flags } from './Flags'
import { InjuryBadge, MovedBadge, OwnerBadge } from './Availability'

type Direction = 'asc' | 'desc'

export interface PlayerColumn {
  key: keyof LeaguePlayer
  label: string
  title: string
  align?: 'left'
  initial?: Direction
  render?: (player: LeaguePlayer) => React.ReactNode
}

export const number = (value: number | null | undefined, digits = 1) =>
  value === null || value === undefined ? <span className="faint">—</span> : value.toFixed(digits)

export const percent = (value: number | null | undefined, scale = 100) =>
  value === null || value === undefined ? (
    <span className="faint">—</span>
  ) : (
    `${(value * scale).toFixed(0)}%`
  )

/** Columns shared by every league-aware table. */
export const IDENTITY: PlayerColumn[] = [
  {
    key: 'rank',
    label: '#',
    title: 'Overall board rank. V1 uses adjusted historical VOR; V2 uses the common-scale overall projection VOR.',
    initial: 'asc',
    render: (p) => <span className="rank">{p.rank}</span>,
  },
  {
    key: 'player_display_name',
    label: 'Player',
    title: 'Player name, with live league status.',
    align: 'left',
    initial: 'asc',
    render: (p) => (
      <span className="player-cell">
        <span className="name">{p.player_display_name}</span>
        <InjuryBadge status={p.injury_status} />
        <MovedBadge player={p} />
        {p.override_reason ? (
          <span className="override" title={p.override_reason}>
            *
          </span>
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
    key: 'espn_team',
    label: 'Tm',
    title: 'Current NFL team, per ESPN.',
    align: 'left',
    initial: 'asc',
    render: (p) => <span className="team">{p.espn_team ?? p.team ?? '—'}</span>,
  },
]

export const OWNERSHIP: PlayerColumn = {
  key: 'availability',
  label: 'Status',
  title: 'Who owns this player in your league right now.',
  align: 'left',
  initial: 'asc',
  render: (p) => <OwnerBadge player={p} />,
}

export const ROSTERED_PERCENT: PlayerColumn = {
  key: 'percent_owned',
  label: '%Rost',
  title: 'Share of ESPN leagues rostering this player — the market’s opinion, not yours.',
  render: (p) => <span className="dim">{number(p.percent_owned, 0)}</span>,
}

export const FLAGS_COLUMN: PlayerColumn = {
  key: 'flags',
  label: 'Flags',
  title: 'Historical heuristics and sample warnings. Hover any chip for its limits.',
  align: 'left',
  initial: 'asc',
  render: (p) => <Flags value={p.flags} />,
}

/**
 * A sortable table over league-aware player rows.
 *
 * Deliberately separate from BoardTable: that one renders the pure metric board and
 * switches column sets by metric version. This one renders league views, where the
 * columns differ per screen (a wire needs value-against-the-pool, a roster needs
 * injury and bye) and every row carries ownership.
 */
export function PlayerTable({
  players,
  columns,
  defaultSort,
  emptyMessage = 'Nothing to show.',
}: {
  players: LeaguePlayer[]
  columns: PlayerColumn[]
  defaultSort: keyof LeaguePlayer
  emptyMessage?: string
}) {
  const [sortKey, setSortKey] = useState<keyof LeaguePlayer>(defaultSort)
  const [direction, setDirection] = useState<Direction>('desc')

  // This component stays mounted when the metric tab changes. Resetting is required:
  // otherwise V2 can keep sorting on V1's adj_vor (or V1 can keep V2's score), making
  // two genuinely different boards appear identical.
  useEffect(() => {
    setSortKey(defaultSort)
    setDirection('desc')
  }, [defaultSort])

  const sorted = useMemo(() => {
    const rows = [...players]
    rows.sort((a, b) => {
      const left = a[sortKey]
      const right = b[sortKey]
      // Nulls sort last in both directions: an unknown value is not a small one, and
      // floating them to the top would misrepresent the ranking.
      if (left == null && right == null) return 0
      if (left === null || left === undefined) return 1
      if (right === null || right === undefined) return -1

      const comparison =
        typeof left === 'number' && typeof right === 'number'
          ? left - right
          : String(left).localeCompare(String(right))
      return direction === 'asc' ? comparison : -comparison
    })
    return rows
  }, [players, sortKey, direction])

  if (!players.length) {
    return <div className="notice">{emptyMessage}</div>
  }

  const onSort = (column: PlayerColumn) => {
    if (column.key === sortKey) {
      setDirection((d) => (d === 'asc' ? 'desc' : 'asc'))
    } else {
      setSortKey(column.key)
      setDirection(column.initial ?? 'desc')
    }
  }

  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            {columns.map((column) => (
              <th
                key={String(column.key)}
                className={column.align === 'left' ? 'left' : undefined}
                title={column.title}
                onClick={() => onSort(column)}
              >
                {column.label}
                {sortKey === column.key && (
                  <span className="dir">{direction === 'asc' ? '↑' : '↓'}</span>
                )}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sorted.map((player) => (
            <tr key={player.player_id} className={player.is_mine ? 'mine-row' : undefined}>
              {columns.map((column) => (
                <td
                  key={String(column.key)}
                  className={column.align === 'left' ? 'left' : undefined}
                >
                  {column.render ? column.render(player) : String(player[column.key] ?? '—')}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
