import { useMemo, useState } from 'react'
import type { Player } from '../api/types'
import { Flags } from './Flags'

type SortKey = keyof Player
type Direction = 'asc' | 'desc'

interface Column {
  key: SortKey
  label: string
  title: string
  align?: 'left'
  /** Default sort direction when this column is first clicked. */
  initial?: Direction
  render?: (player: Player) => React.ReactNode
}

const number = (value: number | null, digits = 1) =>
  value === null || value === undefined ? <span className="faint">—</span> : value.toFixed(digits)

const percent = (value: number | null) =>
  value === null || value === undefined ? (
    <span className="faint">—</span>
  ) : (
    `${(value * 100).toFixed(0)}%`
  )

const COLUMNS: Column[] = [
  {
    key: 'rank',
    label: '#',
    title: 'Board rank, after manual overrides.',
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
  {
    key: 'adj_vor',
    label: 'VOR',
    title:
      'Value over replacement: points per game above the freely-available player at this position. What the board sorts by.',
    render: (p) => <span className="strong">{p.adj_vor.toFixed(1)}</span>,
  },
  {
    key: 'ppg',
    label: 'PPG',
    title: 'League-scored points per game, including big-play bonuses.',
    render: (p) => number(p.ppg),
  },
  {
    key: 'flags',
    label: 'Flags',
    title: 'Model opinions. Hover any chip for what it means.',
    align: 'left',
    initial: 'asc',
    render: (p) => <Flags value={p.flags} />,
  },
  {
    key: 'games',
    label: 'G',
    title: 'Games played. Small samples carry an Ngms flag.',
    render: (p) => <span className="dim">{p.games}</span>,
  },
  {
    key: 'floor',
    label: 'Floor',
    title: '25th-percentile weekly score. What decides who you start.',
    render: (p) => <span className="dim">{number(p.floor)}</span>,
  },
  {
    key: 'volatility',
    label: 'Vol',
    title:
      'Standard deviation of weekly scores. Buy it deliberately when you need a ceiling, avoid it when you are favoured.',
    render: (p) => <span className="dim">{number(p.volatility)}</span>,
  },
  {
    key: 'wtd_opp',
    label: 'Opp',
    title:
      'Weighted opportunity: carries + 2.2 x targets. Volume is the sticky, predictive stat; efficiency regresses.',
    render: (p) => <span className="dim">{number(p.wtd_opp, 0)}</span>,
  },
  {
    key: 'td_over_exp',
    label: 'TDOE',
    title:
      'Touchdowns above or below what the volume implies. Positive is luck that will not repeat; negative with volume is a buy.',
    render: (p) => (
      <span style={{ color: p.td_over_exp > 2 ? 'var(--sell)' : p.td_over_exp < -2 ? 'var(--buy)' : undefined }}>
        {p.td_over_exp > 0 ? '+' : ''}
        {p.td_over_exp.toFixed(1)}
      </span>
    ),
  },
  {
    key: 'bonus_pts',
    label: 'Bonus',
    title:
      "Big-play bonus points from 40+ and 50+ yard touchdowns. This league's private edge, kept visible.",
    render: (p) => (
      <span className={p.bonus_pts > 0 ? 'dim' : 'faint'}>{p.bonus_pts.toFixed(0)}</span>
    ),
  },
  {
    key: 'target_share',
    label: 'Tgt%',
    title: 'Share of the team’s targets, weighted by weekly usage.',
    render: (p) => <span className="dim">{percent(p.target_share)}</span>,
  },
  {
    key: 'air_yards_share',
    label: 'AY%',
    title:
      'Share of the team’s air yards. Matters extra here: deep targets feed the bonus brackets.',
    render: (p) => <span className="dim">{percent(p.air_yards_share)}</span>,
  },
  {
    key: 'age_at_season',
    label: 'Age',
    title: 'Age as of Sept 1 of the season being played.',
    render: (p) => <span className="dim">{number(p.age_at_season)}</span>,
  },
]

export function BoardTable({ players }: { players: Player[] }) {
  const [sortKey, setSortKey] = useState<SortKey>('rank')
  const [direction, setDirection] = useState<Direction>('asc')

  const sorted = useMemo(() => {
    const rows = [...players]
    rows.sort((a, b) => {
      const left = a[sortKey]
      const right = b[sortKey]

      // Nulls always sort last, whichever direction is active — an unknown value is
      // not a small one, and letting it float to the top would be misleading.
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

  const onSort = (column: Column) => {
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
            {COLUMNS.map((column) => (
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
            <tr key={player.player_id}>
              {COLUMNS.map((column) => (
                <td
                  key={String(column.key)}
                  className={column.align === 'left' ? 'left' : undefined}
                >
                  {column.render ? column.render(player) : String(player[column.key])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
