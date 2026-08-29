import { useMemo, useState } from 'react'
import type { MetricVersion, Player } from '../api/types'
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

const number = (value: number | null | undefined, digits = 1) =>
  value === null || value === undefined ? <span className="faint">—</span> : value.toFixed(digits)

const percent = (value: number | null | undefined) =>
  value === null || value === undefined ? (
    <span className="faint">—</span>
  ) : (
    `${(value * 100).toFixed(0)}%`
  )

const IDENTITY_COLUMNS: Column[] = [
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

const V1_COLUMNS: Column[] = [
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

const V2_COLUMNS: Column[] = [
  {
    key: 'projected_team',
    label: 'Proj Tm',
    title: 'Team whose forward offensive, quarterback, and teammate context is applied.',
    align: 'left',
    initial: 'asc',
    render: (p) => <span className="team">{p.projected_team ?? p.team}</span>,
  },
  {
    key: 'v2_score',
    label: 'V2 Score',
    title: 'Balanced draft score: 75% expected VOR, 15% floor VOR, and 10% ceiling VOR, plus manual overrides. This sorts v2.',
    render: (p) => <span className="strong">{number(p.v2_score)}</span>,
  },
  {
    key: 'adj_proj_vor',
    label: 'Proj VOR',
    title: 'Expected projected points per game above projected positional replacement, including manual overrides.',
    render: (p) => <span className="dim">{number(p.adj_proj_vor)}</span>,
  },
  {
    key: 'projected_floor',
    label: 'Floor',
    title: 'Projected 25th-percentile score using the shrunk historical floor profile.',
    render: (p) => <span className="dim">{number(p.projected_floor)}</span>,
  },
  {
    key: 'projected_ceiling',
    label: 'Ceil',
    title: 'Projected 75th-percentile score from the shrunk weekly volatility profile.',
    render: (p) => <span className="dim">{number(p.projected_ceiling)}</span>,
  },
  {
    key: 'proj_ppg',
    label: 'Proj PPG',
    title: 'Forward PPG after individual shrinkage, regression, age, role, and team context.',
    render: (p) => number(p.proj_ppg),
  },
  {
    key: 'ppg',
    label: 'Last PPG',
    title: 'Most recent season’s actual league-scored PPG; evidence, not the v2 sort key.',
    render: (p) => <span className="dim">{number(p.ppg)}</span>,
  },
  {
    key: 'projection_confidence',
    label: 'Conf',
    title: 'Sample support for the projection. Team moves and manual role assumptions reduce it.',
    render: (p) => <span className="dim">{percent(p.projection_confidence)}</span>,
  },
  {
    key: 'team_context_factor',
    label: 'Team x',
    title: 'Projected-team environment relative to the contexts already embedded in the player history.',
    render: (p) => <span className="dim">{number(p.team_context_factor, 2)}x</span>,
  },
  {
    key: 'qb_context',
    label: 'QB Ctx',
    title: 'Team passing fantasy environment per game: pass yards and TDs, less interceptions.',
    render: (p) => <span className="dim">{number(p.qb_context)}</span>,
  },
  {
    key: 'teammate_competition',
    label: 'Comp',
    title: 'Largest teammate’s share of target opportunity (or RB backfield opportunity). Lower means more available role.',
    render: (p) => <span className="dim">{percent(p.teammate_competition)}</span>,
  },
  {
    key: 'projected_targets_pg',
    label: 'Tgt/G',
    title: 'Projected targets per game from target share, WOPR, raw weighted opportunity, team pass volume, and competition.',
    render: (p) => <span className="dim">{number(p.projected_targets_pg)}</span>,
  },
  {
    key: 'projected_carries_pg',
    label: 'Car/G',
    title: 'Projected carries per game from carry share, team rushing volume, and backfield competition.',
    render: (p) => <span className="dim">{number(p.projected_carries_pg)}</span>,
  },
  {
    key: 'projected_target_share',
    label: 'Tgt%',
    title: 'Projected target share, shrunk and blended with WOPR before team competition.',
    render: (p) => <span className="dim">{percent(p.projected_target_share)}</span>,
  },
  {
    key: 'projected_air_yards_share',
    label: 'AY%',
    title: 'Projected air-yards share; used to adjust yards per target and long-touchdown bonus expectation.',
    render: (p) => <span className="dim">{percent(p.projected_air_yards_share)}</span>,
  },
  {
    key: 'projected_wopr',
    label: 'WOPR',
    title: 'Projected WOPR role signal. It contributes 25% of projected target share rather than being added again at full weight.',
    render: (p) => <span className="dim">{number(p.projected_wopr, 2)}</span>,
  },
  {
    key: 'projected_bonus_pg',
    label: 'Bonus/G',
    title: 'Projected long-touchdown bonus points per game, regressed by opportunity and adjusted for air-yard role.',
    render: (p) => <span className="dim">{number(p.projected_bonus_pg, 2)}</span>,
  },
  {
    key: 'flags',
    label: 'Flags',
    title: 'Historical model signals retained as supporting evidence.',
    align: 'left',
    initial: 'asc',
    render: (p) => <Flags value={p.flags} />,
  },
  {
    key: 'age_at_season',
    label: 'Age',
    title: 'Age as of Sept 1 of the projected season; v2 applies a gradual position-specific curve.',
    render: (p) => <span className="dim">{number(p.age_at_season)}</span>,
  },
]

export function BoardTable({ players, version }: { players: Player[]; version: MetricVersion }) {
  const [sortKey, setSortKey] = useState<SortKey>('rank')
  const [direction, setDirection] = useState<Direction>('asc')
  const columns = version === 'v2' ? [...IDENTITY_COLUMNS.slice(0, 3), ...V2_COLUMNS] : [...IDENTITY_COLUMNS, ...V1_COLUMNS]

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
            <tr key={player.player_id}>
              {columns.map((column) => (
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
