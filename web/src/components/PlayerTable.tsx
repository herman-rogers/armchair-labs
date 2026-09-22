import type { LeaguePlayer } from '../api/types'
import { DataTable, type Column } from './DataTable'
import { Flags } from './Flags'
import { InjuryBadge, MovedBadge, OwnerBadge } from './Availability'

export type PlayerColumn = Column<LeaguePlayer>

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
    title: 'Combined display rank. Patron model order is preserved; flagged ESPN-only players enter at ESPN’s current PPR draft rank.',
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

export const ESPN_RANK: PlayerColumn = {
  key: 'espn_draft_rank',
  label: 'ESPN',
  title: 'Current ESPN PPR draft-room rank among skill players. This is a market ordering, not ESPN projected points or Patron’s model output.',
  initial: 'asc',
  render: (p) => <span className="dim">{p.espn_draft_rank ?? '—'}</span>,
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
 * League-aware player rows on the common DataTable. The columns differ per screen
 * (a wire needs value-against-the-pool, a roster needs injury and bye) and every
 * row carries ownership.
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
  return (
    <DataTable
      rows={players}
      columns={columns}
      defaultSort={defaultSort}
      rowKey={(player) => player.player_id}
      rowClass={(player) => (player.is_mine ? 'mine-row' : undefined)}
      emptyMessage={emptyMessage}
    />
  )
}
