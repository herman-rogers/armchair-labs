/** Column definitions shared by the archived league player tables. */
import type { LeaguePlayer } from '../api/types'
import type { Column } from './DataTable'
import { Flags } from './Flags'
import { InjuryBadge, MovedBadge, OwnerBadge } from './Availability'
import { fixed } from '../format'

export type PlayerColumn = Column<LeaguePlayer>

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
  label: 'Roster',
  title: 'Who owns this player in your league right now.',
  align: 'left',
  initial: 'asc',
  render: (p) => <OwnerBadge player={p} />,
}

export const ROSTERED_PERCENT: PlayerColumn = {
  key: 'percent_owned',
  label: '%Rost',
  title: 'Share of ESPN leagues rostering this player — the market’s opinion, not yours.',
  render: (p) => <span className="dim">{fixed(p.percent_owned, 0)}</span>,
}

export const FLAGS_COLUMN: PlayerColumn = {
  key: 'flags',
  label: 'Flags',
  title: 'Historical heuristics and sample warnings. Hover any chip for its limits.',
  align: 'left',
  initial: 'asc',
  render: (p) => <Flags value={p.flags} />,
}
