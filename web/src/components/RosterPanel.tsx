import { useQuery } from '@tanstack/react-query'
import { fetchRoster } from '../api/client'
import type { LeagueTeam, MetricVersion } from '../api/types'
import {
  FLAGS_COLUMN,
  IDENTITY,
  PlayerTable,
  number,
  type PlayerColumn,
} from './PlayerTable'

const COLUMNS: PlayerColumn[] = [
  ...IDENTITY,
  {
    key: 'adj_vor',
    label: 'VOR',
    title: 'Points per game above the replacement-level player at this position.',
    render: (p) => <span className="strong">{number(p.adj_vor, 2)}</span>,
  },
  {
    key: 'ppg',
    label: 'PPG',
    title: 'League-scored points per game.',
    render: (p) => number(p.ppg),
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
    title: 'Weekly standard deviation. Buy it when you need a ceiling, avoid it when favoured.',
    render: (p) => <span className="dim">{number(p.volatility)}</span>,
  },
  FLAGS_COLUMN,
  {
    key: 'percent_started',
    label: '%Start',
    title: 'Share of ESPN leagues starting this player this week — the market’s read.',
    render: (p) => <span className="dim">{number(p.percent_started, 0)}</span>,
  },
]

const V2_COLUMNS: PlayerColumn[] = [
  ...IDENTITY,
  {
    key: 'v2_score',
    label: 'V2 Score',
    title: 'Availability-aware projected value over replacement.',
    render: (p) => <span className="strong">{number(p.v2_score, 2)}</span>,
  },
  {
    key: 'proj_ppg',
    label: 'V2 PPG',
    title: 'Combined points per active game: normalized actual history plus the bottom-up forecast.',
    render: (p) => number(p.proj_ppg),
  },
  {
    key: 'expected_games',
    label: 'Exp G',
    title: 'Expected games from nflverse participation and injury history.',
    render: (p) => <span className="dim">{number(p.expected_games)}</span>,
  },
  {
    key: 'projected_floor',
    label: 'Floor',
    title: 'Projected 25th-percentile weekly score when active.',
    render: (p) => <span className="dim">{number(p.projected_floor)}</span>,
  },
  FLAGS_COLUMN,
  {
    key: 'percent_started',
    label: '%Start',
    title: 'ESPN market usage; displayed but not used by V2.',
    render: (p) => <span className="dim">{number(p.percent_started, 0)}</span>,
  },
]

/** One team's roster, joined to the board, decisions first. */
export function RosterPanel({
  team,
  version,
}: {
  team: LeagueTeam
  version: MetricVersion
}) {
  const roster = useQuery({
    queryKey: ['roster', team.team_id, version],
    queryFn: () => fetchRoster(team.team_id, version),
  })

  if (roster.isError) return <div className="notice">{(roster.error as Error).message}</div>
  if (!roster.data) return <div className="notice">Loading roster…</div>

  const flagged = roster.data.players.filter(
    (p) => p.injury_status && !['ACTIVE', 'NORMAL'].includes(p.injury_status),
  )
  const moved = roster.data.players.filter((p) => p.changed_team)

  return (
    <section>
      {(flagged.length > 0 || moved.length > 0) && (
        <p className="legend tight">
          {flagged.length > 0 && (
            <>
              <b>{flagged.length} carrying an injury tag.</b>{' '}
            </>
          )}
          {moved.length > 0 && (
            <>
              <b>{moved.length} changed teams</b> since the tape these metrics come from — their
              target share and air-yards share describe a situation that no longer exists.
            </>
          )}
        </p>
      )}
      <PlayerTable
        players={roster.data.players}
        columns={version === 'v2' ? V2_COLUMNS : COLUMNS}
        defaultSort={version === 'v2' ? 'v2_score' : 'adj_vor'}
        emptyMessage="No ranked players on this roster."
      />
      <p className="legend tight faint">
        Only players the board can rank appear here. Rookies and anyone without prior-season
        tape are on the roster but unrankable — see the wire tab for that list.
      </p>
    </section>
  )
}
