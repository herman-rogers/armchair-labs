import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { fetchUnrankable, fetchWire } from '../api/client'
import type { MetricVersion } from '../api/types'
import { SEASON_EQUIVALENT_TITLE } from '../metricPresentation'
import { Freshness } from './Freshness'
import {
  FLAGS_COLUMN,
  IDENTITY,
  PlayerTable,
  ROSTERED_PERCENT,
  number,
  type PlayerColumn,
} from './PlayerTable'

const COLUMNS: PlayerColumn[] = [
  ...IDENTITY,
  {
    key: 'wire_vor',
    label: 'Wire VOR',
    title:
      'Historical PPG above the configured replacement-ranked player in the current free-agent pool. This is not preseason draft VOR.',
    render: (p) => <span className="strong">{number(p.wire_vor, 2)}</span>,
  },
  {
    key: 'ppg',
    label: 'PPG',
    title: 'League-scored points per game from last season.',
    render: (p) => number(p.ppg),
  },
  FLAGS_COLUMN,
  {
    key: 'floor',
    label: 'Floor',
    title: 'Historical 25th-percentile weekly score; descriptive, not a guaranteed floor.',
    render: (p) => <span className="dim">{number(p.floor)}</span>,
  },
  ROSTERED_PERCENT,
]

const V2_COLUMNS: PlayerColumn[] = [
  ...IDENTITY,
  {
    key: 'wire_vor',
    label: 'Wire VOR',
    title: 'Season-equivalent projected PPG above the configured replacement-ranked player in the current free-agent pool.',
    render: (p) => <span className="strong">{number(p.wire_vor, 2)}</span>,
  },
  {
    key: 'season_equivalent_ppg',
    label: 'Avail PPG',
    title: SEASON_EQUIVALENT_TITLE,
    render: (p) => number(p.season_equivalent_ppg),
  },
  {
    key: 'proj_ppg',
    label: 'Active PPG',
    title: 'Hand-built projected points per active game. Compare with Avail PPG to see the effect of expected missed time.',
    render: (p) => <span className="dim">{number(p.proj_ppg)}</span>,
  },
  {
    key: 'expected_games',
    label: 'Exp G',
    title: 'Expected games from nflverse participation and injury history.',
    render: (p) => <span className="dim">{number(p.expected_games)}</span>,
  },
  {
    key: 'projected_targets_pg',
    label: 'Tgt/G',
    title: 'Projected targets per game.',
    render: (p) => <span className="dim">{number(p.projected_targets_pg)}</span>,
  },
  {
    key: 'projected_carries_pg',
    label: 'Car/G',
    title: 'Projected carries per game.',
    render: (p) => <span className="dim">{number(p.projected_carries_pg)}</span>,
  },
  {
    key: 'ppg',
    label: 'Actual PPG',
    title: 'Most recent season’s PPG, shown as baseline evidence.',
    render: (p) => <span className="dim">{number(p.ppg)}</span>,
  },
  ROSTERED_PERCENT,
]

/**
 * The ranked wire.
 *
 * The one screen that wins waivers. Its numbers deliberately differ from the draft
 * board's: replacement is recalculated at the configured positional slot within the
 * players currently unowned. Using the preseason pool would misstate pickup value.
 */
export function WirePanel({ version }: { version: MetricVersion }) {
  const [healthyOnly, setHealthyOnly] = useState(false)
  const [position, setPosition] = useState<string | null>(null)

  const wire = useQuery({
    queryKey: ['wire', version, healthyOnly],
    queryFn: () => fetchWire(version, healthyOnly),
  })
  const unrankable = useQuery({
    queryKey: ['unrankable', version],
    queryFn: () => fetchUnrankable(version),
  })

  if (wire.isError) {
    return <div className="notice">{(wire.error as Error).message}</div>
  }
  if (!wire.data) return <div className="notice">Loading the wire…</div>

  const players = position
    ? wire.data.players.filter((p) => p.position === position)
    : wire.data.players

  return (
    <section>
      <div className="controls">
        {['QB', 'RB', 'WR', 'TE'].map((pos) => (
          <button
            key={pos}
            type="button"
            className="chip"
            aria-pressed={position === pos}
            onClick={() => setPosition((current) => (current === pos ? null : pos))}
          >
            {pos}
          </button>
        ))}
        <span style={{ width: 8 }} />
        <button
          type="button"
          className="chip"
          aria-pressed={healthyOnly}
          onClick={() => setHealthyOnly((v) => !v)}
          title="Hide ESPN OUT, IR, and suspension tags. This does not account for byes. Off by default because a stash can still be useful."
        >
          Hide OUT/IR/SUSP
        </button>
        <span className="spacer" />
        <span className="count">{players.length} ranked free agents</span>
      </div>

      <p className="legend tight">
        <b>Free-agent replacement:</b>{' '}
        {Object.entries(wire.data.replacement_levels)
          .sort()
          .map(([pos, ppg]) => `${pos} ${ppg.toFixed(1)}`)
          .join(' · ')}{' '}
        — the configured replacement slot within the players currently unowned, not the
        best free agent and not the preseason pool. {version !== 'v1' &&
          'Both the pool baseline and Wire VOR use availability-adjusted season-equivalent PPG.'}
      </p>

      <PlayerTable
        players={players}
        columns={version !== 'v1' ? V2_COLUMNS : COLUMNS}
        defaultSort="wire_vor"
        emptyMessage="No ranked free agents at this position."
      />

      {unrankable.data && unrankable.data.total > 0 && (
        <div className="unrankable">
          <h3>{unrankable.data.total} players the model can’t price</h3>
          <p className="faint">
            Rookies and players with no usable prior tape. They are listed rather than
            assigned a fake zero, sorted by ESPN roster percentage. This list includes
            rostered players as well as free agents.
          </p>
          <div className="unrankable-grid">
            {unrankable.data.players.slice(0, 12).map((p) => (
              <div key={`${p.player_display_name}-${p.position}`} className="unrankable-card">
                <span className="name">{p.player_display_name}</span>
                <span className={`pos ${p.position}`}>{p.position}</span>
                <span className="faint">{p.espn_team ?? '—'}</span>
                <span className="dim">{(p.percent_owned ?? 0).toFixed(0)}% rostered</span>
                <span className="faint owner">{p.owner_team_name ?? 'Unowned or unlisted'}</span>
              </div>
            ))}
          </div>
        </div>
      )}
      <Freshness data={wire.data} />
    </section>
  )
}
