import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { fetchUnrankable, fetchWire } from '../api/client'
import type { MetricVersion } from '../api/types'
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

const COLUMNS: PlayerColumn[] = [
  ...IDENTITY,
  {
    key: 'wire_vor',
    label: 'Wire VOR',
    title:
      'Points per game above the best player still available at this position. Not the draft-board VOR — replacement is measured against who is actually free right now.',
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
    title: '25th-percentile weekly score — what you can count on if you start him.',
    render: (p) => <span className="dim">{number(p.floor)}</span>,
  },
  ROSTERED_PERCENT,
  OWNERSHIP,
]

const V2_COLUMNS: PlayerColumn[] = [
  ...IDENTITY,
  {
    key: 'wire_vor',
    label: 'Wire VOR',
    title: 'Availability-adjusted projected value above the actual free-agent pool.',
    render: (p) => <span className="strong">{number(p.wire_vor, 2)}</span>,
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
  FLAGS_COLUMN,
  ROSTERED_PERCENT,
  OWNERSHIP,
]

/**
 * The ranked wire.
 *
 * The one screen that wins waivers. Its numbers deliberately differ from the draft
 * board's: replacement level here is the best player still unowned, which after a
 * draft sits far below the preseason baseline. Using the preseason number would
 * understate every pickup on the list.
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
          title="Hide players who cannot be started this week. Off by default — a cheap stash is a real play."
        >
          Startable only
        </button>
        <span className="spacer" />
        <span className="count">{players.length} available</span>
      </div>

      <p className="legend tight">
        <b>Replacement right now:</b>{' '}
        {Object.entries(wire.data.replacement_levels)
          .sort()
          .map(([pos, ppg]) => `${pos} ${ppg.toFixed(1)}`)
          .join(' · ')}{' '}
        — the best player still unowned at each position. Wire VOR is measured against
        this, not against the preseason draft baseline. {version === 'v2' &&
          'V2 uses season-equivalent projected PPG, so expected availability affects both the pool baseline and the pickup value.'}
      </p>

      <PlayerTable
        players={players}
        columns={version === 'v2' ? V2_COLUMNS : COLUMNS}
        defaultSort="wire_vor"
        emptyMessage="Nobody on the wire clears replacement level at this position."
      />

      {unrankable.data && unrankable.data.total > 0 && (
        <div className="unrankable">
          <h3>{unrankable.data.total} players the model can’t price</h3>
          <p className="faint">
            2026 rookies and players with no prior-season tape. They are listed rather than
            hidden — a wire that silently omits a hyped rookie is worse than one that admits
            it cannot rank him. Sorted by how much of the market already believes.
          </p>
          <div className="unrankable-grid">
            {unrankable.data.players.slice(0, 12).map((p) => (
              <div key={`${p.player_display_name}-${p.position}`} className="unrankable-card">
                <span className="name">{p.player_display_name}</span>
                <span className={`pos ${p.position}`}>{p.position}</span>
                <span className="faint">{p.espn_team ?? '—'}</span>
                <span className="dim">{(p.percent_owned ?? 0).toFixed(0)}% rostered</span>
              </div>
            ))}
          </div>
        </div>
      )}
      <Freshness data={wire.data} />
    </section>
  )
}
