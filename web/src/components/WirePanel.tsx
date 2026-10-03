import { forecastColumns } from './ForecastColumns'
import type { LeaguePlayer, MetricVersion } from '../api/types'
import { useQuery } from '@tanstack/react-query'
import { SEASON_EQUIVALENT_TITLE } from '../metricPresentation'
import { Freshness } from './Freshness'
import { useUrlFlag, useUrlState } from '../navigation'
import { SKILL_POSITIONS } from '../positions'
import { unrankableQuery, wireQuery } from '../api/queries/archive'
import { PlayerTable } from './PlayerTable'
import { FLAGS_COLUMN, IDENTITY, ROSTERED_PERCENT, type PlayerColumn } from './playerColumns'
import { fixed } from '../format'

const COLUMNS: PlayerColumn[] = [
  ...IDENTITY,
  {
    key: 'wire_vor',
    label: 'Wire VOR',
    title:
      'Historical PPG above the best freely available player in the current free-agent pool. This is not preseason draft VOR.',
    render: (p) => <span className="strong">{fixed(p.wire_vor, 2)}</span>,
  },
  {
    key: 'ppg',
    label: 'PPG',
    title: 'League-scored points per game from last season.',
    render: (p) => fixed(p.ppg),
  },
  FLAGS_COLUMN,
  {
    key: 'floor',
    label: 'Floor',
    title: 'Historical 25th-percentile weekly score; descriptive, not a guaranteed floor.',
    render: (p) => <span className="dim">{fixed(p.floor)}</span>,
  },
  ROSTERED_PERCENT,
]

/**
 * The ranked wire.
 *
 * The one screen that wins waivers. Its numbers deliberately differ from the draft
 * board's: replacement is the best player at each position among the
 * players currently unowned. Using the preseason pool would misstate pickup value.
 */
export function WirePanel({ version }: { version: MetricVersion }) {
  const [healthyOnly, setHealthyOnly] = useUrlFlag('healthy')
  const [positionParam, setPosition] = useUrlState('position', '')
  const position = positionParam || null

  const wire = useQuery(wireQuery(version, healthyOnly))
  const unrankable = useQuery(unrankableQuery(version))

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
        {SKILL_POSITIONS.map((pos) => (
          <button
            key={pos}
            type="button"
            className="chip"
            aria-pressed={position === pos}
            onClick={() => setPosition(position === pos ? '' : pos)}
          >
            {pos}
          </button>
        ))}
        <span style={{ width: 8 }} />
        <button
          type="button"
          className="chip"
          aria-pressed={healthyOnly}
          onClick={() => setHealthyOnly(!healthyOnly)}
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
        — the best freely available player at each position. Wire VOR is zero for that
        player and negative for weaker alternatives. Lineup gain measures improvement to your starters. {version !== 'v1' &&
          'Both the pool baseline and Wire VOR use availability-adjusted season-equivalent PPG.'}
      </p>

      <PlayerTable
        version={version}
        players={players}
        columns={version === 'v1' ? COLUMNS : [
          ...IDENTITY,
          { key: 'wire_vor', label: 'Wire VOR', title: 'Season-equivalent PPG above the best free agent at this position.', render: p => <b>{fixed(p.wire_vor, 2)}</b> },
          { key: 'lineup_improvement', label: 'Lineup gain', title: 'Preseason season-equivalent improvement to your best legal lineup before choosing a drop.', render: p => fixed(p.lineup_improvement) },
          { key: 'season_equivalent_ppg', label: 'Season PPG', title: SEASON_EQUIVALENT_TITLE, render: p => fixed(p.season_equivalent_ppg) },
          ...forecastColumns<LeaguePlayer>(version).filter(column => !['v2_overall_vor', 'v2_position_rank'].includes(column.key)),
        ]}
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
