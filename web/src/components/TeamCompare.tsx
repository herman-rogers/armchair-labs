import { useState } from 'react'
import type { LineupSlot, MetricVersion, TeamStrength } from '../api/types'
import { boardMetricLabel, SYSTEM_LABELS } from '../metricPresentation'

function Side({ team, opponent }: { team: TeamStrength; opponent: TeamStrength }) {
  const ahead = team.total > opponent.total
  return (
    <div className={`compare-side ${ahead ? 'ahead' : ''}`}>
      <div className="compare-head">
        <span className="compare-name">{team.team_name}</span>
        <span className={`compare-total ${ahead ? 'ahead' : ''}`}>{team.total.toFixed(1)}</span>
      </div>
      {team.projected_total != null && (
        <p className="faint tight" title="What ESPN projected for this lineup before the week.">
          projected {team.projected_total.toFixed(1)}
        </p>
      )}
      {team.source === 'projected' && team.unranked_starters > 0 && (
        <p
          className="faint tight"
          title="Players with no prior-season tape score nothing here, so this team's total is understated."
        >
          {team.unranked_starters} unrankable on roster — total understated
        </p>
      )}
    </div>
  )
}

/**
 * Reasons a total understates a team, shown even when the heads are hidden.
 *
 * A roster leaning on rookies scores nothing for them, so a comparison that stays
 * silent about it is quietly wrong in a way the reader cannot see. A played week has
 * no such gap — every player in it scored what they scored — so this says nothing
 * there rather than inventing a caveat.
 */
function Caveats({ left, right }: { left: TeamStrength; right: TeamStrength }) {
  if (left.source === 'actual') return null
  const noted = [left, right].filter((team) => team.unranked_starters > 0)
  if (!noted.length) return null

  return (
    <p className="faint tight compare-caveats">
      {noted
        .map((team) => `${team.team_name}: ${team.unranked_starters} unrankable`)
        .join(' · ')}{' '}
      — players with no prior-season tape score nothing here, so those totals are
      understated.
    </p>
  )
}

/** One slot on both teams, with the better side highlighted. */
function SlotRow({ slot, left, right }: { slot: string; left?: LineupSlot; right?: LineupSlot }) {
  const leftValue = left?.value ?? 0
  const rightValue = right?.value ?? 0
  const edge = leftValue - rightValue
  // Judged on the displayed precision. A gap of 0.04 is a tie to anyone reading the
  // screen, and pointing an arrow at it claims a distinction the numbers do not show.
  const shown = Math.abs(edge) < 0.05 ? 0 : edge

  return (
    <div className="slot-row">
      <div className={`slot-player left ${shown > 0 ? 'winning' : ''}`}>
        <span className="slot-name">{left?.player_display_name ?? '—'}</span>
        <span className="slot-value">{left ? leftValue.toFixed(1) : '—'}</span>
      </div>
      <div className="slot-label">
        <span className={`pos ${left?.position ?? right?.position ?? ''}`}>{slot}</span>
        {/* An arrow toward the stronger side rather than a signed, coloured number.
            Red and green read as bad and good, which is wrong here: whether a negative
            edge is good news depends entirely on which side you are. */}
        <span className="slot-edge">
          {shown === 0 ? (
            'even'
          ) : (
            <>
              {shown > 0 && <span className="edge-arrow">◀</span>}
              {Math.abs(shown).toFixed(1)}
              {shown < 0 && <span className="edge-arrow">▶</span>}
            </>
          )}
        </span>
      </div>
      <div className={`slot-player right ${shown < 0 ? 'winning' : ''}`}>
        <span className="slot-value">{right ? rightValue.toFixed(1) : '—'}</span>
        <span className="slot-name">{right?.player_display_name ?? '—'}</span>
      </div>
    </div>
  )
}

/**
 * Bench players who outscored a starter at their own position.
 *
 * The only question worth asking of a finished bench. Compared within position rather
 * than against the whole lineup: a kicker outscoring a wide receiver is not a decision
 * anyone could have made, and flagging it would train the reader to ignore the mark.
 */
function missedStarts(team: TeamStrength): Set<string> {
  if (team.source !== 'actual') return new Set()

  const worstStarter = new Map<string, number>()
  for (const starter of team.starters) {
    if (!starter.position) continue
    const current = worstStarter.get(starter.position)
    if (current === undefined || starter.value < current) {
      worstStarter.set(starter.position, starter.value)
    }
  }

  const missed = new Set<string>()
  for (const player of team.bench) {
    if (!player.position || !player.player_display_name) continue
    const floor = worstStarter.get(player.position)
    if (floor !== undefined && player.value > floor) missed.add(player.player_display_name)
  }
  return missed
}

function BenchColumn({ team, align }: { team: TeamStrength; align: 'left' | 'right' }) {
  const missed = missedStarts(team)
  if (!team.bench.length) return <div className={`bench-column ${align}`} />

  return (
    <div className={`bench-column ${align}`}>
      {team.bench.map((player) => {
        const left = (
          <span className="slot-name">
            {player.player_display_name ?? '—'}
            {missed.has(player.player_display_name ?? '') && (
              <span
                className="bench-missed"
                title={`Outscored your worst starting ${player.position} this week.`}
              >
                ▲
              </span>
            )}
          </span>
        )
        const value = (
          <span className="slot-value">
            {player.value.toFixed(1)}
            {player.position && <span className="faint bench-pos"> {player.position}</span>}
          </span>
        )
        return (
          <div className="bench-row" key={`${player.espn_id ?? player.player_id}-${player.player_display_name}`}>
            {align === 'left' ? (
              <>
                {left}
                {value}
              </>
            ) : (
              <>
                {value}
                {left}
              </>
            )}
          </div>
        )
      })}
    </div>
  )
}

/** What the numbers in this comparison actually are. */
function Legend({ team, context }: { team: TeamStrength; context: 'rosters' | 'matchup' }) {
  if (team.source === 'actual') {
    return (
      <p className="legend tight faint">
        The lineup each manager actually set that week, with the points it actually
        scored — not the best lineup available. ▲ marks a bench player who outscored
        the worst starter at his position.
      </p>
    )
  }
  return (
    <p className="legend tight faint">
      {context === 'matchup' ? 'This week has not been played. ' : ''}Best fieldable lineups by {boardMetricLabel(team.metric)} — a season-long strength
      comparison, not projected weekly points: it does not know byes, the injury
      report, or opponent. Kicker and defense slots are excluded because this league
      tiers those rather than ranking them.
    </p>
  )
}

/**
 * Two teams, slot by slot, then bench by bench.
 *
 * For a played week this is ESPN's record: the starters that counted and the bench
 * that did not. For a week still to come there is no such record, so it falls back to
 * comparing the lineups each team could field — five good running backs are worth two
 * starters and a flex, and a comparison that ignores that ranks hoarders above
 * balanced teams.
 */
export function TeamCompare({
  left,
  right,
  margin,
  showHeads = true,
  version,
  context = 'matchup',
}: {
  left: TeamStrength
  right: TeamStrength
  margin: number
  /** Off when the caller already shows both names and totals directly above. */
  showHeads?: boolean
  version?: MetricVersion
  context?: 'rosters' | 'matchup'
}) {
  const hasBench = left.bench.length > 0 || right.bench.length > 0
  // Open by default on a played week, where the bench is half the point. Closed on a
  // projected one, where it is leftovers from a lineup nobody set.
  const [benchOpen, setBenchOpen] = useState(left.source === 'actual')

  // Slots appear in the order the lineup was built, which is dedicated positions
  // before flex — the order a manager actually fills them.
  const slots: string[] = []
  for (const slot of [...left.starters, ...right.starters]) {
    if (!slots.includes(slot.slot)) slots.push(slot.slot)
  }

  const takeFrom = (team: TeamStrength, slot: string, index: number) =>
    team.starters.filter((s) => s.slot === slot)[index]

  return (
    <div className="compare">
      {left.source !== 'actual' && version && <p className="comparison-basis"><b>{SYSTEM_LABELS[version]}</b> · lineup value above replacement. These values are not projected weekly scores.</p>}
      {showHeads ? (
        <div className="compare-heads">
          <Side team={left} opponent={right} />
          <div className="compare-margin">
            <span className="faint">edge</span>
            <span>
              {margin === 0 ? 'even' : `${Math.abs(margin).toFixed(1)}`}
            </span>
          </div>
          <Side team={right} opponent={left} />
        </div>
      ) : (
        <Caveats left={left} right={right} />
      )}

      <div className="slot-rows">
        {slots.flatMap((slot) => {
          const count = Math.max(
            left.starters.filter((s) => s.slot === slot).length,
            right.starters.filter((s) => s.slot === slot).length,
          )
          return Array.from({ length: count }, (_, index) => (
            <SlotRow
              key={`${slot}-${index}`}
              slot={slot}
              left={takeFrom(left, slot, index)}
              right={takeFrom(right, slot, index)}
            />
          ))
        })}
      </div>

      {hasBench && (
        <div className={`bench ${benchOpen ? 'open' : ''}`}>
          <button type="button" className="bench-toggle" onClick={() => setBenchOpen(!benchOpen)}>
            <span className="faint">{benchOpen ? '−' : '+'}</span>
            <span>Bench</span>
            <span className="faint">
              {left.bench.length} · {right.bench.length}
            </span>
          </button>
          {benchOpen && (
            <div className="bench-rows">
              <BenchColumn team={left} align="left" />
              <BenchColumn team={right} align="right" />
            </div>
          )}
        </div>
      )}

      <Legend team={left} context={context} />
    </div>
  )
}
