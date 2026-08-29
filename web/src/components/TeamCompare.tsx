import type { LineupSlot, TeamStrength } from '../api/types'

function Side({ team, opponent }: { team: TeamStrength; opponent: TeamStrength }) {
  const ahead = team.total > opponent.total
  return (
    <div className={`compare-side ${ahead ? 'ahead' : ''}`}>
      <div className="compare-head">
        <span className="compare-name">{team.team_name}</span>
        <span className={`compare-total ${ahead ? 'ahead' : ''}`}>{team.total.toFixed(1)}</span>
      </div>
      {team.unranked_starters > 0 && (
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
 * silent about it is quietly wrong in a way the reader cannot see.
 */
function Caveats({ left, right }: { left: TeamStrength; right: TeamStrength }) {
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
 * Two teams, slot by slot.
 *
 * Compares the lineups each team can actually field rather than total roster value —
 * five good running backs are worth two starters and a flex, and a comparison that
 * ignores that ranks hoarders above balanced teams.
 */
export function TeamCompare({
  left,
  right,
  margin,
  showHeads = true,
}: {
  left: TeamStrength
  right: TeamStrength
  margin: number
  /** Off when the caller already shows both names and totals directly above. */
  showHeads?: boolean
}) {
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

      <p className="legend tight faint">
        Best fieldable lineup by <code>{left.metric}</code>, not a weekly projection — it
        does not know byes, this week’s injury report, or opponent. Kicker and defense
        slots are excluded because this league tiers those rather than ranking them.
      </p>
    </div>
  )
}
