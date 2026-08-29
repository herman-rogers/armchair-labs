import type { Availability as AvailabilityValue, LeaguePlayer } from '../api/types'

const LABELS: Record<AvailabilityValue, string> = {
  rostered: 'Rostered',
  free_agent: 'Free',
  unknown: 'Unlisted',
}

const TITLES: Record<AvailabilityValue, string> = {
  rostered: 'On a team in this league.',
  free_agent: 'ESPN lists this player as claimable right now.',
  unknown:
    'ESPN never mentioned this player in the snapshot — he may sit below the free-agent depth we pull, or not be in ESPN’s player universe. Not the same as available.',
}

/** Who owns a player, or an honest "we do not know". */
export function OwnerBadge({ player }: { player: LeaguePlayer }) {
  if (player.is_mine) {
    return (
      <span className="badge mine" title="On your roster.">
        Mine
      </span>
    )
  }
  if (player.availability === 'rostered') {
    return (
      <span className="badge rostered" title={player.owner_team_name ?? undefined}>
        {player.owner_team_name ?? LABELS.rostered}
      </span>
    )
  }
  return (
    <span className={`badge ${player.availability === 'free_agent' ? 'free' : 'unlisted'}`}
          title={TITLES[player.availability]}>
      {LABELS[player.availability]}
    </span>
  )
}

const INJURY_LABELS: Record<string, string> = {
  QUESTIONABLE: 'Q',
  DOUBTFUL: 'D',
  OUT: 'OUT',
  INJURY_RESERVE: 'IR',
  SUSPENSION: 'SUSP',
  DAY_TO_DAY: 'DTD',
}

/** Injury tag, shown only when it is news. ACTIVE and NORMAL are not. */
export function InjuryBadge({ status }: { status: string | null }) {
  if (!status) return null
  const label = INJURY_LABELS[status]
  if (!label) return null

  const severe = status === 'OUT' || status === 'INJURY_RESERVE' || status === 'SUSPENSION'
  return (
    <span className={`badge ${severe ? 'out' : 'questionable'}`} title={status}>
      {label}
    </span>
  )
}

/**
 * Marks a player whose current team differs from the team his metrics came from.
 *
 * Worth its own signal: target share, air-yards share and every situational number on
 * the row were earned somewhere else, so they describe a situation that no longer
 * exists.
 */
export function MovedBadge({ player }: { player: LeaguePlayer }) {
  if (!player.changed_team) return null
  return (
    <span
      className="badge moved"
      title={`Metrics are from ${player.team}; now on ${player.espn_team}. Situational stats describe a team he has left.`}
    >
      {player.team}→{player.espn_team}
    </span>
  )
}
