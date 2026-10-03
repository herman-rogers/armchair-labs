import { useQuery } from '@tanstack/react-query'
import type { LeagueTeam, MetricVersion, LeaguePlayer } from '../api/types'
import { PlayerTable } from './PlayerTable'
import { IDENTITY } from './playerColumns'
import { forecastColumns } from './ForecastColumns'
import { rosterQuery } from '../api/queries/archive'
import { RankingExplanation } from './PlayerDetails'

/** One team's roster, joined to the board, decisions first. */
export function RosterPanel({
  team,
  version,
}: {
  team: LeagueTeam
  version: MetricVersion
}) {
  const roster = useQuery(rosterQuery(team.team_id, version))

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
      <RankingExplanation version={version} />
      <PlayerTable
        version={version}
        players={roster.data.players}
        columns={[...IDENTITY, ...forecastColumns<LeaguePlayer>(version)]}
        defaultSort="rank"
        emptyMessage="No skill players on this roster."
      />
      <p className="legend tight faint">
        Rookies and players without prior NFL production remain visible as ESPN-only.
        ESPN’s current PPR draft rank places them in the list; their model metrics stay
        blank and lineup comparisons still report them as unrankable.
      </p>
    </section>
  )
}
