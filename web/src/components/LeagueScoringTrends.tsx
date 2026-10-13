import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router'
import { leagueScoringQuery } from '../api/queries'
import { useDataRelease } from '../dataRelease'
import { fixed } from '../format'
import { teamPath } from '../navigation'
import { QueryError } from './Controls'
import { ScoringChart, ScoringLegend } from './ScoringChart'
import { scoringDomain } from '../scoringTrends'

export function LeagueScoringTrends({ campaignId }: { campaignId: string }) {
  const { token } = useDataRelease()
  const query = useQuery(leagueScoringQuery(campaignId, token))
  const data = query.data
  return <section className="league-scoring-trends" id="scoring-trends" aria-label="Team scoring trends">
    <div className="outlook-section-head"><h3>Actual vs predicted · Full season</h3></div>
    <QueryError query={query} label="Team scoring trends unavailable" />
    {query.isPending && <p role="status">Loading team scoring history…</p>}
    {data && <>
      <ScoringLegend teams />
      <p className="legend">Future weeks fill bye slots with the best eligible bench players under league position and FLEX rules. Uncovered slots score zero. Hover a week to see lineup changes.</p>
      {data.stale && <p className="notice">These charts use a saved league snapshot. Refresh data for the latest lineups and results.</p>}
      <div className="scoring-grid">{data.teams.map(team => <Link className="scoring-card scoring-card-link" key={team.team_id} to={teamPath({ team_id: team.team_id, team_name: team.name })} aria-label={`View ${team.name} team breakdown`}>
        <ScoringChart series={team} description={`Season avg ${fixed(team.average)} pts · Average error ${fixed(team.mae)} pts`} throughWeek={data.through_week} lastWeek={data.last_week} domain={scoringDomain(data.teams)} teams showValues={false} />
      </Link>)}</div>
      <details className="outlook-method"><summary>Future lineup assumptions</summary><p>{data.future_basis}</p></details>
    </>}
  </section>
}
