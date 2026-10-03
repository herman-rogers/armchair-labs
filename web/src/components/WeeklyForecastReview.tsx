import { weeklyForecastLabel } from '../weeklyForecasts'
import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router'
import { forecastAccuracyQuery, weeklyEvidenceQuery } from '../api/queries'
import type { WeeklyForecasts } from '../api/nextgen'
import { useDataRelease } from '../dataRelease'
import { matchupPath } from '../navigation'
import { fixed } from '../format'
import { DataTable } from './DataTable'
import { QueryError } from './Controls'

export function WeeklyForecastReview({ forecasts }: { forecasts?: WeeklyForecasts }) {
  const { token } = useDataRelease()
  const query = useQuery(forecastAccuracyQuery(token))
  const evidence = useQuery({ ...weeklyEvidenceQuery(token), enabled: forecasts?.source === 'nextgen_weekly_model' })
  return <>
    {forecasts && <details className="outlook-method"><summary>Weekly player point forecasts</summary>
      <p>{weeklyForecastLabel(forecasts)} · Forecast inputs through Week {forecasts.through_week}. Missing forecasts are not counted as zero.</p>
      {forecasts.matchups.flatMap(g => [g.home, g.away]).map(team => <section key={team.team_id}><h4>{team.team_name} · {team.covered_starters}/{team.starter_count} starters covered</h4>
        <DataTable rows={team.forecasts} columns={[
          {key:'player_display_name',label:'Player',title:'Captured lineup selection.',align:'left'},
          {key:'slot',label:'Slot',title:'Captured roster slot.',align:'left'},
          {key:'prediction',label:'NextGen weekly points',title:'Dedicated one-week estimate; reference scope is shown in Forecast basis. Bench points do not count toward the team total.',render:r=>fixed(r.prediction)},
          {key:'basis',label:'Forecast basis',title:'Recipe or reason no estimate is available.',align:'left',render:r=>r.basis ?? r.unavailable_reason},
        ]} defaultSort="slot" rowKey={r=>r.espn_id} />
      </section>)}
    </details>}
    {forecasts?.source === 'nextgen_weekly_model' && <section aria-label="Weekly model evaluation"><h3>Weekly model evaluation</h3>
      <p>Historical season-by-season testing, 2019–2025. Each forecast used a model trained and selected on earlier seasons. Errors include all eligible offensive players, including those who scored zero; starter errors can be higher.</p>
      <QueryError query={evidence} label="Weekly evaluation unavailable" />
      {evidence.data && <>
        <DataTable rows={evidence.data.positions} columns={[
          {key:'position',label:'Position',title:'Offensive position.',align:'left'},
          {key:'mae',label:'Model point error',title:'Mean absolute error of the chronological challenger policy.',value:r=>r.modern.model.mae,render:r=>fixed(r.modern.model.mae,2)},
          {key:'reference',label:'Reference error',title:'Mean absolute error of the chronological weekly reference policy.',render:r=>fixed(r.modern.reference.mae,2)},
          {key:'cases',label:'Player-weeks',title:'Evaluated candidate player-weeks, including low-usage players.',render:r=>r.modern.model.n.toLocaleString()},
          {key:'approved',label:'Serving decision',title:'Only positions passing every promotion gate serve the learned challenger.',align:'left',render:r=>r.approved ? 'Validated weekly policy' : 'Weekly reference retained'},
        ]} defaultSort="position" rowKey={r=>r.position} />
        <p className="legend">Promotion requires a meaningful improvement over weekly references, consistent results across seasons, and no material cohort regressions. A smaller average error alone does not pass.</p>
        <details className="outlook-method"><summary>This season’s reconstructed matchup test</summary>
          <p>{evidence.data.replay.method}</p>
          <p>This replay uses the serving decisions above, including references for positions that did not pass promotion.</p>
          <p>{evidence.data.replay.accuracy == null ? 'No complete matchups to score.' : `${evidence.data.replay.correct}/${evidence.data.replay.evaluated} winner picks correct (${fixed(evidence.data.replay.accuracy*100)}%).`} Average team point error: {fixed(evidence.data.replay.point_mae)}. {evidence.data.replay.total-evidence.data.replay.evaluated} games excluded.</p>
          <DataTable rows={evidence.data.replay.games} columns={[
            {key:'week',label:'Week',title:'Model inputs stop before this week.',initial:'asc'},
            {key:'home',label:'Home',title:'Recorded home team.',align:'left'}, {key:'away',label:'Away',title:'Recorded away team.',align:'left'},
            {key:'prediction',label:'Model points',title:'Reconstructed home–away estimates.',render:r=>`${fixed(r.projected_home)} – ${fixed(r.projected_away)}`},
            {key:'actual',label:'Actual points',title:'Recorded home–away scores.',render:r=>`${fixed(r.home_score)} – ${fixed(r.away_score)}`},
            {key:'status',label:'Result',title:'Incomplete or tied matchups do not enter winner accuracy.',align:'left'},
          ]} defaultSort="week" rowKey={r=>`${r.week}-${r.home_id}-${r.away_id}`} />
        </details>
        <p className="legend">{evidence.data.limitations.join(' ')}</p>
      </>}
    </section>}
    <section aria-label="Prediction accuracy"><h3>Season prediction accuracy</h3>
      <p className="legend">Only forecasts saved before the week started count. Later reconstructions and today’s rankings cannot stand in for a pregame pick.</p>
      <QueryError query={query} label="Prediction accuracy unavailable" />
      {query.isPending && <p role="status">Loading saved predictions…</p>}
      {query.data && <>
        <p><strong>{query.data.accuracy == null ? 'No verified pregame picks yet' : `${fixed(query.data.accuracy * 100)}% correct`}</strong> · {query.data.correct}/{query.data.evaluated} scored picks · {query.data.total - query.data.evaluated} games excluded through Week {query.data.through_week}</p>
        {query.data.point_mae != null && <p>Average point error: <strong>{fixed(query.data.point_mae)} points</strong> per team across {query.data.point_teams} scored team forecasts.</p>}
        <p className="legend">{query.data.method} Forecasts are saved when the weekly forecast view loads; no historical records are backdated.</p>
        <DataTable rows={query.data.games} columns={[
          {key:'week',label:'Week',title:'Completed regular-season week.',initial:'asc'},
          {key:'matchup',label:'Matchup',title:'View recorded scores and lineups.',align:'left',render:r=><Link to={matchupPath(r.week,r.home_id,r.away_id)}>{r.home} vs {r.away}</Link>},
          {key:'predicted_winner',label:'Pregame pick',title:'Verified saved model pick.',align:'left',render:r=>r.predicted_winner ?? '—'},
          {key:'projected_score',label:'Model points',title:'Verified pregame point forecasts, home–away.',render:r=>`${fixed(r.projected_home)} – ${fixed(r.projected_away)}`},
          {key:'score',label:'Final score',title:'Recorded scores, home–away.',render:r=>`${fixed(r.home_score)} – ${fixed(r.away_score)}`},
          {key:'status',label:'Result',title:'Only verified pregame picks with decisive completed results enter accuracy.',align:'left'},
        ]} defaultSort="week" rowKey={r=>`${r.week}-${r.home_id}-${r.away_id}`} emptyMessage="No completed matchups yet." />
      </>}
    </section>
  </>
}
