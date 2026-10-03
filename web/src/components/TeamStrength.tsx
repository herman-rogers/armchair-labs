import type { ReactNode } from 'react'
import { useQuery } from '@tanstack/react-query'
import type { LeagueObservations, RankingsResponse } from '../api/nextgen'
import { teamStrengthQuery } from '../api/queries'
import { useDataRelease } from '../dataRelease'
import { fixed, rank } from '../format'
import { leagueSummary } from '../leagueSummary'
import { DataTable } from './DataTable'
import { QueryError } from './Controls'
import { PlayerLink } from './PlayerLink'
import { StatCard as Stat, StatCards, StatList } from './StatCards'
import { AnalysisPanel } from './AnalysisPanel'
import { CompositionChart, ContributionChart, PositionChart, ReplacementChart, ResultsChart } from './TeamStrengthCharts'
import { Link } from 'react-router'
import { teamPath } from '../navigation'
import { TeamCorrelationRisk } from './TeamCorrelationRisk'

const pct = (v: number | null | undefined) => v == null ? '—' : `${fixed(v * 100)}%`
const signed = (v: number | null | undefined) => v == null ? '—' : `${v > 0 ? '+' : ''}${fixed(v)}`

export function TeamStrength({ data, rankings, teamId, children }: {
  data: LeagueObservations; rankings?: RankingsResponse; teamId: number; children: ReactNode;
}) {
  const { token } = useDataRelease()
  const query = useQuery(teamStrengthQuery(token))
  const matching = query.data?.season === data.season && query.data.week === data.week && query.data.captured_at === data.captured_at
    && (!query.data.version || query.data.version === data.version)
  const team = matching ? query.data?.teams.find(t => t.team_id === teamId) : undefined
  const roster = leagueSummary(data, rankings).find(t => t.team_id === teamId)
  const playerId = (id: number) => data.players.find(p => p.espn_id === id)?.player_id
  const results = team?.results, current = team?.current
  const strengths = team?.positions.filter(p => p.rank != null && p.rank <= Math.ceil(data.teams.length * .3)) ?? []
  const weaknesses = team?.positions.filter(p => p.rank != null && p.rank > Math.floor(data.teams.length * .7)) ?? []
  const unavailable = team?.players.filter(p => !p.usable) ?? []
  const missingStarters = team?.lineup.filter(p => !p.usable) ?? []
  return <section className="team-strength" aria-label="Team strength breakdown">
    <QueryError query={query} label="Team strength unavailable" />
    {query.isPending && <p role="status">Loading team strength…</p>}
    {query.data && !matching && <p className="notice">Team strength is waiting for a matching league snapshot. Refresh league to align the data.</p>}
    {team && results && current && <>
      <nav className="page-links strength-sections" aria-label="Team strength sections">
        <a href="#strength-lineup">Starting lineup</a><a href="#strength-balance">Scoring balance</a><a href="#strength-depth">Usable depth</a><a href="#strength-positions">Positions</a><a href="#strength-results">Results</a><a href="#strength-correlation">Correlation risk</a><a href="#strength-roster">Players</a>
      </nav>
      <StatCards columns={4} compact label="Team summary">
        <Stat label="Actual scoring" value={<>{fixed(results.ppg)} <small>pts / week</small></>}><p>{rank(results.ppg_rank)} · {results.record} · {signed(results.ppg_change)} vs last week</p></Stat>
        <Stat label={`Week ${data.week} · NextGen`} value={<>{fixed(current.weekly_total)} <small>points</small></>}><p>Includes K/DST · {current.covered}/{current.required} offensive starters covered</p></Stat>
        <Stat label="Against every team" value={pct(results.all_play)}><p>{rank(results.all_play_rank)} · opponents beaten; ties count half</p></Stat>
        <Stat label="Roster forecast" value={rank(roster?.nextgen_team_rank)}><p>{fixed(roster?.forecast_points)} remaining points · includes bench/IR</p></Stat>
      </StatCards>
      {(current.adjustments.length > 0 || current.omitted.length > 0) && <p className="notice" aria-label="Assumed lineup adjustments">Estimates use roster backups for unavailable starters. {current.omitted.map(p => `${p.name} (${p.reason ?? 'unavailable'})`).join('; ')}{current.omitted.length > 0 && '. '}{current.adjustments.join(' · ')}. {current.covered < current.required && 'Some offensive slots still lack an eligible player with a forecast. '}These are assumed substitutions for analysis.</p>}
      {!query.data?.forecast_available && <p className="notice">Weekly forecasts unavailable. Actual results remain available.{query.data?.forecast_issue && ` ${query.data.forecast_issue}`}</p>}
      <AnalysisPanel compact aria-label="Team strengths and weaknesses"><div className="strength-insights">
        <section><h4>Where this team is strong</h4>{strengths.length ? <ul>{strengths.map(p => <li key={p.position}><strong>{p.position}: {rank(p.rank)} in the league</strong> · {fixed(p.actual_ppg)} actual starter points/week.</li>)}</ul> : <p>No position in the league’s top third on completed scoring, or comparable history is unavailable.</p>}</section>
        <section><h4>Where this team is exposed</h4>{weaknesses.length || missingStarters.length ? <ul>{weaknesses.map(p => <li key={p.position}><strong>{p.position}: {rank(p.rank)} in the league</strong> · {fixed(p.actual_ppg)} actual starter points/week.</li>)}{missingStarters.map(p => <li key={p.espn_id}><strong>{p.name}</strong> · {p.reason}</li>)}</ul> : <p>No position in the bottom third of completed scoring and no captured starter availability gaps.</p>}</section>
      </div></AnalysisPanel>
      <div className="strength-chart-grid">
      <AnalysisPanel compact id="strength-lineup" aria-label="Starting-lineup strength" heading="Starting-lineup strength" meta={`NextGen · Week ${data.week}`}>
        <ContributionChart team={team} />
        <StatList items={[
          {label:'Starting offense',value:fixed(current.total),note:'K/DST excluded'},
          {label:'Middle four',value:fixed(current.middle),note:'Starters ranked 3–6'},
          {label:'Weakest three',value:fixed(current.weakest)},
          {label:'Median starter',value:fixed(current.median)},
          {label:'Best legal offense',value:fixed(current.best_legal)},
          {label:'Possible lineup gain',value:current.total != null && current.best_legal != null ? signed(current.best_legal-current.total) : '—'},
        ]} />
        <p className="legend">Selected starters with eligible roster backups filling unavailable or empty slots. Totals require forecasts for every slot.</p>
      </AnalysisPanel>
      <AnalysisPanel compact id="strength-balance" aria-label="Scoring concentration" heading="Scoring concentration" meta="Actuals + forecast">
        <CompositionChart team={team} />
        <StatList items={[
          {label:'Projected top-two share',value:pct(current.top2_share),note:`${fixed(current.top2)} points · rest ${fixed(current.other_starters)}`},
          {label:'Effective contributors',value:fixed(current.effective_contributors),note:`Equal shares = ${current.required}`},
          {label:'Actual top-two share',value:pct(results.top2_share),note:`${results.breakdown_weeks} reconciled weeks`},
          {label:'Same two, season share',value:pct(results.cumulative_top2_share)},
          {label:'Different top-two scorers',value:results.different_top_scorers},
        ]} />
        <p className="legend">Balance describes contributions; it does not establish a reliable floor or win probability.</p>
      </AnalysisPanel>
      <AnalysisPanel compact id="strength-depth" aria-label="Usable depth" heading="Usable depth" meta={`NextGen · Week ${data.week}`}>
        <ReplacementChart team={team} />
        <StatList items={[
          {label:'Usable backups',value:current.usable_bench},
          {label:'Absences covered',value:`${current.replaceable}/${current.required}`},
          {label:'Average loss',value:fixed(current.mean_drop),note:'Requires every slot covered'},
          {label:'Largest covered loss',value:fixed(current.worst_drop)},
        ]} />
        <p className="legend">One absence at a time, allowing FLEX moves. Backups cannot cover simultaneous absences.</p>
        <details className="outlook-method"><summary>Replacement details & FLEX moves</summary>
        <DataTable rows={team.depth} columns={[
          {key:'starter',label:'Missing starter',title:'Missing starter',align:'left',render:r=><PlayerLink playerId={playerId(r.espn_id)}>{r.starter}</PlayerLink>},
          {key:'starter_points',label:'Weekly points',title:'Weekly points',render:r=>fixed(r.starter_points)},
          {key:'replacement',label:'Backup entering lineup',title:'Backup entering lineup',align:'left',render:r=>r.replacement ?? r.reason},
          {key:'replacement_points',label:'Backup points',title:'Backup points',render:r=>fixed(r.replacement_points)},
          {key:'drop',label:'Point loss',title:'Positive is a loss; negative means the model prefers the backup. Unknown is not zero.',render:r=>fixed(r.drop)},
          {key:'replacement_status',label:'Backup status',title:'Backup status',align:'left'},
          {key:'conservative_drop',label:'Loss excluding questionable',title:'Loss excluding questionable',render:r=>fixed(r.conservative_drop)},
        ]} defaultSort="drop" rowKey={r=>r.espn_id} rowLabel={r=>r.starter} renderDetails={r=><p>{r.moves.length ? r.moves.join(' · ') : r.reason ?? 'No slot reshuffling needed.'}</p>} emptyMessage="No offensive starters captured." /></details>
      </AnalysisPanel>
      <AnalysisPanel compact id="strength-positions" aria-label="Position breakdown" heading="Position breakdown" meta="Actual scoring">
        <PositionChart team={team} teams={query.data!.teams} />
        <details className="outlook-method"><summary>Position ranks, backups & availability</summary>
        <DataTable rows={team.positions} columns={[
          {key:'position',label:'Position',title:'Position',align:'left'}, {key:'actual_ppg',label:'Actual points / week',title:'Actual points / week',render:r=>fixed(r.actual_ppg)},
          {key:'rank',label:'League rank',title:'League rank',render:r=>rank(r.rank)}, {key:'weekly_points',label:`Week ${data.week} points`,title:'Published weekly starter estimates for this position.',render:r=>fixed(r.weekly_points)},
          {key:'starters',label:'Assumed starters',title:'Starters after filling unavailable or empty slots from the roster'}, {key:'usable_backups',label:'Usable backups',title:'Usable backups remaining after assumed substitutions'},
          {key:'best_backup',label:'Highest projected backup',title:'Highest projected backup',align:'left'}, {key:'unavailable',label:'Unavailable / uncovered',title:'Unavailable / uncovered'},
        ]} defaultSort="actual_ppg" rowKey={r=>r.position} /></details>
      </AnalysisPanel>
      </div>
      <AnalysisPanel compact id="strength-results" aria-label="Results and consistency" heading="Results & consistency" meta={`${results.weeks} completed weeks`}><p className="legend">{results.weeks} completed weeks. Short samples do not establish a dependable scoring floor.</p>
        <div className="strength-chart-grid">
        <div><ResultsChart team={team} teams={query.data!.teams} /></div>
        <StatList items={[
          {label:'Points scored',value:fixed(results.points)},
          {label:'Opponent pts / week',value:fixed(results.points_against)},
          {label:'Average margin',value:signed(results.average_margin)},
          {label:'Best / worst week',value:`${fixed(results.high)} / ${fixed(results.low)}`},
          {label:'Scoring std. deviation',value:fixed(results.volatility),note:'Observed, not forecast uncertainty'},
          {label:'Season scoring pace',value:fixed(results.season_pace),note:`${data.regular_season_weeks} weeks at this average`},
          {label:'Actual middle four / week',value:fixed(results.middle_ppg)},
          {label:'Actual weakest three / week',value:fixed(results.weakest_ppg)},
          {label:'FAAB remaining',value:team.faab_remaining == null ? '—' : `$${team.faab_remaining}`},
        ]} /></div>
        <details className="outlook-method"><summary>Weekly scoring history</summary>
        <DataTable rows={team.history} columns={[
          {key:'week',label:'Week',title:'Week',initial:'asc'}, {key:'score',label:'Points',title:'Points',render:r=>fixed(r.score)},
          {key:'opponent_team_id',label:'Opponent',title:'Opponent',align:'left',render:r=>{ const opponent=data.teams.find(t=>t.team_id===r.opponent_team_id); return opponent ? <Link to={teamPath(opponent)}>{opponent.team_name}</Link> : 'Unknown' }},
          {key:'opponent_score',label:'Opponent points',title:'Opponent points',render:r=>fixed(r.opponent_score)}, {key:'outcome',label:'Result',title:'Result'},
          {key:'scoring_rank',label:'Scoring rank',title:'Scoring rank',render:r=>rank(r.scoring_rank)}, {key:'all_play',label:'Against every team',title:'Against every team',render:r=>pct(r.all_play)},
        ]} defaultSort="week" rowKey={r=>r.week} emptyMessage="No completed scoring history." /></details>
      </AnalysisPanel>
      <TeamCorrelationRisk team={team} />
      {unavailable.length > 0 && <details className="outlook-method"><summary>Availability & forecast gaps · {unavailable.length} players</summary><ul>{unavailable.map(p=><li key={p.espn_id}><strong>{p.name}</strong> · {p.slot} · {p.reason}</li>)}</ul></details>}
    </>}
    <AnalysisPanel compact id="strength-roster" aria-label="Team roster" heading="Players & roster forecasts" meta="Current roster · Remaining season">{children}</AnalysisPanel>
    <details className="outlook-method"><summary>How these team statistics work</summary><p>Current strength and replacement estimates use published NextGen weekly points. Historical results use ESPN actual scores and recorded starters. Neither ESPN predictions nor rest-of-season ranks enter weekly estimates.</p><p>Unavailable starters (OUT, IR, suspended, doubtful or on bye) and empty slots are filled using the highest projected eligible roster backups, allowing FLEX moves while retaining available selected starters. The same assumed lineup drives totals, player contributions and depth; backups already used cannot be reused as depth. Questionable players remain eligible; the depth table also shows losses when they are excluded. Assumed substitutions do not change the ESPN roster. Game locks are not modeled. Missing forecasts stay unknown; the best legal lineup is limited to covered players.</p><p>Middle four means ranks 3–6 among offensive starters; weakest three means the bottom three. Concentration uses positive offensive points. Effective contributors is one divided by the sum of squared scoring shares. These describe balance, not independence between players or a calibrated win probability.</p></details>
  </section>
}
