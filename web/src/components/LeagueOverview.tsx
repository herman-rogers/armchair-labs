import type { LeagueObservations, RankingsResponse } from '../api/nextgen'
import { leagueSummary, recentTransactions } from '../leagueSummary'
import { Link } from 'react-router'
import { DataTable } from './DataTable'
import { leaguePath, to } from '../navigation'

const n = (v: number | null | undefined) => v == null ? '—' : v.toFixed(1)
const rosterLink = (teamId: number) => to(leaguePath('rosters'), { team: teamId })

export function LeagueOverview({data, rankings}: { data: LeagueObservations; rankings?: RankingsResponse }) {
  const teams = leagueSummary(data, rankings)
  const mine = teams.find(t => t.is_mine)
  const transactions = recentTransactions(data)
  return <section aria-label="League overview">
    <div className="league-update-cards">
      <section><span className="eyebrow">Your team</span><h3>{mine?.team_name ?? 'Team not identified'}</h3><p>{mine ? `${mine.wins} wins · ${mine.losses} losses` : 'Choose a roster below'}</p><p>{mine?.last_result}</p></section>
      <section><span className="eyebrow">NextGen roster forecast</span><h3>{mine?.nextgen_team_rank == null ? 'Incomplete / unavailable' : `#${mine.nextgen_team_rank} of ${teams.filter(t=>t.nextgen_team_rank != null).length} covered teams`}</h3><p>{n(mine?.forecast_points)} remaining points across QB/RB/WR/TE</p><p>{mine?.forecast_count ?? 0}/{mine?.skill_count ?? 0} skill players covered · includes bench</p></section>
      <section><span className="eyebrow">Around the league</span><h3>{teams.reduce((sum,t)=>sum+t.urgent,0)} urgent player flags</h3><p>{teams.reduce((sum,t)=>sum+t.alerts,0)} rostered players need review</p><p>{transactions.length} captured transactions · Your FAAB: {mine?.faab_remaining ?? 'Unknown'}</p></section>
    </div>
    <h3>Standings & roster forecasts</h3>
    <p className="legend">NextGen team rank orders the sum of current roster players’ remaining-season forecasts, including bench and reserve. It is not a win forecast or an optimized starting lineup; larger rosters can have larger totals. K/DST are not modeled. Incomplete rosters receive no team rank.</p>
    <p className="legend">NextGen average rank is the average overall rank of ranked players on each roster, including bench and reserve. Lower is better; unranked players, kickers and defenses are excluded.</p>
    <DataTable rows={teams} columns={[
      {key:'team_name',label:'Team',title:'Open this team’s current roster and individual forecasts.',align:'left', render:r=><Link className="player-profile-link" to={rosterLink(r.team_id)}>{r.team_name}{r.is_mine ? ' · You' : ''}</Link>},
      {key:'wins',label:'Wins',title:'Recorded wins; sorting is not an official playoff tiebreaker.'},
      {key:'losses',label:'Losses',title:'Recorded losses.'},
      {key:'points_for',label:'Points for',title:'Sum of captured completed schedule results; excludes the current partial week.',render:r=>n(r.points_for)},
      {key:'points_against',label:'Points against',title:'Opponent scores for those completed games. Unknown if an opposing result is missing.',render:r=>n(r.points_against)},
      {key:'results_captured',label:'Results captured',title:'Completed schedule scores included in points for/against.'},
      {key:'nextgen_team_rank',label:'NextGen · team',title:'Rank by full remaining-season skill-roster total among completely covered teams; ties share rank.',initial:'asc',render:r=>r.nextgen_team_rank == null ? '—' : `#${r.nextgen_team_rank}`},
      {key:'average_overall_rank',label:'NextGen · average rank',title:'Sum of rostered players’ published overall ranks divided by the number of ranked players, including bench and reserve. Lower is better; unranked players are excluded.',initial:'asc',render:r=><span title={`${r.ranked_count} of ${r.skill_count} skill players ranked`}>{n(r.average_overall_rank)}</span>},
      {key:'forecast_points',label:'Remaining points',title:'Sum of published remaining-season forecasts for every rostered skill player, including reserve.',render:r=>n(r.forecast_points)},
      {key:'forecast_count',label:'Coverage',title:'Covered QB/RB/WR/TE players / captured QB/RB/WR/TE players.',render:r=>`${r.forecast_count}/${r.skill_count}`},
      {key:'urgent',label:'Urgent flags',title:'Rostered players with a captured urgent alert; review the roster.'},
      {key:'faab_remaining',label:'FAAB',title:'Captured remaining acquisition budget.'},
    ]} defaultSort="wins" sortParam="sort" rowKey={r=>r.team_id} rowClass={r=>r.is_mine?'mine-row':undefined} rowLabel={r=>`${r.team_name} forecast breakdown`} renderDetails={r=><div className="player-details">
      <p>{r.last_result} · Division: {r.division_name ?? 'Not recorded'}</p>
      <p>NextGen average overall rank: {n(r.average_overall_rank)} · {r.ranked_count}/{r.skill_count} skill players ranked, including bench and reserve.</p>
      <p>{n(r.starter_points)} known remaining points in currently filled starting slots (not optimized). All roster slots: {n(r.known_points)} points from covered players; {r.reference_count} reference forecasts; {r.constrained_count} dated availability constraints; {r.unmodeled_count} unmodeled roster entries.</p>
      <DataTable rows={r.positions} columns={[
        {key:'position',label:'Position',title:'Skill position.',align:'left'},
        {key:'players',label:'Rostered',title:'All current roster players at this position.'},
        {key:'covered',label:'Forecasts',title:'Players with a published forecast; missing is not zero.'},
        {key:'points',label:'Known remaining points',title:'Sum of available player forecasts; check coverage before comparing.',render:p=>n(p.points)},
      ]} defaultSort="position" rowKey={p=>p.position} />
      <Link className="button" to={rosterLink(r.team_id)}>View player forecasts</Link>
    </div>} />
    <details className="league-recent-activity"><summary>Latest captured activity · {transactions.length} transactions</summary>
      {transactions.length ? <ul>{transactions.slice(0,8).map((t,i)=><li key={i}><strong>{t.team_name ?? 'Unknown team'}</strong> · {t.kind ?? 'Transaction'} · {t.player_name ?? 'Unknown player'}{t.bid_amount != null ? ` · $${t.bid_amount}` : ''}<small>{t.date ? new Date(t.date).toLocaleString() : 'Date not captured'}</small></li>)}</ul> : <p>No transactions captured.</p>}
      <Link className="button" to={leaguePath('transactions')}>All transactions</Link>
    </details>
  </section>
}
