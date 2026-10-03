import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router'
import type { LeagueObservations, RankingsResponse } from '../api/nextgen'
import { weeklyForecastsQuery, rankingHistoryQuery } from '../api/queries'
import { useDataRelease } from '../dataRelease'
import { leagueOutlook } from '../leagueOutlook'
import { currentWeeklyForecasts, weeklyForecastLabel } from '../weeklyForecasts'
import { leaguePath, to } from '../navigation'
import { fixed, rank } from '../format'
import { DataTable } from './DataTable'
import { RankMovement } from './RankMovement'
import { QueryError } from './Controls'
import { WeeklyForecastReview } from './WeeklyForecastReview'
import { LeagueMatchupWeek } from './LeagueMatchupWeek'

const rosterLink = (teamId: number) => to(leaguePath('rosters'), { team: teamId })
function PointsChange({ value, label }: { value: number | null; label: string }) {
  if (value == null) return <small className="faint">No prior week</small>
  const rounded = Math.round(value * 10) / 10
  return <small className={rounded > 0 ? 'rank-up' : rounded < 0 ? 'rank-down' : 'faint'} aria-label={`${label}: ${rounded > 0 ? '+' : ''}${fixed(rounded)}`}>
    {rounded > 0 ? '+' : ''}{fixed(rounded)} vs last week
  </small>
}

export function LeagueOverview({data, rankings}: { data: LeagueObservations; rankings?: RankingsResponse }) {
  const { token } = useDataRelease()
  const history = useQuery(rankingHistoryQuery('rest_of_season', token))
  const matchups = useQuery(weeklyForecastsQuery(token))
  const forecasts = currentWeeklyForecasts(data, matchups.data)
  const { teams, prior_week, completed_weeks } = leagueOutlook(data, rankings, history.data, forecasts)
  const mine = teams.find(t => t.is_mine)
  const myGame = mine?.game
  const margin = myGame?.margin
  const favoriteText = myGame ? myGame.favorite ? `${myGame.favorite.team_name} by ${Math.abs(margin ?? 0) < .1 ? '<0.1' : fixed(Math.abs(margin ?? 0))}` : margin === 0 ? 'Even projection' : 'Projection unavailable' : 'Projection unavailable'
  return <section className="league-outlook" aria-label="League overview">
    <div className="league-update-cards outlook-kpis">
      <section><span className="eyebrow">Your roster forecast rank</span><h3>{rank(mine?.nextgen_team_rank)} <RankMovement current={mine?.nextgen_team_rank} previous={mine?.previous_forecast_rank} /></h3><p>{mine?.team_name ?? 'Team not identified'} · {mine?.record ?? '—'}</p><p>{prior_week != null ? `Forecasts through W${prior_week} → W${rankings?.report.through_week}` : 'Weekly rank comparison unavailable'}</p></section>
      <section><span className="eyebrow">Your Week {data.week} matchup</span><h3>{fixed(mine?.weekly_projection)} <span className="faint">vs</span> {fixed(mine?.opponent?.model_projection)}</h3><p>Projected winner: {favoriteText}</p><p>{weeklyForecastLabel(forecasts)} · Current starters</p></section>
      <section><span className="eyebrow">Your projected season pace</span><h3>{fixed(mine?.season_pace)} <span className="faint">pts</span></h3><PointsChange value={mine?.season_pace_change ?? null} label="Season pace change" /><p>{fixed(mine?.ppg)} points/week · {data.regular_season_weeks}-week regular season</p></section>
    </div>

    <QueryError query={matchups} label="NextGen weekly forecasts unavailable" />
    <LeagueMatchupWeek data={data} forecasts={forecasts} />

    <section className="outlook-standings" aria-label="Current league outlook">
    <div className="outlook-section-head"><div><h3>League outlook</h3><p>Current Patron ranks · Scoring through Week {completed_weeks} · Changes vs previous week</p></div></div>
    <QueryError query={history} label="Previous forecast ranks unavailable">. Current ranks and matchup projections remain available.</QueryError>
    <DataTable rows={teams} columns={[
      {key:'nextgen_team_rank',label:'Rank',title:'Patron rank by remaining-season points across the full QB/RB/WR/TE roster. Includes bench and IR; not a win probability.',initial:'asc',align:'left',render:r=>rank(r.nextgen_team_rank)},
      {key:'forecast_change',label:'Change',align:'left',title:prior_week != null ? `Current rosters scored with W${prior_week} and W${rankings?.report.through_week} forecasts. Up is a better rank; #2 → #4 is down two.` : 'No comparable previous-week forecast available.',render:r=><RankMovement current={r.nextgen_team_rank} previous={r.previous_forecast_rank} />},
      {key:'team_name',label:'Team',title:'Open this team’s roster.',align:'left',render:r=><Link className="player-profile-link" to={rosterLink(r.team_id)}>{r.team_name}{r.is_mine ? ' · You' : ''}</Link>},
      {key:'record',label:'Record',title:'Completed regular-season wins–losses–ties.',align:'left'},
      {key:'weekly_projection',label:`Week ${data.week} projected`,title:'Dedicated weekly model points for the current starting lineup; K/DST and unpromoted positions use labeled references.',render:r=><div className="outlook-cell"><strong>{fixed(r.weekly_projection)}</strong><small title={r.opponent?.team_name}>{r.opponent ? `vs ${r.opponent.team_name}` : 'No matchup captured'}</small></div>},
      {key:'ppg',label:'Points / week',title:'Actual completed regular-season points divided by weeks played. Change compares the average through the prior week.',render:r=><div className="outlook-cell"><strong>{fixed(r.ppg)}</strong><PointsChange value={r.ppg_change} label="Weekly scoring average change" /></div>},
      {key:'season_pace',label:'Projected season pace',title:`Actual points per week × ${data.regular_season_weeks} regular-season weeks. A scoring-pace estimate, not a schedule-adjusted forecast. Change is versus last week’s pace.`,render:r=><div className="outlook-cell"><strong>{fixed(r.season_pace)}</strong><PointsChange value={r.season_pace_change} label="Season pace change" /></div>},
    ]} defaultSort="nextgen_team_rank" sortParam="sort" rowKey={r=>r.team_id} rowClass={r=>r.is_mine?'mine-row':undefined} rowLabel={r=>`${r.team_name} forecast breakdown`} renderDetails={r=><div className="player-details">
      <h4>Weekly scoring history</h4>
      <ol className="standings-history">{(r.schedule ?? []).filter(g => g.week >= 1 && g.week <= completed_weeks && ['W','L','T'].includes(g.outcome)).map(g => <li key={g.week}>Week {g.week}: <strong>{fixed(g.score)} pts</strong> · {g.outcome} vs {data.teams.find(t=>t.team_id===g.opponent_team_id)?.team_name ?? 'Unknown opponent'}</li>)}</ol>
      <p>{fixed(r.completed_points)} points scored · {fixed(r.ppg)} per week · {fixed(r.season_pace)} projected at this pace over {data.regular_season_weeks} weeks.</p>
      <p>Patron roster forecast: {rank(r.nextgen_team_rank)}{r.previous_forecast_rank != null ? `, previously #${r.previous_forecast_rank} on the same roster` : ''}. {fixed(r.forecast_points)} remaining player points across {r.forecast_count}/{r.skill_count} skill players, including bench and IR. This total is separate from your starting-lineup season pace.</p>
      <p>{r.reference_count} reference forecasts · {r.constrained_count} availability constraints · FAAB: {r.faab_remaining == null ? 'Unknown' : `$${r.faab_remaining}`}</p>
      <Link className="button" to={rosterLink(r.team_id)}>View player forecasts</Link>
    </div>} />
    </section>
    <WeeklyForecastReview forecasts={forecasts} />
    <details className="outlook-method"><summary>How to read these numbers</summary>
      <p><strong>Forecast rank and movement:</strong> current rosters ranked by the sum of published remaining-season QB/RB/WR/TE forecasts. Bench and IR count; K/DST do not. Movement compares the same roster against the previous week’s published forecasts, so it measures forecast changes, not past ownership or win/loss standings. Incomplete comparisons show no prior rank.</p>
      <p><strong>Matchup forecasts:</strong> The dedicated one-week model learns from earlier seasons using scoring, opportunity, snaps and prior opponent results. Positions that fail promotion use a directly evaluated weekly reference. K/DST use up to four earlier recorded scores. Winner picks are comparisons of forecast points, not calibrated win probabilities. ESPN supplies lineup selections and observed scores only.</p>
      <p><strong>Season pace:</strong> actual completed regular-season scoring average multiplied by {data.regular_season_weeks}. It includes K/DST and excludes partial weeks, bench points and playoffs. It assumes the same scoring pace continues; it does not account for future injuries, byes or roster changes. Changes compare with the previous completed week’s average and pace.</p>
    </details>
  </section>
}
