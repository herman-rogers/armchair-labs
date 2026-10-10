import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router'
import type { LeagueObservations } from '../api/league'
import type { RankingsResponse } from '../api/chairstats'
import { weeklyForecastsQuery, teamStrengthQuery } from '../api/queries'
import { useDataRelease } from '../dataRelease'
import { leagueSummary } from '../leagueSummary'
import { teamStrengthSummary } from '../teamStrengthSummary'
import { currentWeeklyForecasts } from '../weeklyForecasts'
import { teamPath } from '../navigation'
import { fixed, rank } from '../format'
import { DataTable } from './DataTable'
import { QueryError } from './Controls'
import { StatCard, StatCards, StatList } from './StatCards'
import { LeagueMatchupWeek } from './LeagueMatchupWeek'

const pct = (value: number | null | undefined) => value == null ? '—' : `${fixed(value * 100)}%`

export function LeagueOverview({data, rankings, campaignId, horizon}: { data: LeagueObservations; rankings?: RankingsResponse; campaignId: string; horizon: string }) {
  const { token } = useDataRelease()
  const strength = useQuery(teamStrengthQuery(campaignId, horizon, token))
  const matchups = useQuery(weeklyForecastsQuery(campaignId, horizon, token))
  const forecasts = currentWeeklyForecasts(data, matchups.data)
  const summary = teamStrengthSummary(data, strength.data)
  const rosters = new Map(leagueSummary(data, rankings).map(t => [t.team_id, t]))
  const teams = summary?.map(t => ({ ...t, roster: rosters.get(t.team_id) })) ?? []
  const mine = teams.find(t => t.is_mine)
  return <section className="league-outlook" aria-label="League overview">
    <QueryError query={strength} label="Team strength unavailable" />
    {strength.isPending && <p role="status">Loading league team strength…</p>}
    {strength.data && !summary && <p className="notice">Team strength is waiting for a matching league snapshot. Refresh league to align the data.</p>}
    {summary && <>
      <StatCards label="Your team strength">
        <StatCard label="Your weekly offense rank" value={rank(mine?.weekly_rank)}><p>{mine?.team_name ?? 'Team not identified'} · {fixed(mine?.current.total)} projected points</p><p>Week {data.week} · Assumed offense · K/DST excluded</p></StatCard>
        <StatCard label="Your actual scoring" value={<>{fixed(mine?.results.ppg)} <span className="faint">pts / week</span></>}><p>{rank(mine?.results.ppg_rank)} in the league · {mine?.results.record ?? '—'}</p><p>{pct(mine?.results.all_play)} against every team</p></StatCard>
        <StatCard label="Your backup coverage" value={mine?.current.complete ? `${mine.current.replaceable}/${mine.current.required}` : '—'}><p>Offensive starter absences covered, one at a time</p><p>{mine?.current.usable_bench ?? '—'} usable backups · {fixed(mine?.current.mean_drop)} pts average loss</p></StatCard>
      </StatCards>
      {!strength.data?.forecast_available && <p className="notice">Weekly forecasts unavailable. Actual results remain available.{strength.data?.forecast_issue && ` ${strength.data.forecast_issue}`}</p>}
      <section className="outlook-standings strength-overview" aria-label="League team strength">
        <div className="outlook-section-head"><div><h3>League team strength</h3><p>Week {data.week} lineup outlook · Results through Week {Math.max(0, Math.min(data.week - 1, data.regular_season_weeks))} · Open a team for its full breakdown</p></div></div>
        <p className="legend">The same lineup, depth and scoring analysis as Team Strength. Weekly rank orders projected offensive lineup points; backup coverage and scoring balance show the wider picture.</p>
        {teams.some(t => t.weekly_rank == null) && <p className="legend">Weekly ranks require a complete offensive lineup forecast for every team. Available point estimates are still shown.</p>}
        <DataTable rows={teams} columns={[
          {key:'weekly_rank',label:'Weekly rank',title:'Rank by projected points for the assumed QB/RB/WR/TE lineup, excluding K/DST. Requires complete forecasts for every team; ties share rank.',initial:'asc',align:'left',render:r=>rank(r.weekly_rank)},
          {key:'team_name',label:'Team',title:'Open the full Team Strength breakdown.',align:'left',render:r=><div className="outlook-cell"><Link className="player-profile-link" to={teamPath(r)}>{r.team_name}{r.is_mine ? ' · You' : ''}</Link><small>{r.results.record}</small></div>},
          {key:'weekly_points',label:'Offense points',title:`Week ${data.week} offensive points excluding K/DST. Uses the same assumed substitutions as Team Strength.`,value:r=>r.current.total,render:r=><div className="outlook-cell"><strong>{fixed(r.current.total)}</strong><small>{r.current.adjustments.length || r.current.omitted.length ? 'Assumed substitutions' : 'Selected starters'}</small></div>},
          {key:'ppg',label:'Points / week',title:'Actual scoring from Team Strength, including K/DST.',value:r=>r.results.ppg,render:r=><div className="outlook-cell"><strong>{fixed(r.results.ppg)}</strong><small>{rank(r.results.ppg_rank)} in league</small></div>},
          {key:'all_play',label:'All-play',title:'Share of league opponents beaten on actual weekly scores; ties count half.',value:r=>r.results.all_play,render:r=>pct(r.results.all_play)},
          {key:'depth',label:'Backup coverage',title:'Offensive starter absences covered one at a time, allowing FLEX moves. Incomplete lineups are unavailable.',value:r=>r.current.complete && r.current.required ? r.current.replaceable / r.current.required : null,render:r=><div className="outlook-cell"><strong>{r.current.complete ? `${r.current.replaceable}/${r.current.required}` : '—'}</strong><small>{r.current.usable_bench} usable backups</small></div>},
          {key:'balance',label:'Top-two share',title:'Share of projected offensive points supplied by the top two starters. Describes concentration, not a strength score.',value:r=>r.current.top2_share,render:r=>pct(r.current.top2_share)},
          {key:'roster_rank',label:'Roster rank',title:'Campaign horizon QB/RB/WR/TE forecast rank, including bench and IR. Separate from weekly lineup strength.',initial:'asc',value:r=>r.roster?.team_rank,render:r=>rank(r.roster?.team_rank)},
        ]} defaultSort="weekly_rank" sortParam="sort" rowKey={r=>r.team_id} rowClass={r=>r.is_mine?'mine-row':undefined} rowLabel={r=>`${r.team_name} strength breakdown`} renderDetails={r=><div className="player-details">
          <h4>{r.team_name} · Strength summary</h4>
          <StatList items={[
            {label:'Starting offense',value:fixed(r.current.total),note:'K/DST excluded'},
            {label:'Best legal offense',value:fixed(r.current.best_legal)},
            {label:'Average replacement loss',value:fixed(r.current.mean_drop),note:'Requires every offensive absence covered'},
            {label:'Weakest three starters',value:fixed(r.current.weakest)},
            {label:'Scoring standard deviation',value:fixed(r.results.volatility),note:'Actual results'},
            {label:'Season scoring pace',value:fixed(r.results.season_pace)},
          ]} />
          <p><strong>Strong positions:</strong> {r.strengths.map(p=>`${p.position} ${rank(p.rank)}`).join(' · ') || 'No position in the top third, or comparable history unavailable.'}</p>
          <p><strong>Exposed positions:</strong> {r.weaknesses.map(p=>`${p.position} ${rank(p.rank)}`).join(' · ') || 'No position in the bottom third, or comparable history unavailable.'} Position ranks use actual starter scoring.</p>
          <p>{r.current.covered}/{r.current.required} offensive starters covered. {r.current.adjustments.join(' · ')}{r.current.omitted.length > 0 && ` · Omitted: ${r.current.omitted.map(p=>`${p.name} (${p.reason ?? 'unavailable'})`).join('; ')}`}</p>
          <p>Roster forecast: {rank(r.roster?.team_rank)} · {fixed(r.roster?.forecast_points)} points, including bench and IR.</p>
          <Link className="button" to={teamPath(r)}>View full team strength</Link>{' '}
          <Link className="button" to={`${teamPath(r)}#strength-correlation`}>Review correlation risk</Link>
        </div>} />
      </section>
    </>}
    <QueryError query={matchups} label="Campaign forecasts unavailable" />
    <p className="legend">Matchup projections below use current selected starters. Team strength above assumes eligible backups fill unavailable or empty slots.</p>
    <LeagueMatchupWeek data={data} forecasts={forecasts} />
    <details className="outlook-method"><summary>How to read team strength</summary>
      <p><strong>Weekly lineup:</strong> the same assumed offensive lineup as each team’s detail page, excluding K/DST. Eligible roster backups fill unavailable or empty slots while available selected starters remain. These assumptions do not change the actual roster. Weekly rank is a points comparison, not a win probability or a combined rating of depth and results.</p>
      <p><strong>Depth and balance:</strong> backup coverage tests one offensive absence at a time, allowing FLEX moves. Top-two share describes scoring concentration; lower concentration does not by itself mean a stronger team. Open a row for replacement losses, positional strengths and scoring consistency.</p>
      <p><strong>Results and roster:</strong> points per week and all-play use completed actual scores. Roster rank sums the chosen campaign's skill-player forecasts, including bench and IR. Correlation analysis is available in each team’s breakdown and does not adjust forecasts.</p>
    </details>
  </section>
}
