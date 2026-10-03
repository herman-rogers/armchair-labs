import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router'
import { matchupsQuery } from '../api/queries'
import type { LeagueObservations, WeeklyForecasts } from '../api/nextgen'
import { useDataRelease } from '../dataRelease'
import { fixed } from '../format'
import { matchupPath, useUrlNumber } from '../navigation'
import { QueryError } from './Controls'
import { currentWeeklyForecasts, weeklyForecastLabel } from '../weeklyForecasts'

/** The overview owns week browsing; a game link opens its dedicated lineup page. */
export function LeagueMatchupWeek({ data, forecasts }: { data: LeagueObservations; forecasts?: WeeklyForecasts }) {
  const { token } = useDataRelease()
  const [selected, setWeek] = useUrlNumber('week', data.week)
  const week = Number.isInteger(selected) && selected >= 1 && selected <= 25 ? selected : data.week
  const query = useQuery(matchupsQuery(week, token))
  const matches = query.data?.season === data.season && query.data.requested_week === week ? query.data : undefined
  const weeks = matches?.available_weeks ?? Array.from({ length: data.regular_season_weeks }, (_, i) => i + 1)
  const weekIndex = weeks.indexOf(week)
  const completed = week < data.week
  const modelWeek = matches ? currentWeeklyForecasts(data, forecasts, matches) : undefined
  return <section id="matchups" className="league-matchup-week" aria-label="Weekly matchups">
    <div className="outlook-section-head">
      <div><h3>Week {week} matchups</h3><p>{completed ? 'Final scores · Select a game for lineups' : week === data.week ? `${weeklyForecastLabel(modelWeek)} points · Select a game for lineups` : 'Upcoming schedule · Projections appear when available'}</p></div>
      <div className="outlook-week-controls">
        <button type="button" className="button" aria-label="Previous matchup week" disabled={weekIndex <= 0} onClick={() => setWeek(weeks[weekIndex - 1])}>←</button>
        <select aria-label="Matchup week" value={week} onChange={e => setWeek(Number(e.target.value))}>
          {!weeks.includes(week) && <option value={week}>Week {week}</option>}
          {weeks.map(w => <option key={w} value={w}>Week {w}{w === data.week ? ' · Current' : ''}</option>)}
        </select>
        <button type="button" className="button" aria-label="Next matchup week" disabled={weekIndex < 0 || weekIndex >= weeks.length - 1} onClick={() => setWeek(weeks[weekIndex + 1])}>→</button>
        {week !== data.week && <button type="button" className="button" onClick={() => setWeek(data.week)}>Current week</button>}
      </div>
    </div>
    <QueryError query={query} label="Matchups unavailable" />
    {query.isPending && <p role="status">Loading Week {week} matchups…</p>}
    {matches && <div className="outlook-matchups">{matches.matchups.map(g => {
      const model = modelWeek?.matchups.find(m => m.home.team_id === g.home.team_id && m.away.team_id === g.away.team_id)
      const points = [g.home, g.away].map(s => completed ? s.score : [model?.home, model?.away].find(m => m?.team_id === s.team_id)?.model_projection ?? null)
      const complete = points.every((p): p is number => typeof p === 'number' && Number.isFinite(p))
      const margin = complete ? points[0] - points[1] : null
      const winner = margin == null || margin === 0 ? null : margin > 0 ? g.home : g.away
      const edge = margin != null && Math.abs(margin) < .1 && margin !== 0 ? '<0.1' : fixed(Math.abs(margin ?? 0))
      const result = margin == null ? completed ? 'Scores unavailable' : 'No projection yet' : margin === 0 ? completed ? 'Tie' : 'Even projection' : `${completed ? 'Won' : 'Favored'} by ${edge}`
      return <Link key={`${g.home.team_id}-${g.away.team_id}`} className={`outlook-matchup${g.involves_me ? ' mine-matchup' : ''}`} to={matchupPath(week, g.home.team_id, g.away.team_id)} aria-label={`${g.home.team_name} vs ${g.away.team_name}: View matchup`}>
        {[g.home, g.away].map((side, index) => <div key={side.team_id} className={winner?.team_id === side.team_id ? 'projected-favorite' : ''}>
          <span className="matchup-team-name" title={side.team_name}>{side.team_name}{side.team_id === data.my_team_id ? ' · You' : ''}</span>
          <span className="matchup-team-score">{winner?.team_id === side.team_id && <span className="matchup-winner-mark" aria-label={completed ? 'Winner' : 'Projected winner'}>✓</span>}<strong>{fixed(points[index])}</strong></span>
        </div>)}
        <p aria-label={winner ? `${completed ? 'Winner' : 'Projected winner'}: ${winner.team_name}. ${result}` : result}><span>{completed ? 'Final' : week === data.week ? 'Projected' : 'Scheduled'}</span><span>{result}</span><span aria-hidden="true">→</span></p>
      </Link>
    })}</div>}
    {matches && !matches.matchups.length && <p>No matchups captured for Week {week}.</p>}
    {matches?.stale && <small className="faint">Saved {new Date(matches.captured_at).toLocaleString()} · Refresh league for updated lineups.</small>}
  </section>
}
