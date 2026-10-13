import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router'
import { playerScoringQuery } from '../api/queries'
import { useCampaignChoice } from '../api/queries/hooks'
import { useDataRelease } from '../dataRelease'
import { playerPath, useUrlPage, useUrlState } from '../navigation'
import { fixed } from '../format'
import { QueryError, SelectField } from './Controls'
import { ScoringChart, ScoringLegend } from './ScoringChart'
import { scoringDomain } from '../scoringTrends'

export function PlayerScoringTrends() {
  const choice = useCampaignChoice()
  const { token } = useDataRelease()
  const [position, setPosition] = useUrlState('position', 'ALL', { resets: ['page'] })
  const [search, setSearch] = useUrlState('q', '', { replace: true, resets: ['page'] })
  const [scoring, setScoring] = useUrlState('scoring', 'model')
  const page = useUrlPage(12)
  const params = new URLSearchParams({ campaign: choice.campaignId, position, search, limit: String(page.limit), offset: String(page.offset) })
  const query = useQuery(playerScoringQuery(params, token, choice.campaign?.manifest_sha256))
  const data = query.data
  const players = data?.players.map(p => scoring === 'league' ? { ...p, average: p.league_average, average_games: p.league_average_games, weeks: p.weeks.map(w => ({ ...w, actual: w.league_actual ?? null })) } : p) ?? []
  const domain = scoringDomain(players)
  return <section aria-label="Player scoring trends">
    <div className="analysis-controls">
      <label>Find player<input type="search" value={search} onChange={e => setSearch(e.target.value)} placeholder="Name…" /></label>
      <SelectField label="Position" value={position} onChange={e => setPosition(e.target.value)}>{['ALL', 'QB', 'RB', 'WR', 'TE'].map(p => <option key={p} value={p}>{p === 'ALL' ? 'All positions' : p}</option>)}</SelectField>
      <SelectField label="Actual scoring" value={scoring} onChange={e => setScoring(e.target.value)}><option value="model">Match Opening Drive scoring</option><option value="league">League scoring</option></SelectField>
    </div>
    <QueryError query={query} label="Scoring trends unavailable" />
    {query.isPending && <p role="status">Loading player scoring history…</p>}
    {data && <>
      <div className="scoring-heading"><p>{data.season} · Actuals through Week {data.through_week} · Full {data.last_week}-week NFL season</p><span className="badge">{data.campaign_name}</span></div>
      <ScoringLegend />
      <p className="legend">The dotted average is actual season-to-date scoring per completed scheduled game, including missed games and excluding byes. Shaded weeks are upcoming. Forecasts change only with new model cutoffs or byes.</p>
      {scoring === 'league' && <p className="notice">Actuals use league scoring; Opening Drive predictions exclude league bonuses, fumbles and other scoring differences.</p>}
      {['QB', 'RB', 'WR', 'TE'].map(pos => {
        const group = players.filter(p => p.position === pos)
        return group.length ? <section className="scoring-position" key={pos} aria-label={`${pos} scoring charts`}><h3>{pos}</h3><div className="scoring-grid">{group.map(p => <article className="scoring-card" key={p.player_id}>
          <ScoringChart series={p} description={`${p.team ?? 'Team unknown'} · Season avg ${fixed(p.average)} pts (${p.average_games} games) · Latest pace ${fixed(p.forecast_rate)} pts/game`} throughWeek={data.through_week} lastWeek={data.last_week} domain={domain} />
          <Link className="scoring-profile-link" to={playerPath(p.player_id)}>Player profile →</Link>
        </article>)}</div></section> : null
      })}
      {!players.length && <p className="notice">No players match these filters{page.offset ? '; try the first page' : ''}.</p>}
      <nav className="scoring-pagination" aria-label="Player chart pages"><button className="button" disabled={!page.offset} onClick={() => page.onPage(Math.max(0, page.offset - page.limit))}>Previous</button><span>{data.total ? `${Math.min(page.offset + 1, data.total)}–${Math.min(page.offset + page.limit, data.total)} of ${data.total}` : '0 players'} · Ordered by forecast pace</span><button className="button" disabled={page.offset + page.limit >= data.total} onClick={() => page.onPage(page.offset + page.limit)}>Next</button></nav>
      <details className="outlook-method"><summary>Scoring and forecast history</summary><p>{data.scoring}</p><p>{data.method}</p><p>Historical lines are reconstructed from saved model cutoffs, not verified pregame captures. Missing forecasts or unknown coverage stay blank. The season-average line describes results through Week {data.through_week}; it was not known in earlier weeks.</p><p>Updates follow the weekly observation refresh and saved campaign rescores. This page checks for changes every minute.</p></details>
    </>}
  </section>
}
