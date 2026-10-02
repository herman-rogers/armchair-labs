import { Link } from 'react-router'
import { useQuery } from '@tanstack/react-query'
import { variationsQuery } from '../api/queries'
import { useDataRelease } from '../dataRelease'
import { DataTable } from './DataTable'
import { researchPath, useUrlParams, useUrlState } from '../navigation'

const number = (n: number | null | undefined, digits = 2) => n == null ? '—' : n.toLocaleString(undefined, { maximumFractionDigits: digits })
const label = (name: string) => ({ rate_policy: 'Selected efficiency policy', yards_policy: 'Selected production policy',
  rate_reference: 'Original career YPA reference', yards_reference: 'Original adaptive yardage reference',
  policy: 'Selected fantasy policy', reference: 'Existing fantasy reference' }[name] ?? name.replaceAll('_', ' '))

/** `paramPrefix` namespaces this panel's search params when it is embedded in another page. */
export function QBVariations({ research = false, paramPrefix = '' }: { research?: boolean; paramPrefix?: string }) {
  const { token } = useDataRelease()
  const key = (name: string) => `${paramPrefix}${name}`
  const defaultHorizon = research ? 'next_game' : 'rest_of_season'
  const [target] = useUrlState(key('target'), research ? 'rate' : 'yards')
  const [horizon, setHorizon] = useUrlState(key('horizon'), defaultHorizon)
  const [window, setWindow] = useUrlState(key('window'), 'modern')
  const [origin, setOrigin] = useUrlState(key('timing'), 'weekly')
  const [search, setSearch] = useUrlState(key('q'), '', { replace: true })
  const [, update] = useUrlParams()
  // Fantasy-point tests only cover weekly updates over calendar weeks, so switching to
  // them adjusts timing and horizon in the same navigation.
  const setTarget = (value: string) => {
    const points = value === 'league_points'
    const nextHorizon = points && horizon === 'next_game' ? 'next4' : horizon
    update({ [key('target')]: value === (research ? 'rate' : 'yards') ? null : value,
      ...(points ? { [key('timing')]: null, [key('horizon')]: nextHorizon === defaultHorizon ? null : nextHorizon } : {}) })
  }
  const params = new URLSearchParams({ target, horizon, window, origin, scope: research ? 'research' : 'analysis' })
  const query = useQuery(variationsQuery(params, token))
  const unit = target === 'rate' ? 'yards/attempt' : target === 'yards' ? 'future yards' : 'league points'
  return <section aria-label="QB model variations">
    <h3>QB passing model variations</h3>
    <p>Passing efficiency, throwing opportunity and fantasy value have separate tests. Every model trains on earlier seasons; every recipe is selected using earlier held-out results. Modern is an evaluation slice of the full historical experiment.</p>
    {research ? <p className="archive-banner">Individual alternatives and failed policies are preserved research. Only a selected policy marked approved affects the current analysis. Historical development is not a prospective accuracy claim.</p> :
      <p><Link to={researchPath('qb-experiments')}>Inspect every variation and rejected result in Research</Link></p>}
    <div className="analysis-controls">
      <label>QB experiment outcome<select value={target} onChange={e => setTarget(e.target.value)}>
        <option value="rate">Passing efficiency · yards/attempt</option><option value="yards">Total future passing yards</option><option value="league_points">QB fantasy rankings · league points</option>
      </select></label>
      <label>QB experiment horizon<select value={horizon} onChange={e => setHorizon(e.target.value)}>
        <option value="rest_of_season">Remaining regular season</option>
        <option value="next4">{target === 'league_points' ? 'Next four calendar weeks' : 'Next four scheduled team games'}</option>
        {target !== 'league_points' && <option value="next_game">Next scheduled team game</option>}
      </select></label>
      <label>QB experiment history<select value={window} onChange={e => setWindow(e.target.value)}><option value="modern">Modern · 2019–2025</option><option value="all_history">{target === 'league_points' ? 'Selection policy · 2011–2025' : 'Full evaluation · 2007–2025'}</option></select></label>
      {research && target !== 'league_points' && <label>QB experiment timing<select value={origin} onChange={e => setOrigin(e.target.value)}><option value="weekly">Weekly updates</option><option value="preseason">Preseason · exploratory</option></select></label>}
      {research && <label>Find a variation<input type="search" value={search} onChange={e => setSearch(e.target.value)} /></label>}
    </div>
    {query.isLoading && <p role="status">Loading QB variation evidence…</p>}
    {query.isError && <p role="alert">QB variation evidence unavailable: {query.error.message}</p>}
    {query.data && <>
      <p>{query.data.report.rate_variations} efficiency variations · {query.data.report.yard_variations} yardage variations · {query.data.report.ranking_variations} fantasy combinations. Run {query.data.report.version}.</p>
      {query.data.decision && <p><strong>{origin === 'preseason' ? 'Preseason is exploratory; approval below covers weekly forecasts only.' : query.data.decision.approved ? 'Approved for this outcome and horizon.' : 'Current reference retained.'}</strong> {query.data.decision.reason}. Adjusted q: {number(query.data.decision.q_value, 3)}.</p>}
      <p>{target === 'rate' ? 'RMSE weights defined future YPA errors by pass attempts, then gives each season equal weight. MAE is unweighted within each season. Zero-attempt outcomes remain in the total-yardage tests.' : target === 'yards' ? 'Errors include every candidate, including injuries, backups and zero-yard outcomes. Each season receives equal weight.' : 'These tests compare actual league points and top-10 point capture, using the exact calendar weeks served by rankings. Passing predictions in training were themselves made without that season’s outcomes.'}</p>
      <DataTable rows={query.data.comparisons.filter(r => label(r.model).toLowerCase().includes(search.toLowerCase()))} columns={[
        { key: 'model', label: 'Variation / policy', title: 'The selected policy chooses a recipe using only earlier-year results.', align: 'left', render: r => label(r.model) },
        { key: 'status', label: 'Use in analysis', title: 'Individual candidates do not inherit their selection policy’s approval.', align: 'left' },
        { key: 'mae', label: `MAE · ${unit}`, title: 'Equal-season absolute forecast error.', render: r => number(r.mae) },
        ...(target !== 'league_points' ? [{ key: 'rmse', label: `RMSE · ${unit}`, title: 'Square root of equal-season squared error; rate errors are weighted by attempts.', render: (r: {rmse: number | null}) => number(r.rmse) }] : []),
        { key: 'improvement_pct', label: `${target === 'league_points' ? 'MAE' : 'MSE'} reduction`, title: 'Reduction relative to the original reference on identical outcomes; negative is worse.', render: r => `${number(r.improvement_pct)}%` },
        { key: 'n', label: 'Forecasts', title: 'Correlated player/cutoff/horizon observations.', render: r => number(r.n, 0) },
      ]} defaultSort="improvement_pct" sortParam={key('sort')} rowKey={r => r.model} renderDetails={r => <div className="player-details">
        <p>{r.years} evaluation seasons. {r.model.endsWith('reference') ? 'Reference compared with itself.' : `95% season-bootstrap gain interval: ${number(r.ci_low)} to ${number(r.ci_high)} ${target === 'league_points' ? 'points' : `squared ${unit}`}. Individual variation intervals are exploratory; approval tests the selected policy and corrects eight policies together.`}</p>
        <DataTable rows={r.annual} columns={[
          { key: 'season', label: 'Held-out season', title: 'Only earlier seasons trained and selected this forecast.', initial: 'asc' },
          { key: 'n', label: 'Forecasts', title: 'Matched outcomes.' },
          { key: 'error', label: `MAE · ${unit}`, title: 'Absolute error for this held-out season.', render: a => number(a.mae ?? a.error) },
        ]} defaultSort="season" rowKey={a => a.season} />
      </div>} />
      <details><summary>Research limitations</summary><ul>{query.data.report.limitations.map(l => <li key={l}>{l}</li>)}</ul></details>
    </>}
  </section>
}
