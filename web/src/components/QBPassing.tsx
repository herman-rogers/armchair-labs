import { Link } from 'react-router'
import { useQuery } from '@tanstack/react-query'
import { passingQuery, passingEvidenceQuery } from '../api/queries'
import type { PassingForecast, PassingEvaluation } from '../api/qbPassing'
import { useDataRelease } from '../dataRelease'
import { intelligencePath, playerPath, useUrlFlag, useUrlPage, useUrlState } from '../navigation'
import { DataTable } from './DataTable'
import { QBVariations } from './QBVariations'
import { Pager, QueryError } from './Controls'
import { grouped } from '../format'

const modelNames: Record<string, string> = {
  reference: 'Adaptive reference policy', prior_season: 'Prior-season reference',
  current_pace: 'Current pace', career_reference: 'Career reference', adaptive_ridge: 'Career + current regression',
  direct_boost: 'Direct boosting · research only', conditional_boost: 'Conditional boosting · research only',
  challenger_policy: 'Selected challenger · research only',
  production_policy: 'Approved production policy',
}
const range = (r: PassingForecast) => r.constraint_reason ? 'Reported absence' : r.retired_known ? 'Known retirement' : r.lower == null || r.upper == null ? 'Not calibrated' : `${grouped(r.lower)}–${grouped(r.upper)}`

function PassingDetails({ row: r }: { row: PassingForecast }) {
  return <div className="player-details">
    <p><strong>Production opportunity:</strong> {r.current_attempts} attempts and {grouped(r.current_yards)} yards through Week {r.through_week}. The forecast covers {r.scheduled_games} scheduled games, through Week {r.end_week}. {r.schedule_known ? '' : 'Team schedule is unknown; a league-week fallback is used.'}</p>
    <p><strong>Evidence about passing production:</strong> {grouped(r.observed_career_ypa, 2)} observed career yards/attempt across {grouped(r.evidence_attempts)} attempts, including the current season. Estimated future efficiency: <strong>{grouped(r.execution_ypa, 2)} yards/attempt</strong>. {r.rate_evidence_status === 'retrospectively_validated' ? `The approved efficiency policy selected ${r.rate_recipe?.replaceAll('_', ' ')} using earlier-year tests.` : 'The career reference weights seasons by recency and shrinks sparse samples toward the earlier league average.'} A zero-attempt absence adds no evidence about yards/attempt.</p>
    <p>Previous season: {grouped(r.prior_yards)} passing yards on {grouped(r.prior_attempts)} attempts. Career evidence remains available after a short season; those missing opportunities still count in the season-total evaluation.</p>
    <p>Production method: {modelNames[r.production_recipe ?? r.reference_recipe] ?? (r.production_recipe ?? r.reference_recipe).replaceAll('_', ' ')}. Method selection used only earlier historical results. The production forecast and per-attempt evidence have separate approval tests; neither is a medical assessment.</p>
    {r.constraint_reason && <p className="notice"><strong>Availability adjustment:</strong> {r.constraint_reason} Known {r.news_known_on}. Forecast before this news: {grouped(r.unconstrained_prediction)} yards. The zero estimate assumes this reported absence; it is not a calibrated zero-width prediction range. {r.constraint_source && <a href={r.constraint_source} target="_blank" rel="noreferrer">Read the report</a>}</p>}
    {r.retired_known && <p>Known retirement constrains future opportunity to zero; career passing evidence is preserved. {r.roster_source && <a href={r.roster_source} target="_blank" rel="noreferrer">Dated source</a>}</p>}
    {r.roster_state === 'off' && !r.retired_known && <p>Dated roster evidence places this player off the roster. The displayed team is the last recorded team and provides a schedule assumption. {r.roster_source && <a href={r.roster_source} target="_blank" rel="noreferrer">Roster source</a>}</p>}
    {!r.constraint_reason && !r.retired_known && <p>Medical availability: {r.medical_status}. Missing injury information does not establish health. Recent passing workload is an observation, not confirmation of a starting assignment.</p>}
    <p>The displayed range uses earlier forecast errors and targets 80% coverage. It includes uncertainty about future opportunity and performance. Actual historical coverage is shown below; this is not a guaranteed bound.</p>
  </div>
}

function EvaluationDetails({ row: r }: { row: PassingEvaluation }) {
  return <div className="player-details">
    <p>{r.n.toLocaleString()} player/cutoff/horizon forecasts across {r.years} seasons. Each season receives equal evaluation weight. Repeated forecasts for the same player are not independent observations.</p>
    {r.model !== 'reference' && <p>Mean squared error reduction against the adaptive reference policy: {grouped(r.mse_gain)} squared yards; exploratory 95% season-bootstrap interval {grouped(r.ci_low)} to {grouped(r.ci_high)}.</p>}
    {['reference', 'production_policy'].includes(r.model) && <p>80% forecast range: {grouped((r.interval_coverage ?? 0) * 100, 1)}% historical coverage across {grouped(r.interval_n)} forecasts with earlier calibration data; average width {grouped(r.interval_width)} yards.</p>}
    <DataTable rows={r.groups} columns={[
      { key: 'group', label: 'Group known at cutoff', title: 'History sample and current workload are measured before the future outcome.', align: 'left', render: v => v.group.replaceAll('_', ' ') },
      { key: 'n', label: 'Forecasts', title: 'Matched forecasts.' },
      { key: 'mae', label: 'MAE · yards', title: 'Equal-season mean absolute error.', render: v => grouped(v.mae, 1) },
      { key: 'rmse', label: 'RMSE · yards', title: 'Square root of equal-season mean squared error.', render: v => grouped(v.rmse, 1) },
      ...(r.model === 'reference' ? [{ key: 'interval_coverage', label: 'Range coverage', title: 'Measured reference interval coverage within this cutoff-defined group.', render: (v: PassingEvaluation) => v.interval_coverage == null ? '—' : `${grouped(v.interval_coverage * 100, 1)}%` }] : []),
    ]} defaultSort="group" rowKey={v => v.group} />
  </div>
}

/** `/intelligence/qb-passing` */
export function QBPassing() {
  const { token } = useDataRelease()
  const [horizon, setHorizon] = useUrlState('horizon', 'rest_of_season', { resets: ['page'] })
  const [search, setSearch] = useUrlState('q', '', { replace: true, resets: ['page'] })
  const [window, setWindow] = useUrlState('evidence_window', 'all_history')
  const [origin, setOrigin] = useUrlState('timing', 'weekly')
  const [research, setResearch] = useUrlFlag('challengers')
  // Forecasts are paginated by the API, so sorting reorders the current page only.
  const page = useUrlPage(100)
  const { offset } = page
  const params = new URLSearchParams({ horizon, search, offset: String(offset), limit: '100' })
  const query = useQuery(passingQuery(params, token))
  const evidenceParams = new URLSearchParams({ horizon, window, origin, scope: research ? 'research' : 'analysis' })
  const evidence = useQuery(passingEvidenceQuery(evidenceParams, token))
  return <section aria-labelledby="qb-passing-heading">
    <h3 id="qb-passing-heading">QB passing: opportunity and production</h3>
    <p>Future yards depend on opportunities to throw and production per attempt. Career evidence remains available when an injury or a backup role limits a season’s sample.</p>
    <div className="analysis-controls">
      <label>Passing forecast horizon<select value={horizon} onChange={e => setHorizon(e.target.value)}>
        <option value="rest_of_season">Remaining regular season</option><option value="next4">Next four scheduled team games</option><option value="next_game">Next scheduled team game</option>
      </select></label>
      <label>Find a quarterback<input type="search" value={search} onChange={e => setSearch(e.target.value)} /></label>
    </div>
    <QueryError query={query} label="QB passing forecasts unavailable" />
    {query.isLoading && <p role="status">Loading QB passing forecasts…</p>}
    {query.data && <>
      <p><strong>{query.data.report.production_approved_horizons?.includes(horizon) ? 'Retrospectively validated production forecasts.' : 'Reference forecasts.'}</strong> {query.data.report.season} production through Week {query.data.report.through_week}; forecast issued {new Date(query.data.report.generated_at).toLocaleString()}. Dated news is shown separately for affected players. Fantasy rankings use passing improvements only when their separate league-point tests pass.</p>
      <DataTable rows={query.data.forecasts} columns={[
        { key: 'player_display_name', label: 'Quarterback', title: 'Includes backups and candidates with no current passing production.', align: 'left' },
        { key: 'team', label: 'Recorded team', title: 'Current or last recorded team at the cutoff; dated roster status is shown in details.', align: 'left' },
        { key: 'prediction', label: 'Expected future yards', title: 'Unconditional yardage over the selected future horizon. Current yards are separate.', render: r => grouped(r.prediction) },
        { key: 'range', label: '80% forecast range', title: 'Calibrated on earlier forecast residuals; inspect measured coverage below.', value: r => r.upper, render: r => range(r) },
        { key: 'current_yards', label: 'Yards so far', title: 'Observed this season; not included in future yards.', render: r => grouped(r.current_yards) },
        { key: 'execution_ypa', label: 'Expected yards/attempt', title: 'Approved efficiency policy where available, otherwise the career reference. Independent of health and starting-role estimates.', render: r => grouped(r.execution_ypa, 2) },
        { key: 'evidence_attempts', label: 'History attempts', title: 'Observed career passing attempts, including current season.', render: r => grouped(r.evidence_attempts) },
      ]} defaultSort="prediction" sortParam="sort" rowKey={r => r.player_id} rowLabel={r => r.player_display_name} profileHref={r => playerPath(r.player_id)} renderDetails={r => <PassingDetails row={r} />} />
      <Pager page={page} total={query.data.total} noun="QBs" summary={`${query.data.total} matching QBs`} />
    </>}
    <h4>Passing forecast evidence</h4>
    <p>All earlier completed seasons are retained for training. Absences and zero-yard outcomes remain in total-production errors. Weekly updates are evaluated on future yards only; they do not rewrite preseason forecasts.</p>
    <div className="analysis-controls">
      <label>Passing evaluation window<select value={window} onChange={e => setWindow(e.target.value)}><option value="all_history">Full history · 2007–2025</option><option value="modern">Modern · 2019–2025</option></select></label>
      <label>Forecast timing<select value={origin} onChange={e => setOrigin(e.target.value)}><option value="weekly">Updates after each completed week</option><option value="preseason">Preseason</option></select></label>
      <label><input type="checkbox" checked={research} onChange={e => setResearch(e.target.checked)} /> Inspect research challengers</label>
    </div>
    {research && <div className="archive-banner"><strong>Original boosting experiment</strong><p>Individual challengers remain research-only. The newer selected production policy is approved only where indicated; every variation and decision is available in QB experiments.</p></div>}
    <QueryError query={evidence} label="Passing evidence unavailable" />
    {evidence.data && <>
      <DataTable rows={evidence.data.comparisons} columns={[
        { key: 'model', label: 'Method', title: 'Current reference policy selects a simple method using earlier out-of-fold results.', align: 'left', render: r => modelNames[r.model] ?? r.model },
        { key: 'mae', label: 'MAE · future yards', title: 'Average absolute miss over the selected horizon.', render: r => grouped(r.mae, 1) },
        { key: 'rmse', label: 'RMSE · future yards', title: 'Primary mean-forecast score expressed in yards; emphasizes large errors.', render: r => grouped(r.rmse, 1) },
        { key: 'n', label: 'Forecasts', title: 'Player/cutoff/horizon observations; not independent players.', render: r => grouped(r.n) },
      ]} defaultSort="model" rowKey={r => r.model} renderDetails={r => <EvaluationDetails row={r} />} />
      {research && evidence.data.decisions.map((d, i) => <p key={i}>{d.reason}.</p>)}
      {research && <><h4>Opportunity and execution are tested separately</h4>
        <p>A substantial-workload game means at least 15 pass attempts. Its forecast frequency is not a probability of medical health or an official starting assignment. Execution errors below use only future outcomes with attempts; total-yardage errors above retain every candidate.</p>
        <DataTable rows={evidence.data.component_scores.filter(r => r.horizon === horizon)} columns={[
          { key: 'primary_fraction_mse', label: 'Workload fraction MSE', title: 'Squared error of the predicted fraction of horizon games with 15+ attempts. For one game this is a Brier score.', render: r => grouped(r.primary_fraction_mse, 3) },
          { key: 'execution_ypa_mae', label: 'YPA MAE', title: 'Mean absolute error of the career-informed per-attempt estimate on defined future rates.', render: r => grouped(r.execution_ypa_mae, 2) },
          { key: 'execution_attempt_weighted_mae', label: 'Attempt-weighted YPA MAE', title: 'The same rate error weighted by actual future attempts. This is an evaluation denominator, not a predictor.', render: r => grouped(r.execution_attempt_weighted_mae, 2) },
          { key: 'execution_observed_n', label: 'Defined rate outcomes', title: 'Player/cutoff/horizon observations with at least one future pass attempt.' },
        ]} defaultSort="horizon" rowKey={r => r.horizon} />
      </>}
      <details><summary>Coverage and limitations</summary><ul>{evidence.data.limitations.map(l => <li key={l}>{l}</li>)}</ul></details>
    </>}
    {query.data?.report.variation_research && <QBVariations paramPrefix="exp_" />}
  </section>
}

export function PlayerPassing({ playerId }: { playerId: string }) {
  const { token } = useDataRelease()
  const query = useQuery(passingQuery(new URLSearchParams({ player_id: playerId, horizon: 'rest_of_season' }), token))
  if (query.isError) return <p className="notice">QB passing forecast unavailable: {query.error.message}</p>
  const r = query.data?.forecasts[0]
  if (!r) return null
  return <section><h4>QB passing · remaining regular season</h4>
    <p><strong>{grouped(r.prediction)} expected future passing yards</strong> · {range(r)} forecast range. Production through Week {r.through_week}; forecast issued {new Date(r.issued_at).toLocaleString()}.</p>
    <PassingDetails row={r} />
    <Link to={intelligencePath('qb-passing')}>Compare quarterbacks and review passing evidence</Link>
  </section>
}
