import { ModelWeights } from './ModelWeights'
import { IntervalCalibration } from './IntervalCalibration'
import { ResearchExplorer } from './ResearchExplorer'
import { SignalEvidenceTable } from './SignalEvidenceTable'
import { useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import type { MetricAssessment, MetricBacktestResult, ResearchDataset } from '../api/types'
import { rankerLabel } from '../metricPresentation'
import { researchModelLabel } from '../researchModels'
import { useUrlFlag, useUrlState } from '../navigation'
import { PositionOptions } from './Controls'
import { positionOrder } from '../positions'
import { metricReportQuery } from '../api/queries/archive'
import { EVIDENCE_WINDOWS } from '../researchWindows'
import { percent, pointLift, signed } from '../format'

const ASSESSMENTS: Array<{ id: MetricAssessment | 'all'; label: string }> = [
  { id: 'all', label: 'All evidence' },
  { id: 'strong', label: 'Strong' },
  { id: 'useful', label: 'Useful' },
  { id: 'harmful', label: 'Harmful' },
  { id: 'redundant', label: 'Redundant' },
  { id: 'weak', label: 'Weak' },
  { id: 'mixed', label: 'Mixed' },
  { id: 'insufficient', label: 'Insufficient' },
]

function evidenceScore(row: MetricBacktestResult) {
  return Math.abs(row.partial_spearman ?? row.spearman ?? 0)
}

function seasonRange(seasons: number[]) {
  if (seasons.length === 0) return 'none'
  if (seasons.length === 1) return String(seasons[0])
  return `${seasons[0]}–${seasons[seasons.length - 1]}`
}

export function MetricReportView({ available, dataset, source }: { available: boolean; dataset?: string; source?: ResearchDataset }) {
  const report = useQuery({ ...metricReportQuery(dataset), enabled: available })
  // Params are prefixed `report_`: this report shares its page with other research panels.
  const [target, setTarget] = useUrlState('report_outcome', 'actual_availability_value')
  const [position, setPosition] = useUrlState('report_position', 'ALL')
  const [group, setGroup] = useUrlState('report_family', 'All groups')
  const [assessmentParam, setAssessmentParam] = useUrlState('report_assessment', 'all')
  const assessment = assessmentParam as MetricAssessment | 'all'
  const setAssessment = (value: MetricAssessment | 'all') => setAssessmentParam(value)
  const [requestedWindow, setWindow] = useUrlState('report_window', 'modern')
  const [showAllWindows, setShowAllWindows] = useUrlFlag('report_all_windows')
  const [rankingTarget, setRankingTarget] = useUrlState('report_ranking', 'actual_season_points')
  const [signalSearch, setSignalSearch] = useUrlState('report_q', '', { replace: true })
  const visibleWindows = useMemo(() => (report.data?.configuration.analysis_windows ?? [])
    .filter(entry => showAllWindows || !EVIDENCE_WINDOWS[entry.key]?.advanced), [report.data, showAllWindows])
  const selectedWindow = useMemo(() => visibleWindows.find(entry => entry.key === requestedWindow)
    ?? visibleWindows.find(entry => entry.key === 'modern') ?? visibleWindows[0], [visibleWindows, requestedWindow])
  const window = selectedWindow?.key ?? ''

  const groups = useMemo(
    () => ['All groups', ...new Set(report.data?.metrics.map((metric) => metric.group) ?? [])],
    [report.data],
  )
  const evidenceRows = useMemo(
    () =>
      (report.data?.results ?? [])
        .filter((row) => row.window === window)
        .filter((row) => row.target === target)
        .filter((row) => row.position === position)
        .filter((row) => group === 'All groups' || row.group === group),
    [group, position, report.data, target, window],
  )
  const rows = useMemo(
    () =>
      evidenceRows
        .filter((row) => assessment === 'all' || row.assessment === assessment)
        .filter(row => row.label.toLowerCase().includes(signalSearch.trim().toLowerCase()))
        .sort((left, right) => evidenceScore(right) - evidenceScore(left)),
    [assessment, evidenceRows, signalSearch],
  )
  const modelRows = useMemo(
    () =>
      (report.data?.model_results ?? [])
        .filter((row) => row.window === window)
        .filter((row) => row.position === position),
    [position, report.data, window],
  )
  const rankingRows = useMemo(
    () =>
      (report.data?.ranking_results ?? [])
        .filter((row) => row.window === window && row.target === rankingTarget)
        .sort(
          (left, right) =>
            positionOrder(left.position) - positionOrder(right.position) ||
            Number(left.role === 'candidate') - Number(right.role === 'candidate') ||
            right.hit_rate - left.hit_rate,
        ),
    [rankingTarget, report.data, window],
  )

  if (!available) {
    return (
      <div className="notice">
        <h2>No metric report built yet</h2>
        Run <code>just metric-report</code>, then reload this page.
      </div>
    )
  }
  if (report.isLoading) return <div className="notice">Loading metric report…</div>
  if (report.isError || !report.data) {
    return <div className="notice">Could not load the metric report: {report.error?.message}</div>
  }

  const data = report.data
  const inWindow = (season: number) => !selectedWindow || (season >= selectedWindow.start && season <= selectedWindow.end)
  return (
    <section className="metric-report">
      <div className="report-hero">
        <div>
          <span className="eyebrow">Saved historical evidence</span>
          <h2>Research & evidence</h2>
          <p>
            Rolling forecasts for {seasonRange(data.data_summary.completed_forecasts)} using{' '}
            {data.configuration.history_seasons} trailing seasons.{' '}
            {data.data_summary.pending_forecasts.length > 0 &&
              `${data.data_summary.pending_forecasts.join(', ')} is saved as a pending prospective forecast.`}
          </p>
        </div>
        <span className="count">
          Updated {new Date(data.generated_at).toLocaleString()} · {data.data_summary.forecast_rows} rows
        </span>
      </div>

      <div className="controls report-controls report-window-control">
        <label className="research-wide-control">
          Evidence window
          <select aria-label="Evidence window" value={window} onChange={(event) => setWindow(event.target.value)}>
            {!visibleWindows.length && <option value="">No standard windows · choose advanced</option>}
            {visibleWindows.map((entry) => (
              <option key={entry.key} value={entry.key}>
                {EVIDENCE_WINDOWS[entry.key]?.label ?? entry.label} · {entry.start}–{entry.end}
              </option>
            ))}
          </select>
        </label>
        <label>Window list
          <select aria-label="Evidence window list" value={showAllWindows ? 'all' : 'standard'}
            onChange={event => setShowAllWindows(event.target.value === 'all')}>
            <option value="standard">Primary comparisons</option>
            <option value="all">All windows (advanced)</option>
          </select>
        </label>
      </div>
      <p className="legend">These are overlapping date windows of the selected dataset—not different dataset versions or quality grades. Changing a window filters saved evaluations; it does not rebuild forecasts or retrain the models.</p>
      {selectedWindow && <p className="legend">{selectedWindow.start}–{selectedWindow.end}: {EVIDENCE_WINDOWS[selectedWindow.key]?.purpose}
        {' '}{data.data_summary.completed_forecasts.filter(inWindow).length} completed forecast seasons in this window; individual models may cover fewer.</p>}
      <details className="analysis-details">
        <summary>Dataset quality & what to keep</summary>
        <p>{source?.accepted
          ? `${source.checks_passed ?? 'Recorded'} integrity checks passed for ${source.id}. Accepted for research, not promoted to production.`
          : 'This report does not have a verified repair-acceptance record here. Use legacy evidence as a reference, not proof that the repaired models work.'}
          {' '}An integrity pass checks recorded defects and provenance; it does not certify complete historical context, publication-time data availability, or predictive value.</p>
        <div className="table-wrap"><table>
          <thead><tr><th className="left">Keep / use</th><th className="left">Purpose and limits</th></tr></thead>
          <tbody>
            <tr><td className="left">Accepted repaired history · primary research</td><td className="left">Prefer the audited rebuild over pre-repair inputs. Preserve its source snapshots, manifests, acceptance record, and dependent studies.</td></tr>
            <tr><td className="left">V1 and frozen forecasts · protected references</td><td className="left">Keep to reproduce the current draft and original forecasts. Older does not make these disposable.</td></tr>
            <tr><td className="left">Legacy and superseded runs · archive</td><td className="left">Use for regression checks and provenance, not current evidence. Archive failed runs separately; review dependencies before removing any files.</td></tr>
            <tr><td className="left">College → NFL and in-season outlook · complementary</td><td className="left">Different populations and forecast horizons, not replacements for historical season forecasts. The separate college study is not incorporated into saved Next-gen rankings.</td></tr>
          </tbody>
        </table></div>
        <p>Judge quality by identity and scoring checks, dated source evidence, observed versus inferred context, missing-data coverage, and held-out performance against a simple baseline. Newer or larger alone is not better.</p>
        <div className="table-wrap"><table>
          <thead><tr><th className="left">Evaluation window</th><th className="left">Why retain it</th></tr></thead>
          <tbody>{data.configuration.analysis_windows.map(entry => <tr key={entry.key}>
            <td className="left">{EVIDENCE_WINDOWS[entry.key]?.label ?? entry.label} · {entry.start}–{entry.end}</td>
            <td className="left">{EVIDENCE_WINDOWS[entry.key]?.purpose ?? 'Saved evaluation slice; inspect source coverage before comparing.'}</td>
          </tr>)}</tbody>
        </table></div>
        <p>Advanced options only declutter the interface. No historical data, report, or model output is deleted.</p>
      </details>

      <nav className="research-nav" aria-label="Research sections">
        <a href="#research-comparisons">Model comparisons</a><a href="#research-coverage">Coverage</a><a href="#research-weights">Model weights</a><a href="#research-signals">Signal evidence</a><a href="#research-calibration">Forecast error</a>
      </nav>
      <div id="research-comparisons"><ResearchExplorer report={data} window={window} /></div>
      <details id="research-coverage" className="analysis-details">
        <summary>Production coverage and comparison populations</summary>
        <p>{data.evaluation_contract ?? 'Legacy evaluation: player pools differ. Regenerate evidence before using these results for model selection.'}</p>
        <p>Automatic production policy, including market fallbacks; historical manual overrides are excluded.</p>
        <div className="table-wrap"><table>
          <thead><tr><th>Season</th><th>Complete-pool hit rate</th><th>Coverage</th><th>Market fallbacks</th></tr></thead>
          <tbody>{data.deployment_results?.filter(row => inWindow(row.forecast_season)).map((row) => <tr key={row.forecast_season}>
            <td>{row.forecast_season}</td><td>{(100 * row.hit_rate).toFixed(1)}%</td>
            <td>{(100 * row.coverage).toFixed(1)}%</td><td>{row.market_fallback_n}</td>
          </tr>)}</tbody>
        </table></div>
        {!data.deployment_results?.filter(row => inWindow(row.forecast_season)).length && <p>No deployment coverage was saved for this window.</p>}
        <p>Ranking quality against overall ECR on common players only; coverage is reported against the retained draft pool.</p>
        <div className="table-wrap"><table>
          <thead><tr><th>Season</th><th>Common players</th><th>Coverage</th><th>Hit-rate lift vs ECR</th></tr></thead>
          <tbody>{data.common_pool_results?.filter((row) => row.ranker === 'fitted_season_points' && inWindow(row.forecast_season)).sort((a, b) => a.forecast_season - b.forecast_season).map((row) => <tr key={row.forecast_season}>
            <td>{row.forecast_season}</td><td>{row.common_n}</td><td>{(100 * row.coverage).toFixed(1)}%</td>
            <td>{(100 * row.hit_rate_lift).toFixed(1)} pp</td>
          </tr>)}</tbody>
        </table></div>
        {!data.common_pool_results?.filter(row => inWindow(row.forecast_season)).length && <p>No common-player comparison was saved for this window.</p>}
      </details>
      <details className="analysis-details"><summary>Full ranking results by position</summary>
      <div className="ranking-verdict">
        <div className="ranking-heading">
          <div>
            <h3>Do projection rankers beat naive history?</h3>
            <p className="legend tight">
              {data.evaluation_contract
                ? 'Missing forecasts count as coverage failures against a fixed outcome universe.'
                : 'Legacy report: candidate player pools differ, so the reported verdict is not a fair model-selection test.'}
              {' '}Reported “beats” requires positive hit-rate lift and more fold wins than losses against the best available baseline. These are diagnostics, not automatic promotion decisions.
            </p>
          </div>
          <label>
            Ranking outcome
            <select aria-label="Ranking outcome" value={rankingTarget} onChange={(event) => setRankingTarget(event.target.value)}>
              {data.configuration.ranking.targets.map((entry) => (
                <option key={entry} value={entry}>{rankerLabel(entry)}</option>
              ))}
            </select>
          </label>
        </div>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Pos</th>
                <th className="left">Ranker</th>
                <th>Role</th>
                <th>K</th>
                <th>Hit @K</th>
                <th>NDCG @K</th>
                <th>Pool corr.</th>
                <th>{rankingTarget === 'actual_ppg' ? 'Top-K PPG' : 'Top-K points'}</th>
                <th>Hit lift</th>
                <th title="Fold wins, ties, and losses versus the best baseline. Equal hit rates are broken by top-K actual production.">W–T–L</th>
                <th>Verdict</th>
              </tr>
            </thead>
            <tbody>
              {rankingRows.map((row) => (
                <tr key={`${row.window}-${row.target}-${row.position}-${row.ranker}`}>
                  <td className="strong">{row.position}</td>
                  <td className="left name">{researchModelLabel(row.ranker)}</td>
                  <td>
                    <span
                      className={`ranking-role ${row.role}`}
                      title={row.best_baseline ? `Compared with ${rankerLabel(row.best_baseline)}` : undefined}
                    >
                      {row.role}
                    </span>
                  </td>
                  <td>{row.k}</td>
                  <td className="strong">{percent(row.hit_rate)}</td>
                  <td>{percent(row.ndcg)}</td>
                  <td>{signed(row.pool_spearman, 3)}</td>
                  <td title={`Ideal: ${row.ideal_top_k_actual_mean.toFixed(1)}`}>
                    {row.top_k_actual_mean.toFixed(1)}
                  </td>
                  <td title={row.best_baseline ? `Versus ${rankerLabel(row.best_baseline)}` : undefined}>
                    {pointLift(row.hit_rate_lift)}
                  </td>
                  <td>
                    {row.folds_won == null
                      ? '—'
                      : `${row.folds_won}–${row.folds_tied}–${row.folds_lost}`}
                  </td>
                  <td>
                    {row.beats_baseline == null ? (
                      '—'
                    ) : (
                      <span className={`ranking-verdict-badge ${row.beats_baseline ? 'passes' : 'loses'}`}>
                        {row.beats_baseline ? 'beats' : 'does not beat'}
                      </span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      </details>
      {!!data.fitted_model_summary?.length && <ModelWeights entries={data.fitted_model_summary} />}

      <section id="research-signals" className="analysis-section">
      <span className="eyebrow">Signal research</span><h3>Which inputs add information?</h3>
      <div className="report-cards">
        {(['strong', 'useful', 'harmful', 'redundant', 'weak'] as MetricAssessment[]).map((key) => (
          <button
            type="button"
            className={`report-card ${key}`}
            key={key}
            aria-pressed={assessment === key}
            onClick={() => setAssessment(assessment === key ? 'all' : key)}
          >
            <span>{evidenceRows.filter((row) => row.assessment === key).length}</span>
            {key}
          </button>
        ))}
      </div>

      <div className="controls report-controls">
        <label>
          Outcome
          <select aria-label="Signal outcome" value={target} onChange={(event) => setTarget(event.target.value)}>
            {data.targets.map((entry) => (
              <option key={entry.key} value={entry.key}>{entry.label}</option>
            ))}
          </select>
        </label>
        <label>
          Position
          <select aria-label="Signal position" value={position} onChange={(event) => setPosition(event.target.value)}>
            <PositionOptions />
          </select>
        </label>
        <label>
          Metric family
          <select aria-label="Metric family" value={group} onChange={(event) => setGroup(event.target.value)}>
            {groups.map((entry) => <option key={entry}>{entry}</option>)}
          </select>
        </label>
        <label>
          Assessment
          <select
            value={assessment}
            onChange={(event) => setAssessment(event.target.value as MetricAssessment | 'all')}
          >
            {ASSESSMENTS.map((entry) => (
              <option key={entry.id} value={entry.id}>{entry.label}</option>
            ))}
          </select>
        </label>
      </div>

      <input className="search" aria-label="Search signal evidence" placeholder="Search signals…" value={signalSearch} onChange={event => setSignalSearch(event.target.value)} />
      <p className="legend tight">
        Spearman measures next-season rank association. “Partial vs prior” measures what remains
        after controlling for the historical PPG prior. Consistency is the share of forecast years
        agreeing with the pooled direction. “Harmful” means a repeatable signal points opposite
        the metric's configured expected direction.
      </p>

      <SignalEvidenceTable rows={rows} catalog={data.metrics} />
      </section>

      <IntervalCalibration rows={data.uncertainty_calibration} position={position} start={selectedWindow?.start} end={selectedWindow?.end} />
      <h3 id="research-calibration" className="section-head">Forecast error by outcome</h3>
      <p className="legend tight">
        Error measures apply only where a projection and outcome share the same units. Positive
        bias means the model over-projected the outcome; rank correlation measures ordering.
      </p>
      <p className="legend tight">Scope: {position} · selected evidence window. Smaller MAE and RMSE indicate smaller errors.</p>
      <div className="table-wrap calibration-table">
        <table>
          <thead>
            <tr>
              <th className="left">Projection</th>
              <th className="left">Outcome</th>
              <th>MAE</th>
              <th>RMSE</th>
              <th>Bias</th>
              <th>Rank corr.</th>
              <th>N</th>
            </tr>
          </thead>
          <tbody>
            {modelRows.map((row) => (
              <tr key={`${row.metric}-${row.target}-${row.position}`}>
                <td className="left name">{row.label}</td>
                <td className="left dim">{row.target_label}</td>
                <td>{row.mae.toFixed(2)}</td>
                <td>{row.rmse.toFixed(2)}</td>
                <td>{signed(row.bias, 3)}</td>
                <td>{signed(row.spearman, 3)}</td>
                <td>{row.n}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {!modelRows.length && <p className="notice">No forecast-error results for this position and window. Choose a position in Signal evidence above.</p>}
      <details className="metric-catalog">
        <summary>Metric catalog ({data.metrics.length})</summary>
        <div className="catalog-grid">
          {data.metrics.map((metric) => (
            <article key={metric.key}>
              <div><b>{metric.label}</b> <span className="badge rostered">{metric.group}</span></div>
              <p>{metric.description}</p>
              <span className="faint">
                {metric.available ? `${percent(metric.coverage)} historical coverage` : 'Unavailable in artifact'}
                {metric.available_from_forecast != null &&
                  ` · evaluated from ${metric.available_from_forecast}`}
              </span>
            </article>
          ))}
        </div>
      </details>

      <div className="report-limitations">
        <h3>How to read this</h3>
        <ul>{data.data_summary.limitations.map((item) => <li key={item}>{item}</li>)}</ul>
      </div>
    </section>
  )
}
