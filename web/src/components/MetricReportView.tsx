import { useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { fetchMetricReport } from '../api/client'
import type { MetricAssessment, MetricBacktestResult } from '../api/types'
import { rankerLabel } from '../metricPresentation'

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

function correlation(value: number | null) {
  if (value == null) return '—'
  return `${value >= 0 ? '+' : ''}${value.toFixed(3)}`
}

function percent(value: number | null) {
  return value == null ? '—' : `${Math.round(value * 100)}%`
}

function percentagePointLift(value: number | null) {
  if (value == null) return '—'
  const points = value * 100
  return `${points >= 0 ? '+' : ''}${points.toFixed(1)} pp`
}

function evidenceScore(row: MetricBacktestResult) {
  return Math.abs(row.partial_spearman ?? row.spearman ?? 0)
}

function seasonRange(seasons: number[]) {
  if (seasons.length === 0) return 'none'
  if (seasons.length === 1) return String(seasons[0])
  return `${seasons[0]}–${seasons[seasons.length - 1]}`
}

export function MetricReportView({ available }: { available: boolean }) {
  const report = useQuery({
    queryKey: ['metric-report'],
    queryFn: fetchMetricReport,
    enabled: available,
  })
  const [target, setTarget] = useState('actual_availability_value')
  const [position, setPosition] = useState('ALL')
  const [group, setGroup] = useState('All groups')
  const [assessment, setAssessment] = useState<MetricAssessment | 'all'>('all')
  const [window, setWindow] = useState('modern')
  const [rankingTarget, setRankingTarget] = useState('actual_season_points')
  const [fittedModel, setFittedModel] = useState('fitted_ppg')

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
        .sort((left, right) => evidenceScore(right) - evidenceScore(left)),
    [assessment, evidenceRows],
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
            ['QB', 'RB', 'WR', 'TE'].indexOf(left.position) -
              ['QB', 'RB', 'WR', 'TE'].indexOf(right.position) ||
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
  return (
    <section className="metric-report">
      <div className="report-hero">
        <div>
          <h2>{data.title}</h2>
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
        <label>
          Evidence window
          <select value={window} onChange={(event) => setWindow(event.target.value)}>
            {data.configuration.analysis_windows.map((entry) => (
              <option key={entry.key} value={entry.key}>
                {entry.label} · {entry.start}–{entry.end}
              </option>
            ))}
          </select>
        </label>
      </div>

      <details>
        <summary>Production coverage and comparison populations</summary>
        <p>{data.evaluation_contract ?? 'Legacy evaluation: player pools differ. Regenerate evidence before using these results for model selection.'}</p>
        <p>Automatic production policy, including market fallbacks; historical manual overrides are excluded.</p>
        <table>
          <thead><tr><th>Season</th><th>Complete-pool hit rate</th><th>Coverage</th><th>Market fallbacks</th></tr></thead>
          <tbody>{data.deployment_results?.map((row) => <tr key={row.forecast_season}>
            <td>{row.forecast_season}</td><td>{(100 * row.hit_rate).toFixed(1)}%</td>
            <td>{(100 * row.coverage).toFixed(1)}%</td><td>{row.market_fallback_n}</td>
          </tr>)}</tbody>
        </table>
        <p>Ranking quality against overall ECR on common players only; coverage is reported against the retained draft pool.</p>
        <table>
          <thead><tr><th>Season</th><th>Common players</th><th>Coverage</th><th>Hit-rate lift vs ECR</th></tr></thead>
          <tbody>{data.common_pool_results?.filter((row) => row.ranker === 'fitted_season_points').sort((a, b) => a.forecast_season - b.forecast_season).map((row) => <tr key={row.forecast_season}>
            <td>{row.forecast_season}</td><td>{row.common_n}</td><td>{(100 * row.coverage).toFixed(1)}%</td>
            <td>{(100 * row.hit_rate_lift).toFixed(1)} pp</td>
          </tr>)}</tbody>
        </table>
      </details>
      <div className="ranking-verdict">
        <div className="ranking-heading">
          <div>
            <h3>Do projection rankers beat naive history?</h3>
            <p className="legend tight">
              Complete-pool results keep the actual outcome universe fixed and count missing forecasts as coverage failures. “Beats” requires a
              positive pooled hit-rate lift and more head-to-head fold wins than losses
              against the best baseline available on the same seasons. This is diagnostic evidence, not a model promotion. Pairwise common-player results, market-fallback deployment coverage, and interval calibration are retained in the report artifact.
            </p>
          </div>
          <label>
            Ranking outcome
            <select value={rankingTarget} onChange={(event) => setRankingTarget(event.target.value)}>
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
                  <td className="left name">{rankerLabel(row.ranker)}</td>
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
                  <td>{correlation(row.pool_spearman)}</td>
                  <td title={`Ideal: ${row.ideal_top_k_actual_mean.toFixed(1)}`}>
                    {row.top_k_actual_mean.toFixed(1)}
                  </td>
                  <td title={row.best_baseline ? `Versus ${rankerLabel(row.best_baseline)}` : undefined}>
                    {percentagePointLift(row.hit_rate_lift)}
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

      {(data.fitted_model_summary?.length ?? 0) > 0 && (
        <div className="ranking-verdict">
          <div className="ranking-heading">
            <div>
              <h3>Learned weights (walk-forward fitted rankers)</h3>
              <p className="legend tight">
                Standardized ridge coefficients, refitted each season on every earlier completed
                fold with the ridge strength chosen by nested walk-forward validation. The small
                number is the coefficient's spread across refits — a weight that swings is not
                evidence.
              </p>
            </div>
            <label>
              Model
              <select value={fittedModel} onChange={(event) => setFittedModel(event.target.value)}>
                {[...new Set(data.fitted_model_summary.map((entry) => entry.model))].map((name) => (
                  <option key={name} value={name}>{rankerLabel(name)}</option>
                ))}
              </select>
            </label>
          </div>
          {(() => {
            const entries = data.fitted_model_summary.filter((entry) => entry.model === fittedModel)
            if (entries.length === 0) return null
            const features = Object.keys(entries[0].coefficients)
            return (
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Pos</th>
                      <th>Scope</th>
                      <th>Train rows</th>
                      <th>λ</th>
                      <th>Refits</th>
                      {features.map((feature) => (
                        <th key={feature}>{rankerLabel(feature)}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {entries.map((entry) => (
                      <tr key={`${entry.model}-${entry.position}`}>
                        <td className="strong">{entry.position}</td>
                        <td className="dim">{entry.scope}</td>
                        <td>{entry.n_train}</td>
                        <td className="dim">{entry.ridge_lambda ?? '—'}</td>
                        <td>{entry.refits}</td>
                        {features.map((feature) => (
                          <td key={feature}>
                            <span className="strong">{correlation(entry.coefficients[feature] ?? null)}</span>
                            {entry.coefficient_sd_across_refits[feature] != null && (
                              <span className="dim"> ±{entry.coefficient_sd_across_refits[feature].toFixed(2)}</span>
                            )}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )
          })()}
        </div>
      )}

      <div className="report-cards">
        {(['strong', 'useful', 'harmful', 'redundant', 'weak'] as MetricAssessment[]).map((key) => (
          <button
            type="button"
            className={`report-card ${key}`}
            key={key}
            onClick={() => setAssessment(key)}
          >
            <span>{evidenceRows.filter((row) => row.assessment === key).length}</span>
            {key}
          </button>
        ))}
      </div>

      <div className="controls report-controls">
        <label>
          Outcome
          <select value={target} onChange={(event) => setTarget(event.target.value)}>
            {data.targets.map((entry) => (
              <option key={entry.key} value={entry.key}>{entry.label}</option>
            ))}
          </select>
        </label>
        <label>
          Position
          <select value={position} onChange={(event) => setPosition(event.target.value)}>
            {['ALL', 'QB', 'RB', 'WR', 'TE'].map((entry) => (
              <option key={entry} value={entry}>{entry}</option>
            ))}
          </select>
        </label>
        <label>
          Metric family
          <select value={group} onChange={(event) => setGroup(event.target.value)}>
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

      <p className="legend tight">
        Spearman measures next-season rank association. “Partial vs prior” measures what remains
        after controlling for the historical PPG prior. Consistency is the share of forecast years
        agreeing with the pooled direction. “Harmful” means a repeatable signal points opposite
        the metric's configured expected direction.
      </p>

      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th className="left">Metric</th>
              <th className="left">Family</th>
              <th>Assessment</th>
              <th>Spearman</th>
              <th>Partial vs prior</th>
              <th>Fold 2.5–97.5%</th>
              <th>Consistency</th>
              <th>Coverage</th>
              <th>N</th>
              <th>Year folds</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => {
              const definition = data.metrics.find((metric) => metric.key === row.metric)
              return (
                <tr key={`${row.metric}-${row.target}-${row.position}`}>
                  <td className="left">
                    <span className="name" title={definition?.description}>{row.label}</span>
                  </td>
                  <td className="left dim">{row.group}</td>
                  <td><span className={`metric-assessment ${row.assessment}`}>{row.assessment}</span></td>
                  <td className="strong">{correlation(row.spearman)}</td>
                  <td className="strong">{correlation(row.partial_spearman)}</td>
                  <td className="dim">
                    {row.spearman_ci_low == null
                      ? '—'
                      : `${correlation(row.spearman_ci_low)} to ${correlation(row.spearman_ci_high)}`}
                  </td>
                  <td>{percent(row.direction_consistency)}</td>
                  <td>{percent(row.coverage)}</td>
                  <td>{row.n}</td>
                  <td className="folds">
                    {row.fold_results.map((fold) => (
                      <span key={fold.forecast_season} title={`n=${fold.n}`}>
                        {fold.forecast_season}: {correlation(fold.spearman)}
                      </span>
                    ))}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      {rows.length === 0 && <div className="notice">No metrics match these filters.</div>}

      <h3 className="section-head">Direct forecast calibration</h3>
      <p className="legend tight">
        Error measures apply only where a projection and outcome share the same units. Positive
        bias means the model over-projected the outcome; rank correlation measures ordering.
      </p>
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
                <td>{correlation(row.bias)}</td>
                <td>{correlation(row.spearman)}</td>
                <td>{row.n}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

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
