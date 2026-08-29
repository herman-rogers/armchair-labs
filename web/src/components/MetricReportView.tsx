import { useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { fetchMetricReport } from '../api/client'
import type { MetricAssessment, MetricBacktestResult } from '../api/types'

const ASSESSMENTS: Array<{ id: MetricAssessment | 'all'; label: string }> = [
  { id: 'all', label: 'All evidence' },
  { id: 'strong', label: 'Strong' },
  { id: 'useful', label: 'Useful' },
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

function evidenceScore(row: MetricBacktestResult) {
  return Math.abs(row.partial_spearman ?? row.spearman ?? -1)
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

  const groups = useMemo(
    () => ['All groups', ...new Set(report.data?.metrics.map((metric) => metric.group) ?? [])],
    [report.data],
  )
  const rows = useMemo(
    () =>
      (report.data?.results ?? [])
        .filter((row) => row.target === target)
        .filter((row) => row.position === position)
        .filter((row) => group === 'All groups' || row.group === group)
        .filter((row) => assessment === 'all' || row.assessment === assessment)
        .sort((left, right) => evidenceScore(right) - evidenceScore(left)),
    [assessment, group, position, report.data, target],
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
            Rolling forecasts for {data.data_summary.completed_forecasts.join(', ')} using{' '}
            {data.configuration.history_seasons} trailing seasons.{' '}
            {data.data_summary.pending_forecasts.length > 0 &&
              `${data.data_summary.pending_forecasts.join(', ')} is saved as a pending prospective forecast.`}
          </p>
        </div>
        <span className="count">
          Updated {new Date(data.generated_at).toLocaleString()} · {data.data_summary.forecast_rows} rows
        </span>
      </div>

      <div className="report-cards">
        {(['strong', 'useful', 'redundant', 'weak'] as MetricAssessment[]).map((key) => (
          <button
            type="button"
            className={`report-card ${key}`}
            key={key}
            onClick={() => setAssessment(key)}
          >
            <span>{data.data_summary.assessment_counts[key] ?? 0}</span>
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
        agreeing with the pooled direction. Negative values can still be predictive.
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
              <th>95% interval</th>
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

      <details className="metric-catalog">
        <summary>Metric catalog ({data.metrics.length})</summary>
        <div className="catalog-grid">
          {data.metrics.map((metric) => (
            <article key={metric.key}>
              <div><b>{metric.label}</b> <span className="badge rostered">{metric.group}</span></div>
              <p>{metric.description}</p>
              <span className="faint">
                {metric.available ? `${percent(metric.coverage)} historical coverage` : 'Unavailable in artifact'}
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
