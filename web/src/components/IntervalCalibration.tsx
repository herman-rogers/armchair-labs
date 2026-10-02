import type { MetricReport } from '../api/types'

export function IntervalCalibration({ rows = [], start, end, position }: {
  rows: MetricReport['uncertainty_calibration']; start?: number; end?: number; position: string
}) {
  const selected = rows.filter(row => (start == null || row.forecast_season >= start)
    && (end == null || row.forecast_season <= end) && (position === 'ALL' || row.position === position))
  const groups = new Map<string, { position: string; population: string; n: number; covered: number; nominal: number; folds: number }>()
  for (const row of selected) {
    const key = `${row.position}-${row.population}`
    const group = groups.get(key) ?? { position: row.position, population: row.population, n: 0, covered: 0, nominal: 0, folds: 0 }
    group.n += row.n_test; group.covered += row.observed_coverage * row.n_test
    group.nominal += row.nominal_coverage * row.n_test; group.folds += 1
    groups.set(key, group)
  }
  return <section className="analysis-section" aria-label="Historical uncertainty calibration">
    <h3>How often did the historical outcome range contain the result?</h3>
    <p>Core season-point residual ranges were estimated using earlier seasons only. Coverage measures the share of later outcomes inside the range—not the chance an individual player succeeds. These are season-total diagnostics, not weekly floors, rookie analog spreads, or published confidence scores.</p>
    {!selected.length ? <p>No held-out interval results for this population and window.</p> : <>
      <div className="table-wrap"><table><thead><tr><th>Position</th><th>Population</th><th>Nominal coverage</th><th>Observed coverage</th><th>Forecasts</th><th>Seasons</th></tr></thead>
        <tbody>{[...groups.entries()].map(([key, row]) => <tr key={key}><td>{row.position}</td><td>{row.population}</td>
          <td>{row.n ? (100 * row.nominal / row.n).toFixed(1) + '%' : '—'}</td><td>{row.n ? (100 * row.covered / row.n).toFixed(1) + '%' : '—'}</td><td>{row.n}</td><td>{row.folds}</td></tr>)}</tbody></table></div>
      <details><summary>Season-by-season interval checks</summary><div className="table-wrap"><table>
        <thead><tr><th>Season</th><th>Pos</th><th>Population</th><th>Training rows</th><th>Test rows</th><th>Observed coverage</th><th>Residual band (season points)</th></tr></thead>
        <tbody>{selected.map(row => <tr key={`${row.forecast_season}-${row.position}-${row.population}`}>
          <td>{row.forecast_season}</td><td>{row.position}</td><td>{row.population}</td><td>{row.n_train}</td><td>{row.n_test}</td><td>{(100 * row.observed_coverage).toFixed(1)}%</td><td>{row.residual_lower.toFixed(1)} to {row.residual_upper.toFixed(1)}</td>
        </tr>)}</tbody></table></div></details>
    </>}
  </section>
}
