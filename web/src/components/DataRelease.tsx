import { useEffect, type ReactNode } from 'react'
import { useQuery } from '@tanstack/react-query'
import { dataCatalogQuery } from '../api/queries'
import { ReleaseContext, useDataRelease } from '../dataRelease'

export function DataReleaseProvider({ children }: { children: ReactNode }) {
  const query = useQuery(dataCatalogQuery())
  const { refetch } = query
  useEffect(() => {
    const refresh = () => { void refetch() }
    window.addEventListener('data-catalog-changed', refresh)
    return () => window.removeEventListener('data-catalog-changed', refresh)
  }, [refetch])
  if (query.isError) return <p className="notice" role="alert">Current data catalog unavailable: {query.error.message}</p>
  if (!query.data) return <p role="status">Checking the current data release…</p>
  const catalog = query.data
  const token = catalog.available ? `${catalog.gold.version}@${catalog.published_at}` : undefined
  return <ReleaseContext.Provider value={{ token, catalog }}>{children}</ReleaseContext.Provider>
}

export function DataReleaseStatus() {
  const { catalog } = useDataRelease()
  if (!catalog?.available) return <p className="legend">A canonical data release has not been published yet.</p>
  const { coverage, current_observations: observations } = catalog
  const observed = coverage.forecast_rows ? 100 * coverage.observed_roster_state / coverage.forecast_rows : 0
  return <details className="data-release" aria-label="Current data release">
    <summary><strong>Canonical data</strong><span>Published {new Date(catalog.published_at).toLocaleDateString()} · NFL observations through {observations.season}, Week {observations.through_week}</span><span className="data-coverage-tag">{coverage.coverage_complete ? 'Coverage complete' : 'Coverage incomplete'}</span></summary>
    <p>Player measurements and eligible analysis use this verified release. Older forecasts and decision experiments are preserved separately in the Research archive.</p>
    <dl className="profile-facts">
      <div><dt>Forecast feature rows</dt><dd>{coverage.forecast_rows.toLocaleString()}</dd></div>
      <div><dt>Observed roster state</dt><dd>{observed.toFixed(1)}% <small>({coverage.observed_roster_state.toLocaleString()} rows)</small></dd></div>
      <div><dt>Prior team inferred</dt><dd>{coverage.inferred_prior_team.toLocaleString()}</dd></div>
      <div><dt>Known absence constraints</dt><dd>{coverage.known_absence.toLocaleString()}</dd></div>
    </dl>
    <p className="legend">Integrity checks passed. Missing evidence remains unknown; no recorded absence is not proof that a player is available.</p>
    <details><summary>Included tables and remaining gaps</summary>
      <p className="legend">Release: <code>{catalog.gold.version}</code>. Current observations were captured {new Date(observations.saved_at).toLocaleDateString()}.</p>
      <div className="data-table-scroll"><table><thead><tr><th>Table</th><th>Rows</th><th>Use</th></tr></thead><tbody>
        {catalog.tables.map(table => <tr key={table.name}><td title={table.description}>{table.name.replaceAll('_', ' ')}</td><td>{table.rows.toLocaleString()}</td><td>{table.role.replaceAll('_', ' ')}</td></tr>)}
      </tbody></table></div>
      <p className="legend">Dated contract values: {(coverage.nonnull_fields.contract_apy_cap_pct ?? 0).toLocaleString()} forecast rows. Timestamped depth ranks: {(coverage.nonnull_fields.depth_chart_rank ?? 0).toLocaleString()} rows. Quarantined records remain available for review and are excluded from their corresponding analysis tables.</p>
      {catalog.limitations.map(limit => <p className="legend" key={limit}>{limit}</p>)}
    </details>
  </details>
}
