import { useEffect, type ReactNode } from 'react'
import { useQuery } from '@tanstack/react-query'
import { dataCatalogQuery } from '../api/queries'
import { ReleaseContext, useDataRelease } from '../dataRelease'

export function DataReleaseProvider({ children, fallback = message => message }: { children: ReactNode; fallback?: (message: ReactNode) => ReactNode }) {
  const query = useQuery(dataCatalogQuery())
  const { refetch } = query
  useEffect(() => {
    const refresh = () => { void refetch() }
    window.addEventListener('data-catalog-changed', refresh)
    window.addEventListener('table-catalog-changed', refresh)
    return () => { window.removeEventListener('data-catalog-changed', refresh); window.removeEventListener('table-catalog-changed', refresh) }
  }, [refetch])
  if (query.isError) return fallback(<p className="notice" role="alert">Published tables unavailable: {query.error.message}</p>)
  if (!query.data) return fallback(<p role="status">Checking the published tables…</p>)
  const catalog = query.data
  return <ReleaseContext.Provider value={{ token: catalog.catalog_sha256, catalog }}>{children}</ReleaseContext.Provider>
}

export function DataReleaseStatus() {
  const { catalog } = useDataRelease()
  if (!catalog) return <p className="legend">No table catalog has been published yet.</p>
  const cutoff = catalog.observation_cutoff
  return <details className="data-release" aria-label="Published tables">
    <summary><strong>Observations</strong><span>{catalog.published_at ? `Published ${new Date(catalog.published_at).toLocaleDateString()}` : 'Published'} · NFL observations through {cutoff.season ?? '—'}, Week {cutoff.through_week ?? '—'}</span><span className="data-coverage-tag">{catalog.tables.length} tables</span></summary>
    <p>Player stats, rookies, league identities and team analysis read these DuckDB tables. Forecasts come from ChairStats campaign bundles, chosen per page.</p>
    <details><summary>Included tables</summary>
      <p className="legend">Catalog <code>{catalog.catalog_sha256.slice(0, 16)}</code>.</p>
      <div className="data-table-scroll"><table><thead><tr><th>Table</th><th>Rows</th><th>Built</th></tr></thead><tbody>
        {catalog.tables.map(table => <tr key={table.name}><td title={table.description}>{table.name.replaceAll('_', ' ')}</td><td>{table.rows.toLocaleString()}</td><td>{new Date(table.built_at).toLocaleDateString()}</td></tr>)}
      </tbody></table></div>
    </details>
  </details>
}
