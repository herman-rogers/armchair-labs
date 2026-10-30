import type { ReactNode } from 'react'

/** Shared summary cards used by the league overview and team diagnostics. */
export function StatCards({ children, label, columns = 3, compact = false }: { children: ReactNode; label?: string; columns?: 3 | 4; compact?: boolean }) {
  return <div className={`league-update-cards outlook-kpis${columns === 4 ? ' stat-cards-four' : ''}${compact ? ' stat-cards-compact' : ''}`} aria-label={label}>{children}</div>
}

export function StatCard({ label, value, note, source, children }: { label: ReactNode; value: ReactNode; note?: ReactNode; source?: Source; children?: ReactNode }) {
  return <section><span className="eyebrow">{source && <SourceTag source={source} />}{label}</span><h3>{value}</h3>{note && <p>{note}</p>}{children}</section>
}

export type Source = 'model' | 'espn'
/** Marks whether a figure is a model forecast or an observed ESPN result. */
export function SourceTag({ source, children }: { source: Source; children?: ReactNode }) {
  return <span className={`source-tag ${source}`}>{children ?? (source === 'model' ? 'Model' : 'ESPN')}</span>
}

/** Dense secondary metrics, with no individual card surfaces. */
export function StatList({ items, label }: { items: { label: string; value: ReactNode; note?: string }[]; label?: string }) {
  return <dl className="stat-list" aria-label={label}>{items.map(item => <div key={item.label}>
    <dt>{item.label}</dt><dd>{item.value}</dd>{item.note && <small>{item.note}</small>}
  </div>)}</dl>
}

/** Rows of metrics tagged by source; every row shares one column grid so values line up. */
export function SourceMetrics({ rows }: { rows: { source: Source; label: string; items: { label: string; value: ReactNode; title?: string }[] }[] }) {
  return <div className="source-metrics">{rows.map(row => <dl key={row.source} className="source-metrics-row" aria-label={row.label}>
    <SourceTag source={row.source} />
    {row.items.map(item => <div key={item.label} title={item.title}><dt>{item.label}</dt><dd>{item.value}</dd></div>)}
  </dl>)}</div>
}
