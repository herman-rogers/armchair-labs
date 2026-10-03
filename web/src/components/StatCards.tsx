import type { ReactNode } from 'react'

/** Shared summary cards used by the league overview and team diagnostics. */
export function StatCards({ children, label, columns = 3, compact = false }: { children: ReactNode; label?: string; columns?: 3 | 4; compact?: boolean }) {
  return <div className={`league-update-cards outlook-kpis${columns === 4 ? ' stat-cards-four' : ''}${compact ? ' stat-cards-compact' : ''}`} aria-label={label}>{children}</div>
}

export function StatCard({ label, value, note, children }: { label: ReactNode; value: ReactNode; note?: ReactNode; children?: ReactNode }) {
  return <section><span className="eyebrow">{label}</span><h3>{value}</h3>{note && <p>{note}</p>}{children}</section>
}

/** Dense secondary metrics, with no individual card surfaces. */
export function StatList({ items, label }: { items: { label: string; value: ReactNode; note?: string }[]; label?: string }) {
  return <dl className="stat-list" aria-label={label}>{items.map(item => <div key={item.label}>
    <dt>{item.label}</dt><dd>{item.value}</dd>{item.note && <small>{item.note}</small>}
  </div>)}</dl>
}
