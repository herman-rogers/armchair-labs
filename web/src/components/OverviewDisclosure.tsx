import type { ReactNode } from 'react'
import './OverviewDisclosure.css'

/** Compact, matching source and attention controls for league pages. */
export function OverviewDisclosure({ label, meta, badge, children }: {
  label: string; meta?: string; badge?: ReactNode; children: ReactNode
}) {
  return <details className="overview-disclosure">
    <summary><strong>{label}</strong>{meta && <span className="overview-disclosure-meta">{meta}</span>}{badge}<span className="overview-disclosure-chevron" aria-hidden="true">⌄</span></summary>
    <div className="overview-disclosure-content">{children}</div>
  </details>
}
