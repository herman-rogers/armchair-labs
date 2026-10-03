import type { HTMLAttributes, ReactNode } from 'react'
import './AnalysisPanel.css'

/** Shared section surface from Team Analysis, with a dense dashboard variant. */
export function AnalysisPanel({ children, className = '', compact = false, heading, meta, ...props }: HTMLAttributes<HTMLElement> & {
  compact?: boolean; heading?: string; meta?: ReactNode;
}) {
  return <section {...props} className={`analysis-panel${compact ? ' analysis-panel-compact' : ''} ${className}`}>
    {heading && <header className="analysis-panel-heading"><h3>{heading}</h3>{meta && <span>{meta}</span>}</header>}
    {children}
  </section>
}
