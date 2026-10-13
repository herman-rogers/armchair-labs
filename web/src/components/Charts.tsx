import { useId, type ReactNode } from 'react'
import { ResponsiveContainer } from 'recharts'

/** Shared chart sizing, theme and accessible caption for analytics pages. */
export function ChartFrame({ title, description, height = 240, plain = false, children }: {
  title: string; description: string; height?: number
  /** The children draw themselves (an SVG with its own viewBox) instead of filling a Recharts container. */
  plain?: boolean; children: ReactNode
}) {
  const id = useId()
  return <figure className="chart-frame" aria-labelledby={id}>
    <figcaption id={id}><strong>{title}</strong><span>{description}</span></figcaption>
    {plain ? children : <div className="chart-canvas" style={{ height }}>
      <ResponsiveContainer width="100%" height="100%" minWidth={0}>{children}</ResponsiveContainer>
    </div>}
  </figure>
}

export type LegendShape = 'line' | 'dot' | 'ring' | 'diamond' | 'bracket'

export function ChartLegend({ items }: { items: { label: string; color: string; dashed?: boolean; shape?: LegendShape }[] }) {
  return <ul className="chart-legend" aria-label="Chart legend">{items.map(item => <li key={item.label}>
    <span aria-hidden="true" className={item.shape && item.shape !== 'line' ? `legend-${item.shape}` : undefined}
      style={{ borderColor: item.color, borderStyle: item.dashed ? 'dashed' : 'solid', color: item.color }} />{item.label}
  </li>)}</ul>
}
