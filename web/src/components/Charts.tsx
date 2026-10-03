import { useId, type ReactNode } from 'react'
import { ResponsiveContainer } from 'recharts'

/** Shared chart sizing, theme and accessible caption for analytics pages. */
export function ChartFrame({ title, description, height = 240, children }: {
  title: string; description: string; height?: number; children: ReactNode;
}) {
  const id = useId()
  return <figure className="chart-frame" aria-labelledby={id}>
    <figcaption id={id}><strong>{title}</strong><span>{description}</span></figcaption>
    <div className="chart-canvas" style={{ height }}>
      <ResponsiveContainer width="100%" height="100%" minWidth={0}>{children}</ResponsiveContainer>
    </div>
  </figure>
}

export function ChartLegend({ items }: { items: { label: string; color: string; dashed?: boolean }[] }) {
  return <ul className="chart-legend" aria-label="Chart legend">{items.map(item => <li key={item.label}>
    <span aria-hidden="true" style={{ borderColor: item.color, borderStyle: item.dashed ? 'dashed' : 'solid' }} />{item.label}
  </li>)}</ul>
}
