import { Bar, BarChart, CartesianGrid, ReferenceLine, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis } from 'recharts'
import type { TeamPair, TeamPlayer, TeamRisk } from '../api/teamAnalysis'
import { chartAxis, chartTooltip } from '../chartTheme'
import { fixed } from '../format'
import { ChartFrame } from './Charts'

export function VarianceChart({ risk, pending }: { risk: TeamRisk; pending: boolean }) {
  const rows = [
    { label: 'Independent Σσ²', value: risk.independent_variance, fill: 'var(--text-dim)' },
    { label: 'Relationships 2Σcov', value: risk.covariance_effect, fill: (risk.covariance_effect ?? 0) < 0 ? 'var(--te)' : 'var(--wr)' },
    { label: 'Combined σ²', value: risk.variance, fill: 'var(--accent)' },
  ]
  if (pending) return <p className="ta-empty" role="status">Updating variance decomposition…</p>
  if (rows.every(row => row.value == null)) return <p className="ta-empty">Variance decomposition unavailable for this selection.</p>
  return <ChartFrame title="Variance decomposition" description="pts² · selected starters’ common sample" height={134}>
    <BarChart data={rows} layout="vertical" margin={{ left: 0, right: 4, top: 8, bottom: 0 }} accessibilityLayer>
      <CartesianGrid horizontal={false} stroke="var(--border)" />
      <XAxis type="number" {...chartAxis} tickCount={4} />
      <YAxis type="category" dataKey="label" width={123} interval={0} {...chartAxis} />
      <YAxis yAxisId="values" orientation="right" type="category" dataKey="label" width={55} interval={0} {...chartAxis}
        tickFormatter={label => {
          const row = rows.find(r => r.label === label)
          return `${label === 'Relationships 2Σcov' && row?.value != null && row.value > 0 ? '+' : ''}${fixed(row?.value)}`
        }} />
      <ReferenceLine x={0} stroke="var(--text-dim)" />
      <Tooltip {...chartTooltip} formatter={value => [`${typeof value === 'number' ? fixed(value) : '—'} pts²`, 'Variance contribution']} />
      <Bar dataKey="value" name="Variance contribution" barSize={15} radius={3} isAnimationActive={false} />
    </BarChart>
  </ChartFrame>
}

export function PairScatterChart({ pair, a, b }: { pair: TeamPair; a: TeamPlayer; b: TeamPlayer }) {
  if (!pair.points.length) return <p className="ta-empty">No shared scoring observations to plot.</p>
  const short = (name: string) => { const parts = name.split(' '); return parts.length > 1 ? `${parts[0][0]}. ${parts.slice(1).join(' ')}` : name }
  return <ChartFrame title="Observed scoring together" description={`${a.name} × ${b.name} · each point is one shared game`} height={260}>
    <ScatterChart margin={{ left: 8, right: 12, top: 8, bottom: 24 }} accessibilityLayer>
      <CartesianGrid stroke="var(--border)" />
      <XAxis type="number" dataKey="a" name={a.name} {...chartAxis}
        domain={['auto', 'auto']} label={{ value: `${short(a.name)} · pts`, position: 'bottom', offset: 4, fill: 'var(--text-dim)', fontSize: 11 }} />
      <YAxis type="number" dataKey="b" name={b.name} width={48} {...chartAxis}
        domain={['auto', 'auto']} label={{ value: `${short(b.name)} · pts`, angle: -90, position: 'insideLeft', fill: 'var(--text-dim)', fontSize: 11 }} />
      <ZAxis range={[50, 50]} />
      <Tooltip {...chartTooltip} cursor={{ stroke: 'var(--border)', strokeDasharray: '3 3' }} content={({ active, payload }) => {
        const game = payload?.[0]?.payload as TeamPair['points'][number] | undefined
        return active && game ? <div className="chart-tooltip"><strong>{game.season} · Week {game.week}</strong><p>{a.name}: {fixed(game.a)} pts</p><p>{b.name}: {fixed(game.b)} pts</p></div> : null
      }} />
      <Scatter name="Shared games" data={pair.points} fill="var(--accent)" fillOpacity={.8} isAnimationActive={false} />
    </ScatterChart>
  </ChartFrame>
}
