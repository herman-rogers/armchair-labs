import { CartesianGrid, Line, LineChart, ReferenceArea, Tooltip, XAxis, YAxis } from 'recharts'
import type { ScoringSeries, ScoringWeek } from '../api/scoring'
import { chartAxis, chartTooltip } from '../chartTheme'
import { fixed } from '../format'
import { ChartFrame, ChartLegend } from './Charts'
import './ScoringTrends.css'

export function ScoringLegend({ teams = false }: { teams?: boolean }) {
  return <ChartLegend items={[
    { label: teams ? 'Predicted · OD + K/DST' : 'Predicted · Opening Drive', color: 'var(--chart-blue)', dashed: true },
    { label: teams ? 'Actual · ESPN team score' : 'Actual production', color: 'var(--buy)' },
    { label: 'Season-to-date average', color: 'var(--chart-amber)', dashed: true },
  ]} />
}

export function ScoringChart({ series, description, throughWeek, lastWeek, domain, teams = false, showValues = true }: {
  series: ScoringSeries; description: string; throughWeek: number; lastWeek: number
  domain: [number, number]; teams?: boolean; showValues?: boolean
}) {
  const rows = series.weeks.map(w => ({ ...w, average: series.average }))
  return <>
    <ChartFrame title={series.name} description={description} height={255}>
      <LineChart data={rows} margin={{ top: 8, right: 14, bottom: 5, left: -16 }} accessibilityLayer={!teams}>
        <CartesianGrid stroke="var(--border)" vertical={false} />
        <XAxis dataKey="week" type="number" domain={[1, lastWeek]} ticks={rows.map(r => r.week)} tickFormatter={w => `W${w}`} minTickGap={10} {...chartAxis} />
        <YAxis domain={domain} {...chartAxis} width={48} />
        {throughWeek < lastWeek && <ReferenceArea x1={throughWeek + .5} x2={lastWeek} fill="var(--surface-raised)" fillOpacity={.65} />}
        <Tooltip {...chartTooltip} content={({ active, payload }) => {
          const row = payload?.[0]?.payload as (ScoringWeek & { average: number | null }) | undefined
          return active && row ? <div className="chart-tooltip">
            <strong>Week {row.week}{row.opponent ? ` · vs ${row.opponent}` : ''}</strong>
            <p>Actual: {fixed(row.actual)} pts</p><p>Predicted: {fixed(row.predicted)} pts</p>
            <p>Season average: {fixed(row.average)} pts</p><p>{row.status}</p>
            {teams && row.adjustments?.map((change, i) => <p key={i}>{change}</p>)}
            {row.origin_week != null && <p>{row.origins && row.origins.length > 1 ? `Forecast cutoffs: ${row.origins.join(', ')}` : row.origin_week === 0 ? 'Preseason forecast' : `Forecast after Week ${row.origin_week}`}</p>}
            {row.actual != null && row.predicted != null && <p>Actual − predicted: {fixed(row.actual - row.predicted)} pts</p>}
          </div> : null
        }} />
        <Line dataKey="average" name="Season average" stroke="var(--chart-amber)" strokeDasharray="2 5" strokeWidth={1.3} dot={false} activeDot={false} isAnimationActive={false} />
        <Line dataKey="predicted" name="Predicted" stroke="var(--chart-blue)" strokeWidth={2} strokeDasharray="7 5" dot={{ r: 2 }} connectNulls={false} isAnimationActive={false} />
        <Line dataKey="actual" name="Actual" stroke="var(--buy)" strokeWidth={2} dot={{ r: 3 }} connectNulls={false} isAnimationActive={false} />
      </LineChart>
    </ChartFrame>
    {showValues && <details className="scoring-values"><summary>Weekly values &amp; forecast sources</summary>
      <div className="table-wrap"><table><caption>{series.name} · points by week</caption>
        <thead><tr><th>Week</th><th>Actual</th><th>Predicted</th><th>Cutoff</th><th>Status</th></tr></thead>
        <tbody>{rows.map(w => <tr key={w.week}><th scope="row">{w.week}</th><td>{fixed(w.actual)}</td><td>{fixed(w.predicted)}</td><td>{w.origin_week == null ? '—' : w.origin_week === 0 ? 'Preseason' : `W${w.origin_week}`}</td><td>{w.status}
          {teams && w.starters && <details><summary>Lineup ({w.starters.length})</summary><ul>{w.starters.map((p, i) => <li key={i}>{p.slot} · {p.name}: {fixed(p.prediction)} · {p.basis}</li>)}</ul>{w.adjustments?.map((change, i) => <p key={i}>{change}</p>)}</details>}
          {w.reconciled === false && <span className="faint"> · ESPN total; lineup breakdown does not reconcile</span>}
        </td></tr>)}</tbody>
      </table></div>
    </details>}
  </>
}
