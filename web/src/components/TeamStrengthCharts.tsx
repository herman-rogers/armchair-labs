import { Bar, BarChart, CartesianGrid, LabelList, Line, LineChart, ReferenceDot, ReferenceLine, Tooltip, XAxis, YAxis } from 'recharts'
import type { TeamStrength } from '../api/teamStrength'
import { fixed } from '../format'
import { ChartFrame, ChartLegend } from './Charts'
import { chartAxis, chartColors, chartTooltip } from '../chartTheme'

const point = (value: unknown) => typeof value === 'number' ? fixed(value) : '—'
const shortName = (name: string) => { const parts = name.split(' '); return parts.length > 1 ? `${parts[0][0]}. ${parts.slice(1).join(' ')}` : name }

export function ContributionChart({ team }: { team: TeamStrength }) {
  const starters = team.lineup.filter(p => ['QB', 'RB', 'WR', 'TE'].includes(p.position))
  const rows = starters.filter(p => p.usable && p.prediction != null).sort((a, b) => b.prediction! - a.prediction!)
  const missing = starters.filter(p => !p.usable || p.prediction == null)
  return <>
    {rows.length ? <ChartFrame title="Projected player contributions" description="NextGen weekly points · offensive starters" height={Math.max(180, rows.length * 26 + 32)}>
      <BarChart data={rows} layout="vertical" margin={{ left: 0, right: 36, top: 8, bottom: 0 }} accessibilityLayer>
        <CartesianGrid horizontal={false} stroke="var(--border)" />
        <XAxis type="number" {...chartAxis} /><YAxis type="category" dataKey="name" tickFormatter={shortName} width={108} interval={0} {...chartAxis} />
        <Tooltip {...chartTooltip} formatter={point} />
        <Bar dataKey="prediction" name="Weekly points" fill="var(--accent)" barSize={13} isAnimationActive={false} radius={[0, 2, 2, 0]}>
          <LabelList dataKey="prediction" position="right" formatter={point} fill="var(--text)" fontSize={11} />
        </Bar>
      </BarChart>
    </ChartFrame> : <p className="legend">No available starter forecasts to chart.</p>}
    {missing.length > 0 && <p className="notice">Not plotted: {missing.map(p => `${p.name} (${p.reason ?? 'forecast unavailable'})`).join('; ')}. These are unknown, not zero.</p>}
  </>
}

export function CompositionChart({ team }: { team: TeamStrength }) {
  const covered = team.history.filter(w => w.contributions != null)
  const players = [...new Map(covered.flatMap(w => w.contributions!.map(p => [p.espn_id, p] as const))).values()].sort((a, b) => a.espn_id - b.espn_id)
  const rows = covered.map(w => ({ week: `W${w.week}`, ...Object.fromEntries(players.map(p => [`p${p.espn_id}`, w.contributions!.find(c => c.espn_id === p.espn_id)?.points ?? 0])) }))
  const missing = team.history.filter(w => w.contributions == null)
  return <>
    {rows.length ? <ChartFrame title="Who supplied the points?" description="Actual offensive starter points · each segment is a player" height={240}>
      <BarChart data={rows} stackOffset="sign" margin={{ left: -12, right: 8, top: 8, bottom: 0 }} accessibilityLayer>
        <CartesianGrid vertical={false} stroke="var(--border)" /><XAxis dataKey="week" {...chartAxis} /><YAxis {...chartAxis} />
        <ReferenceLine y={0} stroke="var(--text-dim)" />
        <Tooltip {...chartTooltip} formatter={point} itemSorter={item => -(Number(item.value) || 0)} />
        {players.map((p, i) => <Bar key={p.espn_id} dataKey={`p${p.espn_id}`} name={p.name} stackId="points" fill={chartColors[i % chartColors.length]} maxBarSize={64} stroke="var(--surface)" strokeWidth={1} isAnimationActive={false} />)}
      </BarChart>
    </ChartFrame> : <p className="legend">No reconciled scoring breakdowns available.</p>}
    {missing.length > 0 && <p className="legend">Breakdown unavailable: {missing.map(w => `W${w.week}`).join(', ')}.</p>}
    {!!players.length && <><ChartLegend items={players.map((p, i) => ({ label: p.name, color: chartColors[i % chartColors.length] }))} />
    <details className="outlook-method"><summary>Weekly player values</summary>
      <div className="table-wrap"><table><caption>Actual offensive starter contributions</caption><thead><tr><th>Player</th>{covered.map(w => <th key={w.week}>W{w.week}</th>)}</tr></thead>
        <tbody>{players.map(p => <tr key={p.espn_id}><th scope="row">{p.name}</th>{covered.map(w => <td key={w.week}>{fixed(w.contributions!.find(c => c.espn_id === p.espn_id)?.points ?? 0)}</td>)}</tr>)}</tbody>
      </table></div><p>Zero means no recorded offensive starter points that week. Negative points remain negative. K/DST are excluded.</p>
    </details></>}
  </>
}

export function ReplacementChart({ team }: { team: TeamStrength }) {
  const total = team.current.total
  const rows = total == null ? [] : team.depth.filter(d => d.drop != null).sort((a, b) => b.drop! - a.drop!).map(d => ({ ...d, after: total - d.drop!, range: [Math.min(total, total-d.drop!), Math.max(total, total-d.drop!)] }))
  if (!rows.length) return <p className="legend">A complete starting offense and eligible backups are needed to chart replacement impact.</p>
  const domain = [Math.floor(Math.min(total!, ...rows.map(r => r.after)) / 5) * 5 - 5, Math.ceil(Math.max(total!, ...rows.map(r => r.after)) / 5) * 5 + 5]
  return <>
    <ChartFrame title="Offense after a starter absence" description="Each row replaces one starter; dots show current and replacement totals." height={Math.max(180, rows.length * 26 + 32)}>
      <BarChart data={rows} layout="vertical" margin={{ left: 0, right: 20, top: 8, bottom: 0 }} accessibilityLayer>
        <CartesianGrid horizontal={false} stroke="var(--border)" /><XAxis type="number" domain={domain} {...chartAxis} />
        <YAxis type="category" dataKey="starter" width={108} tickFormatter={shortName} interval={0} {...chartAxis} />
        <Tooltip {...chartTooltip} content={({ active, payload }) => {
          const row = payload?.[0]?.payload as typeof rows[number] | undefined
          return active && row ? <div className="chart-tooltip"><strong>Without {row.starter}</strong><p>{fixed(total)} → {fixed(row.after)} offensive points</p><p>{fixed(row.drop)} point loss · {row.replacement} enters</p><p>{row.moves.join(' · ')}</p></div> : null
        }} />
        <ReferenceLine x={total!} stroke="var(--text-dim)" strokeDasharray="3 3" />
        <Bar dataKey="range" name="Replacement impact" fill="var(--border)" barSize={4} isAnimationActive={false} />
        {rows.flatMap(r => [<ReferenceDot key={`${r.espn_id}-before`} x={total!} y={r.starter} r={3} fill="var(--text-dim)" stroke="none" />,
          <ReferenceDot key={`${r.espn_id}-after`} x={r.after} y={r.starter} r={4} fill="var(--accent)" stroke="var(--surface)" />])}
      </BarChart>
    </ChartFrame>
    <ChartLegend items={[{label:'Current offense',color:'var(--text-dim)'},{label:'With best eligible replacement',color:'var(--accent)'}]} />
    {team.depth.some(d => d.drop == null) && <p className="legend">Uncovered absences are omitted; see replacement details below.</p>}
  </>
}

export function PositionChart({ team, teams }: { team: TeamStrength; teams: TeamStrength[] }) {
  const rows = team.positions.map(p => {
    const peers = teams.map(t => t.positions.find(q => q.position === p.position))
    const comparable = p.rank != null && peers.every(q => q?.actual_ppg != null)
    return { position:p.position, points:p.actual_ppg, average:comparable ? peers.reduce((sum,q)=>sum+q!.actual_ppg!,0)/peers.length : null }
  })
  return <>
    <ChartFrame title="Position scoring vs league" description="Actual starter points per week · includes FLEX contributions" height={230}>
      <BarChart data={rows} layout="vertical" margin={{left:0,right:20,top:8,bottom:0}} accessibilityLayer>
        <CartesianGrid horizontal={false} stroke="var(--border)" /><XAxis type="number" {...chartAxis} /><YAxis dataKey="position" type="category" width={40} {...chartAxis} />
        <Tooltip {...chartTooltip} formatter={point} />
        <Bar dataKey="points" name="This team" fill="var(--accent)" barSize={10} isAnimationActive={false} />
        <Bar dataKey="average" name="League average" fill="var(--chart-slate)" barSize={4} isAnimationActive={false} />
      </BarChart>
    </ChartFrame><ChartLegend items={[{label:'This team',color:'var(--accent)'},{label:'League average',color:'var(--chart-slate)'}]} />
    {rows.some(r=>r.average==null) && <p className="legend">League benchmarks require comparable scoring coverage.</p>}
  </>
}

export function ResultsChart({ team, teams }: { team: TeamStrength; teams: TeamStrength[] }) {
  const rows = team.history.map(w => {
    const scores = teams.map(t => t.history.find(h => h.week === w.week)?.score)
    return { ...w, label:`W${w.week}`, average:scores.every(s=>s!=null) ? scores.reduce<number>((sum,s)=>sum+s!,0)/scores.length : null }
  })
  if (!rows.length) return <p className="legend">No completed weeks to chart.</p>
  return <>
    <ChartFrame title="Weekly scoring" description="Actual points, including K/DST · completed weeks only" height={200}>
      <LineChart data={rows} margin={{left:-12,right:12,top:8,bottom:0}} accessibilityLayer>
        <CartesianGrid vertical={false} stroke="var(--border)" /><XAxis dataKey="label" {...chartAxis} /><YAxis {...chartAxis} />
        <Tooltip {...chartTooltip} formatter={point} />
        <Line dataKey="score" name="This team" stroke="var(--accent)" strokeWidth={2} dot={{r:4}} isAnimationActive={false} />
        <Line dataKey="opponent_score" name="Opponent" stroke="var(--chart-amber)" dot={{r:3}} isAnimationActive={false} />
        <Line dataKey="average" name="League average" stroke="var(--text-dim)" strokeDasharray="4 4" dot={{r:2}} isAnimationActive={false} />
      </LineChart>
    </ChartFrame><ChartLegend items={[{label:'This team',color:'var(--accent)'},{label:'Opponent',color:'var(--chart-amber)'},{label:'League average',color:'var(--text-dim)',dashed:true}]} />
  </>
}
