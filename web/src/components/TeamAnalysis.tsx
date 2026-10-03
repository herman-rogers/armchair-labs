import { useEffect, useState, type CSSProperties } from 'react'
import { useQuery } from '@tanstack/react-query'
import { teamAnalysisCatalogQuery, teamAnalysisQuery } from '../api/queries'
import type { TeamAnalysisData, TeamPair, TeamPlayer, TeamRisk } from '../api/teamAnalysis'
import { useDataRelease } from '../dataRelease'
import { useUrlParams } from '../navigation'
import { QueryError } from './Controls'
import { AnalysisPanel } from './AnalysisPanel'
import { PairScatterChart, VarianceChart } from './TeamAnalysisCharts'
import './TeamAnalysis.css'

const codes = 'ARI ATL BAL BUF CAR CHI CIN CLE DAL DEN DET GB HOU IND JAX KC LA LAC LV MIA MIN NE NO NYG NYJ PHI PIT SEA SF TB TEN WAS'.split(' ')
const number = (v: number | null | undefined, digits = 1) => v == null ? '—' : v.toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits })
const signed = (v: number | null | undefined, digits = 1) => v == null ? '—' : `${v > 0 ? '+' : ''}${number(v, digits)}`
const short = (name: string) => `${name.split(' ')[0][0]}. ${name.split(' ').slice(1).join(' ')}`

function RiskSummary({ risk, players, pending }: { risk: TeamRisk; players: TeamPlayer[]; pending: boolean }) {
  const value = (v: number | null) => pending ? '…' : number(v)
  const contributions = risk.contributions
  return <AnalysisPanel className="ta-panel ta-risk" aria-labelledby="risk-heading" aria-busy={pending}>
    <div className="ta-section-title"><h3 id="risk-heading">Selected starters</h3>
      <span className="ta-badge" title="Observed games shared by every selected player. Not a forecast or calibrated future risk estimate.">Historical · n = {pending ? '…' : risk.n} · df = {pending ? '…' : risk.df}</span></div>
    <div className="ta-kpis">
      <div><span>Mean points</span><strong>{value(risk.mean)}</strong><small>pts / game</small></div>
      <div><span>Scoring volatility σ</span><strong>{value(risk.sd)}</strong><small>pts · standard deviation</small>
        {risk.sd_interval && !pending && <small className="ta-ci" title="Exploratory 95% interval from 2,000 paired-game bootstrap samples within seasons.">95% CI {number(risk.sd_interval[0])}–{number(risk.sd_interval[1])}</small>}</div>
      <div className={(risk.variance_change_pct ?? 0) >= 0 ? 'ta-positive' : 'ta-negative'}>
        <span>Correlation effect</span><strong>{pending ? '…' : signed(risk.variance_change_pct)}{risk.variance_change_pct != null && !pending ? '%' : ''}</strong>
        <small title="100 × (combined variance / sum of individual variances − 1), on exactly the same games.">change in variance</small>
        {risk.variance_change_interval && !pending && <small className="ta-ci" title="Exploratory 95% paired-game bootstrap interval; not a calibrated forecast range.">95% CI {signed(risk.variance_change_interval[0])}% / {signed(risk.variance_change_interval[1])}%</small>}</div>
      <div><span>Independent σ</span><strong>{value(risk.independent_sd)}</strong><small>pts · zero-covariance baseline</small></div>
    </div>
    {!pending && risk.status !== 'available' && <div className="ta-empty" role="status">
      {risk.status === 'empty_selection' ? 'Select starters to calculate σ².' : `Insufficient overlap · ${risk.n} shared games · ${risk.df} residual df`}
    </div>}
    <div className="ta-variance" aria-label="Variance decomposition"><VarianceChart risk={risk} pending={pending} /></div>
    {risk.quantiles && !pending && <div className="ta-quantiles" aria-label="Observed combined score quantiles">
      <h4>Observed points</h4>{(['p10', 'p25', 'p50', 'p75', 'p90'] as const).map(p => <div key={p}><span>{p.toUpperCase()}</span><strong>{number(risk.quantiles![p])}</strong></div>)}
    </div>}
    {contributions.length > 0 && !pending && <div className="ta-table-scroll"><table className="ta-contributions">
      <caption>Variance contribution · common sample</caption>
      <thead><tr><th>Starter</th><th>Own σ²</th><th>Σcov</th><th>Total pts²</th></tr></thead>
      <tbody>{contributions.map(c => <tr key={c.player_id}>
        <th>{players.find(p => p.player_id === c.player_id)?.name}</th><td>{number(c.variance)}</td><td>{signed(c.covariance)}</td><td>{number(c.total)}</td>
      </tr>)}</tbody>
    </table></div>}
  </AnalysisPanel>
}

function PairDetails({ pair, players }: { pair: TeamPair; players: TeamPlayer[] }) {
  const a = players.find(p => p.player_id === pair.a)!
  const b = players.find(p => p.player_id === pair.b)!
  return <div className="ta-pair-detail">
    <div><h4>{a.name} ↔ {b.name}</h4><div className="ta-pair-stats">
      <div><span>r</span><strong>{signed(pair.r, 2)}</strong></div>
      <div><span>cov · pts²</span><strong>{signed(pair.covariance)}</strong></div>
      <div><span>Shared games</span><strong>{pair.n}</strong></div>
      <div><span title="Approximate Fisher interval for this pair; not adjusted for testing multiple pairs.">95% CI · r</span><strong>{pair.interval ? `${signed(pair.interval[0], 2)} / ${signed(pair.interval[1], 2)}` : '—'}</strong></div>
    </div><small>{pair.years.join(' · ') || 'No shared seasons'}</small></div>
    <PairScatterChart pair={pair} a={a} b={b} />
  </div>
}

// Recharts has no native heatmap. Keep this as a semantic, keyboard-operable
// data table; the variance and observed-score charts use Recharts below it.
function CorrelationMatrix({ data, selected }: { data: TeamAnalysisData; selected: string[] }) {
  const [detail, setDetail] = useState<string | null>(null)
  const [params, update] = useUrlParams()
  const minimum = ['3', '8', '15', '25'].includes(params.get('min_n') ?? '') ? Number(params.get('min_n')) : 8
  const pair = (a: string, b: string) => data.pairs.find(p => (p.a === a && p.b === b) || (p.a === b && p.b === a))
  const inspected = data.pairs.find(p => `${p.a}:${p.b}` === detail)
    ?? data.pairs.find(p => selected.includes(p.a) && selected.includes(p.b)) ?? data.pairs[0]
  return <AnalysisPanel className="ta-panel" aria-labelledby="correlations-heading">
    <div className="ta-section-title"><div><h3 id="correlations-heading">Player relationships</h3><span className="ta-meta">Pearson r · pair-specific games</span></div>
      <label className="ta-minimum">Minimum n<select aria-label="Minimum shared games" value={minimum} onChange={e => update({ min_n: e.target.value })}>{[3, 8, 15, 25].map(n => <option key={n}>{n}</option>)}</select></label>
    </div>
    <div className="ta-matrix-legend"><span>−1</span><i /><span>+1</span><span>— insufficient sample</span></div>
    <div className="ta-matrix-scroll" tabIndex={0} aria-label="Scrollable player correlation matrix">
      <table className="ta-matrix"><thead><tr><th scope="col">QB / WR / TE</th>{data.players.map(p => <th scope="col" key={p.player_id} className={selected.includes(p.player_id) ? 'ta-matrix-selected' : ''}><span title={p.name}>{short(p.name)}</span></th>)}</tr></thead>
        <tbody>{data.players.map(a => <tr key={a.player_id}><th scope="row" className={selected.includes(a.player_id) ? 'ta-matrix-selected' : ''}>{a.name}<small>{a.position}</small></th>
          {data.players.map(b => {
            const p = pair(a.player_id, b.player_id)
            const available = p != null && p.n >= minimum && p.r != null
            const style = available ? { '--ta-weight': `${Math.min(85, Math.abs(p.r!) * 85)}%` } as CSSProperties : undefined
            return <td key={b.player_id}>{a.player_id === b.player_id ? <span className="ta-diagonal">·</span> : <button type="button" style={style}
              className={`ta-cell ${available ? p.r! >= 0 ? 'ta-cell-positive' : 'ta-cell-negative' : 'ta-cell-empty'}`}
              aria-label={`${a.name} and ${b.name}: ${available ? signed(p.r, 2) : 'insufficient sample'}, ${p?.n ?? 0} games`}
              title={`${a.name} ↔ ${b.name}\nr ${signed(p?.r, 2)} · n ${p?.n ?? 0}${p?.interval ? ` · 95% CI ${signed(p.interval[0], 2)} to ${signed(p.interval[1], 2)}` : ''}`}
              onClick={() => p && setDetail(`${p.a}:${p.b}`)}>{available ? signed(p.r, 2) : '—'}<small>n={p?.n ?? 0}</small></button>}</td>
          })}</tr>)}</tbody>
      </table>
    </div>
    {inspected && <PairDetails pair={inspected} players={data.players} />}
  </AnalysisPanel>
}

export function TeamAnalysis() {
  const { token } = useDataRelease()
  const tableCatalog = useQuery(teamAnalysisCatalogQuery(token))
  const { refetch: refreshTableCatalog } = tableCatalog
  useEffect(() => {
    const refresh = () => { void refreshTableCatalog() }
    window.addEventListener('table-catalog-changed', refresh)
    return () => window.removeEventListener('table-catalog-changed', refresh)
  }, [refreshTableCatalog])
  const [params, update] = useUrlParams()
  const team = params.get('team') ?? 'LA'
  const start = params.get('start') ?? '2021'
  const end = params.get('end') ?? 'latest'
  const qb = params.get('qb') ?? 'current'
  const sample = params.get('sample') ?? 'starter'
  const basis = params.get('basis') ?? 'season'
  const request = new URLSearchParams({ team, start, qb, sample, basis })
  if (end !== 'latest') request.set('end', end)
  if (params.has('players')) request.set('players', params.get('players') === 'none' ? '' : params.get('players')!)
  const tableVersion = tableCatalog.isError ? undefined : tableCatalog.data?.table_version
  const query = useQuery(teamAnalysisQuery(request, token, tableVersion))
  const data = !tableCatalog.isError && !query.isError ? query.data : undefined
  const selected = params.has('players') ? params.get('players') === 'none' ? [] : params.get('players')!.split(',') : data?.risk.player_ids ?? []
  const year = data?.report.season ?? tableCatalog.data?.season ?? new Date().getFullYear()
  const earliest = tableCatalog.data?.earliest_season ?? 2013
  const seasons = Array.from({ length: Math.max(1, year - earliest + 1) }, (_, i) => earliest + i).reverse()
  const changeTeam = (value: string) => update({ team: value, players: null, qb: null })
  const toggle = (player: TeamPlayer) => {
    let next = selected.includes(player.player_id) ? selected.filter(id => id !== player.player_id) : [...selected, player.player_id]
    if (player.position === 'QB' && next.includes(player.player_id)) next = next.filter(id => id === player.player_id || data?.players.find(p => p.player_id === id)?.position !== 'QB')
    update({ players: next.length ? next.join(',') : 'none' })
  }
  return <section className="team-analysis" aria-label="Team analysis">
    <div className="analysis-controls ta-controls">
      <label>NFL team<select aria-label="NFL team" value={team} onChange={e => changeTeam(e.target.value)}>{(data?.teams ?? codes.map(id => ({ id, name: id }))).map(t => <option value={t.id} key={t.id}>{t.name}</option>)}</select></label>
      <label>From<select aria-label="From season" value={start} onChange={e => update({ start: e.target.value, players: null, qb: null })}>{seasons.map(s => <option key={s}>{s}</option>)}</select></label>
      <label>Through<select aria-label="Through season" value={end} onChange={e => update({ end: e.target.value, players: null, qb: null })}><option value="latest">Latest{data ? ` · ${year} W${data.report.through_week}` : ''}</option>{seasons.map(s => <option key={s}>{s}</option>)}</select></label>
      <label>Starting quarterback<select aria-label="Starting quarterback" value={qb} onChange={e => update({ qb: e.target.value, players: null })}><option value="current">Primary QB · latest season</option><option value="all">All starters</option>{data?.quarterbacks.map(p => <option key={p.player_id} value={p.player_id}>{p.name} · {p.games} games</option>)}</select></label>
      <label>Participation<select aria-label="Participation" value={sample} onChange={e => update({ sample: e.target.value })}><option value="starter">Starter roles · ≥50% snaps</option><option value="active">All offensive appearances</option></select></label>
      <label>Variance basis<select aria-label="Variance basis" value={basis} onChange={e => update({ basis: e.target.value })}><option value="season">Within season</option><option value="raw">Raw points</option></select></label>
    </div>
    <QueryError query={tableCatalog} label="Team observations unavailable" />
    <QueryError query={query} label="Team analysis unavailable" />
    {!tableCatalog.isError && query.isPending && <div className="ta-loading" role="status">Loading team relationships…</div>}
    {data && <>
      <div className="ta-sample-line"><span>{data.teams.find(t => t.id === team)?.name}</span><span>{data.start}–{data.end}</span><span>{data.team_games} games</span><span>{data.players.length} players</span><span>REG · league scoring</span><span className="ta-badge">Historical</span></div>
      <div className="ta-top-grid">
        <AnalysisPanel className="ta-panel ta-selection" aria-labelledby="select-starters-heading">
          <div className="ta-section-title"><h3 id="select-starters-heading">Starting roster</h3><button type="button" className="ta-text-button" onClick={() => update({ players: 'none' })}>Clear</button></div>
          <div className="ta-selection-header"><span>{selected.length} / 8 selected</span><span>n · μ · σ</span></div>
          {(['QB', 'WR', 'TE'] as const).map(position => <fieldset key={position}><legend>{position}</legend>
            {data.players.filter(p => p.position === position).map(p => <label className={`ta-player ${selected.includes(p.player_id) ? 'ta-player-selected' : ''}`} key={p.player_id}>
              <input type="checkbox" aria-label={`Select ${p.name}`} checked={selected.includes(p.player_id)} disabled={!selected.includes(p.player_id) && selected.length >= 8} onChange={() => toggle(p)} />
              <span className="ta-player-name">{p.name}<small>{p.first_season}–{p.last_season}</small></span><span className="ta-player-stats"><b>{p.games}</b><span>{number(p.mean)}</span><span>{number(p.sd)}</span></span>
            </label>)}
          </fieldset>)}
          {data.players.length === 0 && <div className="ta-empty">No qualifying appearances in this sample.</div>}
        </AnalysisPanel>
        <RiskSummary risk={data.risk} players={data.players} pending={query.isFetching} />
      </div>
      <CorrelationMatrix key={`${team}:${start}:${end}:${qb}`} data={data} selected={selected} />
      <details className="ta-method"><summary>Statistical definitions</summary><dl>
        <dt>Scoring variance</dt><dd>Var(ΣP) = ΣVar(P) + 2ΣCov(Pᵢ,Pⱼ)</dd>
        <dt>Selected roster</dt><dd>One common set of games for every selected player; no pairwise covariance stitching.</dd>
        <dt>Within season</dt><dd>Player scores centered within each shared season; pooled covariance denominator n − seasons.</dd>
        <dt>Raw points</dt><dd>Sample covariance denominator n − 1.</dd>
        <dt>μ and percentiles</dt><dd>Observed raw combined scores in the selected roster’s common games, under either variance basis.</dd>
        <dt>Participation</dt><dd>Positive offensive snaps; byes and absences excluded. WR/TE qualify after three 50%-snap games for the team in the selected period, before filtering QB starts. QBs require a recorded start. Injury-shortened games may be excluded by the starter-role filter.</dd>
        <dt>Sample limits</dt><dd>Minimum three games and two residual degrees of freedom. Gray cells indicate limited data. Pair confidence intervals are unadjusted, exploratory Fisher intervals.</dd>
        <dt>Risk intervals</dt><dd>Exploratory 95% percentile intervals from 2,000 paired-game bootstrap samples within seasons; not calibrated prediction intervals.</dd>
        <dt>Scope</dt><dd>Historical conditional scoring variability; excludes injury/availability risk and correlations with unselected roster players. No forecast calibration or inferred indirect connections.</dd>
      </dl></details>
    </>}
  </section>
}
