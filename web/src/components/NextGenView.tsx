import { useCatalog } from '../api/queries/hooks'
import { useDebounced } from '../api/queries/useDebounced'
import { evidenceQuery, forecastsQuery, registryQuery, individualEvidenceQuery } from '../api/queries'
import { Link, Outlet } from 'react-router'
import { useQuery } from '@tanstack/react-query'
import { getResponse } from '../api/client'
import type { Entry, Evaluation, Forecast } from '../api/nextgen'
import { DataReleaseStatus } from './DataRelease'
import { useDataRelease } from '../dataRelease'
import { DataTable, type Column } from './DataTable'
import { archivePath, intelligencePath, playerPath, useUrlPage, useUrlParams, useUrlState } from '../navigation'
import { useCurrentPage } from '../pageNavigation'
import { consensusColumns } from './RankComparison'
import { RookieView } from './RookieView'
import { QBVariations } from './QBVariations'
import { Pager, PositionOptions, QueryError } from './Controls'
import { useDownload } from '../download'
import { fixed } from '../format'

const outcomeUnits: Record<string, string> = {
  season_points: 'season fantasy points', passing_yards: 'season passing yards',
  rushing_yards: 'season rushing yards', receiving_yards: 'season receiving yards',
  attempts: 'season pass attempts', carries: 'season carries', targets: 'season targets',
  receptions: 'season receptions', scoring_appearances: 'scoring appearances per season',
  passing_efficiency: 'yards/attempt', rushing_efficiency: 'yards/carry',
  receiving_efficiency: 'yards/target', season_appearance: 'Brier score (0–1)',
}
const errorValue = (value: unknown, target: string) =>
  fixed(value, target === 'season_appearance' || target.endsWith('_efficiency') ? 3 : 1)

function EvaluationDetails({ result: r }: { result: Evaluation }) {
  const baseline = r.model === 'baseline'
  const unit = outcomeUnits[r.target] ?? r.target.replaceAll('_', ' ')
  return <div className="player-details"><p>Preseason forecast · {r.target.replaceAll('_', ' ')} · {r.position} · {r.population === 'all' ? 'all candidates' : r.population.replaceAll('_', ' ')} · {r.window === 'modern' ? '2019–2025' : '2007–2025'}</p>
    <p><strong>{r.metric === 'Brier score' ? 'Brier score' : 'Average absolute error'}: {errorValue(r.error, r.target)} {r.metric === 'Brier score' ? '' : unit}.</strong> Each evaluated season has equal weight. {r.n.toLocaleString()} matched player-seasons across {r.years} seasons.</p>
    {r.metric !== 'Brier score' && <p>MAE averages the size of each miss, regardless of direction. It is in the outcome’s units, not a percentage or a player prediction interval.</p>}
    {!baseline && <>
      <p>Error reduction versus baseline: {errorValue(r.improvement, r.target)} {unit}; 95% season-bootstrap interval {errorValue(r.ci_low, r.target)} to {errorValue(r.ci_high, r.target)}. This interval describes the average improvement, not a player’s forecast range. Improved seasons: {r.positive_years ?? '—'} / {r.years}.</p>
      <p>Mean squared error reduction: {fixed(r.mse_improvement, 3)} squared outcome units. Worst leave-one-season-out gain: {errorValue(r.loo_min, r.target)} {unit}. Predictions outside earlier training outcome range: {r.outside_training_range ?? '—'}.</p>
    </>}
    <DataTable rows={r.annual ?? []} columns={[
      {key:'season',label:'Season',title:'Held-out year; training uses all earlier candidate years.',initial:'asc'},
      {key:'n',label:'Player-seasons',title:'Candidates with defined observed outcomes and predictions.'},
      {key:'error',label:baseline ? 'Baseline error' : 'Model error',title:`${r.metric}; ${unit}.`,render:v=>errorValue(v.error,r.target)},
      ...(!baseline ? [
        {key:'baseline_error',label:'Baseline error',title:`Same matched candidates; ${unit}.`,render:(v: NonNullable<Evaluation['annual']>[number])=>errorValue(v.baseline_error,r.target)},
        {key:'gain',label:'Error reduction',title:`Baseline error minus model error; ${unit}.`,render:(v: NonNullable<Evaluation['annual']>[number])=>errorValue(v.gain,r.target)},
      ] : []),
    ]} defaultSort="season" rowKey={v=>v.season}/>
    {r.diagnostics && <>
      <h4>Passing-yard error by prior workload</h4>
      <p>{r.diagnostics.zero_outcomes.toLocaleString()} of {r.diagnostics.n.toLocaleString()} evaluated QB seasons recorded zero passing yards. This candidate pool includes backups and players with no season production.</p>
      <p>{r.diagnostics.note}</p>
      <DataTable rows={r.diagnostics.workload_groups} columns={[
        {key:'label',label:'Prior workload',title:'Observed before the forecast season; unknown history is separate.',align:'left'},
        {key:'n',label:'Player-seasons',title:'Matched forecasts in this group.'},
        {key:'years',label:'Seasons',title:'Evaluated seasons represented in this group.'},
        {key:'error',label:'MAE · season yards',title:'Equal-season mean absolute error.',render:v=>fixed(v.error)},
      ]} defaultSort="label" rowKey={v=>v.label}/>
    </>}
    {!!r.calibration?.length && <><h4>Appearance probability calibration</h4><DataTable rows={r.calibration} columns={[
      {key:'lower',label:'Probability bin',title:'Ten fixed probability bins.',render:v=>`${v.lower.toFixed(1)}–${v.upper.toFixed(1)}`},
      {key:'n',label:'Outcomes',title:'Observed outcomes in this bin.'},
      {key:'predicted',label:'Mean predicted',title:'Predicted probability.',render:v=>fixed(v.predicted, 3)},
      {key:'observed',label:'Observed rate',title:'Actual fraction with a scoring appearance; not medical availability.',render:v=>fixed(v.observed, 3)},
    ]} defaultSort="lower" rowKey={v=>v.lower}/></>}
  </div>
}

function IndividualEvidence({ entry }: { entry: Entry }) {
  const { token } = useDataRelease()
  const query = useQuery({ ...individualEvidenceQuery(entry.id.replace('stat:', ''), token), enabled: entry.kind === 'research_stat' })
  return <div className="player-details"><h4>{entry.label}</h4><p>{entry.definition ?? entry.reason}</p>
    <p>{entry.reason}</p><p>Disposition: {entry.serving} · Input validity: {entry.validity} · Outcome: {entry.target ?? 'Descriptive only'}</p>
    {entry.path && <p>Preserved artifact: <code>{entry.path}</code></p>}
    {entry.alias_of && <p>Alias of {entry.alias_of}; not independent supporting evidence.</p>}
    <QueryError query={query} />
    {query.data && <><p>Original preseason fantasy-point screen. These results do not validate other outcomes or serving uses. {query.data.total} matching comparisons.</p>
      <DataTable rows={query.data.comparisons} columns={[
        { key: 'position', label: 'Position', title: 'Tested position.', align: 'left' },
        { key: 'lane', label: 'Baseline', title: 'Own data or market-informed.', align: 'left' },
        { key: 'action', label: 'Test', title: 'Addition or removal; positive gain favors the tested change.', align: 'left' },
        { key: 'mean_mae_gain', label: 'Error reduction', title: 'Season-point MAE; this is not team value.', render: r => fixed(r.mean_mae_gain, 3) },
        { key: 'q_value', label: 'Adjusted q', title: 'Original exhaustive-screen multiple-comparison adjustment.', render: r => fixed(r.q_value, 3) },
        { key: 'evidence', label: 'Finding', title: 'Candidate screening does not grant serving approval.', align: 'left' },
      ]} defaultSort="mean_mae_gain" rowKey={r => `${r.stat}-${r.position}-${r.lane}-${r.action}-${r.context}`} />
    </>}
  </div>
}

/** Page heading and release status shared by analysis, league and research pages. */
export function DashboardLayout() {
  const page = useCurrentPage()
  return <main className="nextgen-workspace">
    <div className="section-heading"><div><span className="eyebrow">{page?.group}</span><h2>{page?.title}</h2>
      <p>{page?.description}</p></div></div>
    <DataReleaseStatus />
    <Outlet />
  </main>
}

/** Analysis pages render once the current catalog is verified. */
export function AnalysisSection() {
  const catalog = useCatalog()
  return <>
    <QueryError query={catalog} label="Current analysis unavailable">. Archived models are not substituted.</QueryError>
    {!catalog.data && !catalog.isError && <p role="status">Verifying the current analysis release…</p>}
    {catalog.data && <>
      <Outlet />
      <p className="legend faint">Analysis {catalog.data.version} · {catalog.data.archived_count} entries excluded from everyday analysis. Preserved in Research.</p>
    </>}
  </>
}

/** Filters shared by the research pages and rookies; each page reads only the ones it shows. */
function useAnalysisFilters() {
  const [search, setSearch] = useUrlState('q', '', { replace: true, resets: ['page'] })
  const [position, setPosition] = useUrlState('position', 'ALL', { resets: ['page'] })
  const [population, setPopulation] = useUrlState('population', 'all', { resets: ['page'] })
  const [target, setTarget] = useUrlState('outcome', 'season_points', { resets: ['page', 'position'] })
  const [window, setWindow] = useUrlState('window', 'all_history')
  const [kind, setKind] = useUrlState('kind', 'all', { resets: ['page'] })
  const [, update] = useUrlParams()
  const clear = () => update({ q: null, position: null, population: null, page: null })
  return { search, setSearch, position, setPosition, population, setPopulation, target, setTarget, window, setWindow, kind, setKind, clear }
}

function AnalysisFilters({ show, searchLabel = 'Search players' }: {
  show: { search?: boolean; population?: boolean; outcome?: boolean; window?: boolean; kind?: boolean }; searchLabel?: string
}) {
  const f = useAnalysisFilters()
  const models = (useCatalog().data?.entries ?? []).filter(e => e.kind === 'forecast')
  return <div className="analysis-controls">
    {show.search && <label>{searchLabel}<input type="search" value={f.search} onChange={e => f.setSearch(e.target.value)} /></label>}
    <label>Position<select value={f.position} onChange={e => f.setPosition(e.target.value)}><PositionOptions /></select></label>
    {show.population && <label>Population<select value={f.population} onChange={e => f.setPopulation(e.target.value)}>{['all','returner','rookie','market_only'].map(p => <option key={p} value={p}>{p.replace('_',' ')}</option>)}</select></label>}
    {show.outcome && <label>Outcome<select aria-label="Outcome" value={f.target} onChange={e => f.setTarget(e.target.value)}>
      {models.map(e => <option key={e.id} value={e.target!}>{e.label.replace(' · baseline','')}</option>)}</select></label>}
    {show.window && <label>Evaluation window<select aria-label="Evaluation window" value={f.window} onChange={e => f.setWindow(e.target.value)}><option value="all_history">Full evaluable history · 2007–2025</option><option value="modern">Modern · 2019–2025</option></select></label>}
    {show.kind && <label>Entry type<select value={f.kind} onChange={e => f.setKind(e.target.value)}>{['all','research_stat','legacy_model','forecast','measurement','data_release','study'].map(k => <option key={k}>{k}</option>)}</select></label>}
    <button type="button" className="button" onClick={f.clear}>Clear filters</button>
  </div>
}

/** `/intelligence/rookies` */
export function IntelligenceRookies() {
  const { search, position } = useAnalysisFilters()
  const page = useUrlPage(100)
  return <>
    <AnalysisFilters show={{ search: true }} />
    <RookieView search={search} position={position} page={page} />
  </>
}

/** `/research/qb-experiments` */
export function ResearchQBExperiments() {
  return <QBVariations research />
}

function EvidenceComparison({ scope }: { scope: 'analysis' | 'research' }) {
  const { token } = useDataRelease()
  const catalog = useCatalog()
  const { target, position, population, window } = useAnalysisFilters()
  const params = new URLSearchParams({ target, position, population, window, scope })
  const evidence = useQuery({ ...evidenceQuery(params, token), enabled: !!catalog.data })
  const columns: Column<Evaluation>[] = [
    { key: 'position', label: 'Position', title: 'Position evaluated.', align: 'left' },
    { key: 'model', label: 'Model', title: 'Baseline or explicitly archived challenger.', align: 'left', render: r => r.model === 'baseline' ? r.target === 'season_appearance' ? 'Historical appearance frequency' : 'Prior-season reference' : r.model },
    { key: 'metric', label: 'Error measure', title: 'MAE for continuous outcomes; Brier score for probabilities.', align: 'left' },
    { key: 'error', label: 'Average error', title: 'Equal-season error; lower is better.', render: r => errorValue(r.error, r.target) },
    { key: 'unit', label: 'Error units', title: 'Season totals, efficiency rates or probability score.', align: 'left', render: r => outcomeUnits[r.target] ?? r.target.replaceAll('_', ' ') },
    { key: 'n', label: 'Player-seasons', title: 'Matched completed player-season outcomes, including backups and nonappearances.' },
    { key: 'years', label: 'Seasons', title: 'Evaluated years; training retains all earlier candidate years.' },
    ...(scope === 'research' ? [{ key: 'improvement', label: 'Gain vs baseline', title: 'Exploratory; not serving approval.', render: (r: Evaluation) => r.model === 'baseline' ? 'Reference' : errorValue(r.improvement, r.target) }] : []),
  ]
  return <>
    {target === 'passing_yards' && <p><Link className="button" to={intelligencePath('qb-passing')}>Open dynamic QB passing forecasts and evidence</Link></p>}
    <p className="legend"><strong>Preseason forecast accuracy.</strong> These forecasts use information available before each season. Season totals include later absences and role changes; efficiency outcomes measure production per opportunity. Current rest-of-season rankings have a separate evaluation.</p>
    <p className="legend">{target === 'season_appearance'
      ? 'The reference predicts appearance probability from earlier position/population frequencies. Brier score measures squared probability error: 0 is perfect, lower is better.'
      : 'The reference repeats the prior-season observation, adjusting season totals for schedule length. Without a prior observation, it uses an earlier position/population average. MAE is mean absolute error in the displayed units; lower is better.'}</p>
    <p className="legend">Every fit uses all earlier completed candidate years. Modern is an evaluation slice, not a shorter training window. Yardage, appearances and fantasy points have separate tests.</p>
    <QueryError query={evidence} />
    {evidence.data && <DataTable rows={evidence.data.comparisons} columns={columns} defaultSort="position" sortParam="sort" rowKey={r=>r.position+r.model} renderDetails={r=><EvaluationDetails result={r}/>} emptyMessage="No evaluated outcomes for this scope." />}
  </>
}

/** `/research/evidence` */
export function ResearchEvidence() {
  return <>
    <AnalysisFilters show={{ population: true, outcome: true, window: true }} />
    <EvidenceComparison scope="analysis" />
  </>
}

/** `/research/forecasts` */
export function ReferenceForecasts() {
  const { token } = useDataRelease()
  const catalog = useCatalog()
  const { target, position, population, search } = useAnalysisFilters()
  const page = useUrlPage(100)
  const exporter = useDownload()
  const settledSearch = useDebounced(search)
  const forecastParams = new URLSearchParams({ target, position, population, search: settledSearch })
  const forecasts = useQuery({ ...forecastsQuery(forecastParams, token), enabled: !!catalog.data })
  // The CSV endpoint pages at 1,000 rows; later pages drop their repeated header line.
  const exportForecasts = () => exporter.download(`nextgen-${target}.csv`, async () => {
    const params = new URLSearchParams(forecastParams)
    params.set('format', 'csv'); params.set('limit', '1000')
    const chunks: string[] = []
    for (let start = 0; start < Math.max(1, forecasts.data?.total ?? 0); start += 1000) {
      params.set('offset', String(start))
      const csv = await (await getResponse(`/api/nextgen/forecasts?${params}`, token)).text()
      chunks.push(start === 0 ? csv : csv.slice(csv.indexOf('\n') + 1))
    }
    return new Blob(chunks, { type: 'text/csv' })
  })
  return <>
    <AnalysisFilters show={{ search: true, population: true, outcome: true }} />
    <p className="legend">Reference forecasts are research benchmarks, not a Next Gen ranking or rest-of-season recommendation.</p>
    <QueryError query={forecasts} />{forecasts.isFetching && <p role="status">Loading baseline estimates…</p>}
    {exporter.error && <p role="alert">{exporter.error}</p>}{forecasts.data && <><button type="button" className="button" onClick={() => void exportForecasts()}>Export eligible estimates</button><p>{forecasts.data.entry.reason}</p><DataTable<Forecast> rows={forecasts.data.forecasts} pagination={page} sortParam="sort" columns={[
      { key:'player_display_name',label:'Player',title:'Player name.',align:'left' }, {key:'position',label:'Position',title:'Forecast position.',align:'left'},
      ...(target === 'season_points' ? consensusColumns<Forecast>() : []),
      {key:'prediction',label:'Baseline estimate',title:`${forecasts.data.entry.unit}; no calibrated uncertainty claim.`,render:r=>fixed(r.prediction, r.unit==='probability'?3:1)},
      {key:'fallback_source',label:'Reference source',title:'Prior observation or an earlier training-group mean; unknown history is explicit.',align:'left',render:r=>r.fallback_source.replaceAll('_',' ')},
      {key:'prior_observed_weeks',label:'Prior weeks',title:'Weeks in the prior-season scoring record, including backup appearances.'},
      {key:'unit',label:'Units',title:'Outcome units.',align:'left'}, {key:'season',label:'Season',title:'Forecast season.'},
      {key:'cutoff',label:'Forecast cutoff',title:'Saved preseason information cutoff; current news is not retroactively applied.',align:'left'},
    ]} defaultSort="prediction" rowKey={r=>r.player_id} rowLabel={r=>r.player_display_name} profileHref={r=>playerPath(r.player_id)} emptyMessage="No eligible forecasts for this position and outcome." />
      <Pager page={page} total={forecasts.data.total} /></>}
  </>
}

/** `/research/archive`: superseded evidence, the model inventory, and links to the archived workspaces. */
export function ResearchArchive() {
  const { token } = useDataRelease()
  const catalog = useCatalog()
  const { search, kind } = useAnalysisFilters()
  const page = useUrlPage(100)
  const registryParams = new URLSearchParams({ scope: 'research', kind, search })
  const registry = useQuery({ ...registryQuery(registryParams, token), enabled: !!catalog.data })
  return <>
    <div className="archive-banner" role="note"><strong>Research archive — excluded from everyday analysis</strong>
      <p>Old input vintages, superseded models and unapproved experiments are retained for inspection. Their scores and claims do not affect the current analysis. Each entry records its status and reason.</p></div>
    <AnalysisFilters show={{ search: true, population: true, outcome: true, window: true, kind: true }} searchLabel="Search research" />
    <EvidenceComparison scope="research" />
    <h3>Evidence and model inventory</h3><QueryError query={registry} />
    {registry.data && <><DataTable rows={registry.data.entries} pagination={page} sortParam="inventory_sort" columns={[
      {key:'label',label:'Item',title:'Preserved definition or model.',align:'left'}, {key:'serving',label:'Disposition',title:'Only approved uses enter analysis.',align:'left'},
      {key:'validity',label:'Input status',title:'Invalid or older inputs require revalidation.',align:'left'}, {key:'evidence',label:'Evidence',title:'Research screening is not serving approval.',align:'left'},
      {key:'reason',label:'Reason',title:'Why this use is included or excluded.',align:'left'},
    ]} defaultSort="label" rowKey={r=>r.id} rowLabel={r=>r.label} renderDetails={r=><IndividualEvidence entry={r}/>} />
      <Pager page={page} total={registry.data.total} /></>}
    <section className="reference-section"><h4>Older model runs and contaminated input vintages</h4>
      <p className="archive-banner">ARCHIVED: these runs retain their original data, including known timing or target-count defects. “Accepted” in an old report refers to its original checks, not current approval.</p>
      <p><Link to={archivePath('intelligence')}>Open the archived intelligence workspace →</Link></p></section>
    <section className="reference-section"><h4>Legacy league projections and decision experiments</h4>
      <p className="archive-banner">ARCHIVED: old power rankings, matchup simulations and waiver/draft valuations are unapproved research scenarios.</p>
      <p><Link to={archivePath('league')}>Open the archived league workspace →</Link></p></section>
  </>
}
