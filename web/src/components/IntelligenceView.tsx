import { PlayerProfiles, PlayerProfileLink } from './PlayerProfile'
import { NavLink, Outlet, useParams } from 'react-router'
import { CollegePathways } from './CollegePathways'
import { useQuery } from '@tanstack/react-query'
import type { IntelligenceTeam, ResearchPlayer, Status } from '../api/types'
import { fetchLeaguePlayers, fetchLeagueImpact, fetchResearchCatalog, fetchResearchPlayers, fetchResearchSources } from '../api/client'
import { PLAYER_MODELS, researchModelLabel, researchModelDescription } from '../researchModels'
import { DataTable, type Column } from './DataTable'
import { MetricReportView } from './MetricReportView'
import { OwnerBadge } from './Availability'
import { ReferenceBoard } from './ReferenceBoard'
import { PlayerOutlook } from './PlayerOutlook'
import { DataReleaseProvider, DataReleaseStatus } from './DataRelease'
import { useDataRelease } from '../dataRelease'
import { archiveGet } from '../api/client'
import { ARCHIVED_VIEWS, archivePath, useUrlFlag, useUrlNumber, useUrlParams, useUrlState } from '../navigation'

const number = (value: number | null | undefined) => value == null ? '—' : value.toFixed(1)
const rank = (value: number | null) => value == null ? '—' : `#${value}`
const teamColumns: Column<IntelligenceTeam>[] = [
  { key: 'scenario_rank', label: 'Scenario rank', title: 'Power ordering under this model and these assumptions.', initial: 'asc', render: r => rank(r.scenario_rank) },
  { key: 'team_name', label: 'Team', title: 'Current Sweaty Plays roster.', align: 'left' },
  { key: 'baseline_rank', label: 'Production rank', title: 'The same legal lineups evaluated with the production forecast.', initial: 'asc', render: r => rank(r.baseline_rank) },
  { key: 'rank_change', label: 'Rank change', title: 'Positive means the team rises under this scenario.', render: r => `${r.rank_change > 0 ? '+' : ''}${r.rank_change}` },
  { key: 'wins', label: 'Record', title: 'Actual league wins and losses, independent of the scenario.', render: r => `${r.wins}–${r.losses}` },
  { key: 'scenario_value', label: 'Lineup value', title: 'Best legal lineup on the scenario value scale.', render: r => number(r.scenario_value) },
  { key: 'value_change', label: 'Value change', title: 'Scenario lineup value minus production lineup value.', render: r => number(r.value_change) },
  { key: 'modeled_players', label: 'Modeled', title: 'Rostered skill players with usable research forecasts.' },
  { key: 'fallback_players', label: 'Fallbacks', title: 'Players using the declared production fallback.' },
  { key: 'missing_players', label: 'Missing', title: 'Players excluded because they have no usable scenario score.' },
]

const SECTION_LABELS: Record<typeof ARCHIVED_VIEWS.intelligence[number], string> = { players: 'Players', 'league-impact': 'League impact', research: 'Research' }
const archivedStatus = () => archiveGet<Status>('/api/status?scope=research')

/** Layout for `/research/archive/intelligence/*`: the archived status and section links. */
export function ArchivedIntelligence() {
  const status = useQuery({ queryKey: ['archived-status'], queryFn: archivedStatus, retry: false })
  return <>
    <p className="archive-banner">ARCHIVED: these runs retain their original data, including known timing or target-count defects. “Accepted” in an old report refers to its original checks, not current approval.</p>
    {status.isError && <p role="alert">{status.error.message}</p>}
    {status.data && <DataReleaseProvider><section className="intelligence-workspace">
      <div className="section-heading"><div><span className="eyebrow">Analysis workspace</span>
        <h2>Intelligence</h2><p>Find players to investigate. Check current opportunity. Compare the historical evidence.</p></div></div>
      <DataReleaseStatus />
      <nav className="subnav" aria-label="Archived intelligence views">
        {ARCHIVED_VIEWS.intelligence.map(id => <NavLink key={id} className="subtab" to={archivePath('intelligence', id)} preventScrollReset>{SECTION_LABELS[id]}</NavLink>)}
      </nav>
      <Outlet />
    </section></DataReleaseProvider>}
  </>
}

/** `/research/archive/intelligence/:view` */
export function ArchivedIntelligenceView() {
  const { view } = useParams()
  const status = useQuery({ queryKey: ['archived-status'], queryFn: archivedStatus, retry: false })
  if (!status.data) return null
  return <IntelligenceWorkspace status={status.data} section={view === 'league-impact' ? 'league' : view === 'research' ? 'research' : 'players'} />
}

const DEFAULT_WEIGHTS = { QB: 1, RB: 1, WR: 1, TE: 1 }

function IntelligenceWorkspace({ status, section }: { status: Status; section: 'players' | 'league' | 'research' }) {
  const { token: dataRelease } = useDataRelease()
  const [params, update] = useUrlParams()
  const [playerView, setPlayerView] = useUrlState('view', 'profiles')
  const [playerGroup] = useUrlState('group', 'all')
  const [researchView, setResearchView] = useUrlState('topic', 'evidence')
  const tab = section === 'league' ? 'league' : section === 'research' ? researchView
    : playerGroup === 'college' ? 'college' : playerView
  const savedResearch = ['league', 'players', 'mispricing', 'evidence'].includes(tab)
  const comparisonView = tab === 'players' || tab === 'mispricing'
  const [model, setModel] = useUrlState('model', 'fitted_season_points')
  const [season, setSeason] = useUrlNumber('season', status.league.draft_season)
  const [position, setPosition] = useUrlState('position', 'ALL', { resets: ['page'] })
  const [researchPopulation] = useUrlState('population', 'all')
  const population = section === 'players' ? playerGroup : researchPopulation
  const [reality, setReality] = useUrlState('reality', 'saved')
  const [target, setTarget] = useUrlState('outcome', 'actual_season_points')
  const [benchmark, setBenchmark] = useUrlState('benchmark', 'production')
  const [coverage, setCoverage] = useUrlState('coverage', 'all')
  const [pricingBasis, setPricingBasis] = useUrlState('compare', 'espn')
  const [pricingDirection, setPricingDirection] = useUrlState('direction', 'all')
  const [minimumGap, setMinimumGap] = useUrlNumber('gap', 10, { replace: true })
  const [ownership, setOwnership] = useUrlState('roster', 'all')
  const owners = useQuery({ queryKey: ['league-players', 'v2'], queryFn: () => fetchLeaguePlayers('v2'), retry: false })
  const ownerById = new Map(owners.data?.players.map(p => [p.player_id, p]))
  const [search, setSearch] = useUrlState('q', '', { replace: true, resets: ['page'] })
  const [missing, setMissing] = useUrlState('missing', 'production')
  // Position weights are stored as whole percentages (`qb_weight=120`); 100% is the default.
  const weights = Object.fromEntries((Object.keys(DEFAULT_WEIGHTS) as (keyof typeof DEFAULT_WEIGHTS)[]).map(pos => {
    const percent = Number(params.get(`${pos.toLowerCase()}_weight`))
    return [pos, params.has(`${pos.toLowerCase()}_weight`) && Number.isFinite(percent) ? percent / 100 : 1]
  })) as typeof DEFAULT_WEIGHTS
  const [dataset, setDataset] = useUrlState('dataset', '')
  const [showAllModels, setShowAllModels] = useUrlFlag('all_models')
  // Rookie populations have no core-model forecasts, so choosing one also switches the model.
  const choosePopulation = (key: 'group' | 'population', value: string) => update({ [key]: value === 'all' ? null : value, page: null,
    ...(value === 'rookie' && ['fitted_season_points', 'fitted_ppg'].includes(model) ? { model: 'fitted_nextgen_season_points' } : {}) })
  const sources = useQuery({ queryKey: ['research-sources', dataRelease], queryFn: fetchResearchSources, enabled: savedResearch, retry: false })
  const activeDataset = dataset || sources.data?.default_id || ''
  const source = sources.data?.datasets.find(entry => entry.id === activeDataset)
  const catalog = useQuery({ queryKey: ['research-catalog', activeDataset],
    queryFn: () => fetchResearchCatalog(activeDataset), enabled: savedResearch && !!activeDataset, retry: false })
  const activeSeason = tab === 'league' || reality === 'current' ? status.league.draft_season : season
  const availableModels = catalog.data?.models.filter(m => m.seasons.includes(activeSeason)
    && (tab !== 'league' || ['season points', 'active-game PPG'].includes(m.unit))
    && (showAllModels || m.id in PLAYER_MODELS)) ?? []
  const activeModel = availableModels.find(m => m.id === model) ?? availableModels[0]
  const cutoffs = catalog.data?.seasons.find(entry => entry.season === activeSeason)?.cutoff_dates ?? []
  const impactParams = new URLSearchParams({ dataset: activeDataset, model: activeModel?.id ?? '', missing,
    ...Object.fromEntries(Object.entries(weights).map(([key, value]) => [key.toLowerCase(), String(value)])) })
  const impact = useQuery({ queryKey: ['league-impact', impactParams.toString()],
    queryFn: () => fetchLeagueImpact(impactParams), enabled: tab === 'league' && !!activeModel, retry: false })
  const overallEspnComparison = tab === 'mispricing' && pricingBasis === 'espn'
  const playerParams = new URLSearchParams({ dataset: activeDataset, model: activeModel?.id ?? '', position: overallEspnComparison ? 'ALL' : position,
    population: overallEspnComparison ? 'all' : population,
    ...(reality === 'current' ? { benchmark: tab === 'mispricing' ? 'market' : 'production' } : { season: String(activeSeason), target, benchmark: tab === 'mispricing' ? 'market' : benchmark }) })
  const players = useQuery({ queryKey: ['research-players', reality, playerParams.toString()],
    queryFn: () => fetchResearchPlayers(playerParams, reality === 'current'),
    enabled: comparisonView && !!activeModel, retry: false })
  const visiblePlayers = (players.data?.players ?? []).filter(row =>
    (!overallEspnComparison || ((position === 'ALL' || row.position === position) &&
      (population === 'all' || row.population === population))) &&
    row.player_display_name.toLowerCase().includes(search.trim().toLowerCase()) &&
    (coverage === 'all' || (row.model_value != null) === (coverage === 'scored')) &&
    (ownership === 'all' || (ownership === 'mine' ? ownerById.get(row.player_id)?.is_mine
      : ownerById.get(row.player_id)?.availability === ownership)))
  const mismatchRank = (row: ResearchPlayer) => pricingBasis === 'actual' ? row.actual_rank
    : pricingBasis === 'espn' ? ownerById.get(row.player_id)?.espn_draft_rank ?? null : row.benchmark_rank
  const mismatch = (row: ResearchPlayer) => {
    const reference = mismatchRank(row)
    return reference == null || row.model_rank == null ? null : reference - row.model_rank
  }
  const mismatchedPlayers = visiblePlayers.filter(row => {
    const gap = mismatch(row)
    return gap != null && Math.abs(gap) >= minimumGap &&
      (pricingDirection === 'all' || (pricingDirection === 'higher' ? gap > 0 : gap < 0))
  })
  const referenceLabel = pricingBasis === 'actual' ? 'Actual rank' : pricingBasis === 'espn' ? 'ESPN rank' : 'Market rank'
  const playerColumns: Column<ResearchPlayer>[] = [
    { key: 'model_rank', label: 'Research rank', title: 'Raw scoring rank, not positional value over replacement or current trade value. Prefer a position filter. Search does not renumber ranks.', initial: 'asc', render: r => rank(r.model_rank) },
    { key: 'player_display_name', label: 'Player', title: 'Player name.', align: 'left' },
    { key: 'ownership', label: 'Current roster', title: 'Current league ownership, including when viewing a past research season.', align: 'left',
      value: r => ownerById.get(r.player_id)?.owner_team_name,
      render: r => { const owner = ownerById.get(r.player_id); return owner ? <OwnerBadge player={owner} /> : <span className="faint">Unknown</span> } },
    { key: 'position', label: 'Pos', title: 'Position.', align: 'left' },
    { key: 'current_health', label: 'Current health', title: 'Latest saved league health status—not historical status or an input update to the preseason forecast.', align: 'left',
      value: r => ownerById.get(r.player_id)?.injury_status,
      render: r => ownerById.get(r.player_id)?.injury_status ?? 'Unknown' },
    { key: 'actual_rank', label: 'Actual rank', title: 'Rank by observed fantasy production, including players missing research scores.', initial: 'asc', render: r => rank(r.actual_rank) },
    { key: 'rank_gap', label: 'Rank gap', title: 'Actual rank minus research rank. Positive means research ranked the player higher.', render: r => r.rank_gap == null ? '—' : `${r.rank_gap > 0 ? '+' : ''}${r.rank_gap}` },
    { key: 'model_value', label: 'Research score', title: `Saved model output in ${activeModel?.unit ?? 'model units'}.`, render: r => number(r.model_value) },
    { key: 'actual_value', label: 'Actual result', title: reality === 'current' ? 'Observed ESPN points to date; this can be partial.' : target === 'actual_ppg' ? 'Observed active-game fantasy PPG.' : 'Observed fantasy season points.', render: r => number(r.actual_value) },
    { key: 'benchmark_rank', label: 'Benchmark rank', title: reality !== 'current' && benchmark === 'market' ? 'Market ECR order in the selected pool.' : 'Production season-points order in the same saved fold.', initial: 'asc', render: r => rank(r.benchmark_rank) },
    { key: 'population', label: 'Population', title: 'Returning player, rookie, or market-only candidate.', align: 'left' },
  ]

  const mispricingColumns: Column<ResearchPlayer>[] = [
    { key: 'mismatch', label: 'Mismatch', title: 'Absolute difference between research rank and comparison rank. Largest disagreements first.',
      value: r => { const gap = mismatch(r); return gap == null ? null : Math.abs(gap) },
      render: r => { const gap = mismatch(r); return gap == null ? '—' : Math.abs(gap) } },
    ...playerColumns.slice(0, 4),
    { key: 'reference_rank', label: referenceLabel, title: 'Rank from the selected comparison source. Unknown ranks are excluded.',
      value: mismatchRank, render: r => rank(mismatchRank(r)), initial: 'asc' },
    { key: 'direction', label: 'Research view', title: 'Positive gap means research places the player higher than the comparison source.',
      align: 'left', value: mismatch,
      render: r => { const gap = mismatch(r); return gap == null ? '—' : gap > 0 ? 'Higher by ' + gap : gap < 0 ? 'Lower by ' + (-gap) : 'Aligned' } },
    ...playerColumns.filter(c => c.key === 'model_value' || c.key === 'actual_value'),
    ...playerColumns.filter(c => c.key === 'current_health'),
  ]

  const filters = { search, position, population }

  return <>
    <span className="instrument-tag">{section === 'players' ? 'Career history & current opportunity' : section === 'league' ? 'Roster scenarios' : 'Models & evidence'}</span>
    {section === 'players' && <div className="analysis-controls player-browser-controls">
      <label>Search players<input type="search" value={search} onChange={e => setSearch(e.target.value)} placeholder="Player name…" /></label>
      <label>Position<select aria-label="Position" value={position} onChange={e => setPosition(e.target.value)}>{['ALL', 'QB', 'RB', 'WR', 'TE'].map(p => <option key={p}>{p}</option>)}</select></label>
      <label>Population<select aria-label="Population" value={playerGroup} onChange={e => choosePopulation('group', e.target.value)}><option value="all">All NFL players</option><option value="rookie">Rookies</option><option value="returner">Returners</option><option value="college">College → NFL</option></select></label>
      <label>Player view<select aria-label="Player view" value={playerGroup === 'college' && playerView === 'outlook' ? 'players' : playerView} onChange={e => setPlayerView(e.target.value)}>
        <option value="profiles">{playerGroup === 'college' ? 'College careers' : 'Career history'}</option>
        {playerGroup !== 'college' && <option value="outlook">Four-week outlook</option>}
        <option value="players">{playerGroup === 'college' ? 'NFL translation forecasts' : 'Preseason rankings'}</option>
      </select></label>
    </div>}
    {section === 'research' && <div className="analysis-controls">
      <label>Research topic<select value={researchView} onChange={e => setResearchView(e.target.value)}>
        <option value="evidence">Model performance</option><option value="mispricing">Market comparisons</option>
        <option value="college-evidence">College backtests</option><option value="college-identities">College identity audit</option>
      </select></label>
    </div>}
    {tab === 'profiles' && <PlayerProfiles filters={filters} />}
    {tab === 'outlook' && <PlayerOutlook filters={filters} />}
    {tab === 'college' && <CollegePathways view={playerView === 'profiles' ? 'identities' : 'players'} filters={filters} />}
    {(tab === 'college-evidence' || tab === 'college-identities') && <CollegePathways view={tab === 'college-evidence' ? 'evidence' : 'identities'} audit />}
    {savedResearch && <>
      {sources.isError && <p className="notice">Could not load research datasets: {sources.error.message}</p>}
      {sources.isLoading && <p role="status">Checking accepted research datasets…</p>}
      {sources.data && !sources.data.datasets.length && <p className="notice">No accepted or legacy research datasets are available.</p>}
      <div className="analysis-controls">
        <label>Saved model run<select aria-label="Research dataset" value={activeDataset} onChange={e => setDataset(e.target.value)}>
          {sources.data?.datasets.map(entry => <option key={entry.id} value={entry.id}>{entry.label}</option>)}
        </select></label>
        {source && <span className="control-note">{source.accepted ? `${source.checks_passed} integrity checks passed · ${source.forecast_rows?.toLocaleString()} forecast rows` : 'Legacy · not repair-audited'}</span>}
      </div>
      {source && <p className="legend">{source.description}</p>}
      <p className="legend">These are archived model results. Selecting a run changes this comparison; player histories and current outlooks continue to use the canonical data release. Saved runs have not been refit with the gold pipeline’s market timing corrections.</p>
      {sources.data?.unavailable.map(entry => <p className="notice" key={entry.id}>{entry.reason}</p>)}
    </>}
    {(tab === 'league' || comparisonView) && <>
      {catalog.isError && <div className="notice">{catalog.error.message}</div>}
      {!catalog.data && !catalog.isError && <p className="notice">Loading research models…</p>}
      <div className="analysis-controls">
        <label>Research model<select aria-label="Research model" value={activeModel?.id ?? ''} onChange={e => setModel(e.target.value)}>
          {availableModels.map(m => <option key={m.id} value={m.id}>{researchModelLabel(m.id)}</option>)}
        </select></label>
        <label>Model list<select aria-label="Model list" value={showAllModels ? 'all' : 'player'} onChange={e => setShowAllModels(e.target.value === 'all')}>
          <option value="player">Player forecasts</option><option value="all">All research outputs (advanced)</option>
        </select></label>
        <div className="control-note">{activeModel?.unit ?? 'No model for this season'}
          {cutoffs.length > 0 ? <> · preseason inputs through {cutoffs.join(', ')}</> : ' · input cutoff not recorded'}
          {catalog.data && <> · file saved {new Date(catalog.data.saved_at).toLocaleDateString()}</>}</div>
      </div>
      <p className="legend">{activeModel && researchModelDescription(activeModel.id)}</p>
      <p className="notice">Saved preseason forecasts—not rest-of-season projections. For trades and pickups, check current usage, health, remaining schedule, and league availability. Raw points ranks are not cross-position trade values.</p>
      <details className="reference-section"><summary>Which numbers should I use?</summary>
        <p>For a player shortlist, start with Core season forecast and filter by position and current roster. Use Next-gen for rookie preseason comparisons. Active-game PPG separates scoring strength from availability.</p>
        <p>For pickups, use current usage and availability; Players → Four-week outlook can be filtered to rookies. For trades, compare roster needs and current opportunity—this workspace does not yet provide validated rest-of-season trade values.</p>
        <p>Expected games, return probability, direct/two-stage variants, full-stack, career, athletic, and security outputs remain in the advanced list for diagnosis. More complex does not mean better. Adaptive chooses a component; it is not a consensus-beating master model.</p>
      </details>
    </>}
    {tab === 'league' && <>
      <div className="assumption-panel"><div><h3>Positional assumptions</h3><p>Adjust forecast strength by position. 100% keeps the model unchanged.</p></div>
        <div className="analysis-controls">{(['QB', 'RB', 'WR', 'TE'] as const).map(pos => <label key={pos}>{pos} weight (%)
          <input type="number" min="0" max="200" step="5" value={Math.round(weights[pos] * 100)}
            onChange={e => { const n = Number(e.target.value); if (Number.isFinite(n)) { const percent = Math.max(0, Math.min(200, n)); update({ [`${pos.toLowerCase()}_weight`]: percent === 100 ? null : percent }, { replace: true }) } }} />
        </label>)}
          <label>Missing forecasts<select aria-label="Missing forecasts" value={missing} onChange={e => setMissing(e.target.value)}>
            <option value="production">Use production fallback</option><option value="exclude">Leave unscored</option>
          </select></label>
          <button type="button" className="button" onClick={() => update({ qb_weight: null, rb_weight: null, wr_weight: null, te_weight: null, missing: null })}>Reset assumptions</button>
        </div>
      </div>
      {impact.isError && <div className="notice">{impact.error.message}</div>}
      {impact.isFetching && <p role="status">Recalculating roster values…</p>}
      {impact.data && <><p className="legend">{impact.data.basis} {impact.data.stale && 'League snapshot is stale.'}</p>
        <DataTable rows={impact.data.teams} columns={teamColumns} defaultSort="scenario_rank" rowKey={r => r.team_id}
          rowLabel={r => r.team_name} emptyMessage="No current rosters are available."
          renderDetails={r => <div className="player-details"><h3>{r.team_name} · scenario starters</h3>
            {r.starters.map((s, i) => <p key={i}>{s.slot} · {s.name ?? 'Unfilled'} · {number(s.value)}</p>)}</div>} />
      </>}
    </>}
    {comparisonView && <>
      {tab === 'mispricing' && <div className="assumption-panel">
        <h3>Ranking disagreements—not current trade prices</h3>
        <p>Use these differences to choose players to investigate. Neither preseason consensus nor ESPN draft-room order measures today’s trade price or waiver cost.</p>
        <div className="analysis-controls">
          <label>Compare against<select aria-label="Compare against" value={pricingBasis} onChange={e => setPricingBasis(e.target.value)}>
            <option value="espn">Current ESPN draft-room rank</option>
            <option value="market">Market consensus ECR (saved season)</option>
            <option value="actual">Actual results</option>
          </select></label>
          <label>Research view<select aria-label="Research view" value={pricingDirection} onChange={e => setPricingDirection(e.target.value)}>
            <option value="all">Higher and lower</option><option value="higher">Research ranks higher</option><option value="lower">Research ranks lower</option>
          </select></label>
          <label>Minimum rank gap<input aria-label="Minimum rank gap" type="number" min="0" step="1" value={minimumGap}
            onChange={e => setMinimumGap(Math.max(0, Math.floor(Number(e.target.value) || 0)))} /></label>
        </div>
        <p className="legend">{pricingBasis === 'espn'
          ? 'ESPN is the current overall PPR draft-room ordering, not a live performance ranking. Position and population filters keep both overall ranks unchanged. Past research seasons still compare against current ESPN ranks.'
          : pricingBasis === 'market' ? 'Market consensus is ECR from the selected saved season, ranked within the selected pool; it is not an official results ranking.'
          : 'Actual rank is based on the selected outcome source. Pending or unobserved results cannot produce a mismatch.'}
          {' '}Players missing either comparison rank are excluded.</p>
      </div>}
      <div className="analysis-controls">
        <label>Reality source<select aria-label="Reality source" value={reality} onChange={e => setReality(e.target.value)}>
          <option value="saved">Saved season outcomes</option><option value="current">Current league observations</option>
        </select></label>
        {reality === 'saved' && <><label>Season<select aria-label="Season" value={season} onChange={e => setSeason(Number(e.target.value))}>
          {catalog.data?.seasons.map(s => <option key={s.season} value={s.season}>{s.season} · {s.complete ? 'completed' : 'pending'}</option>)}
        </select></label><label>Actual outcome<select aria-label="Actual outcome" value={target} onChange={e => setTarget(e.target.value)}>
          <option value="actual_season_points">Season points</option><option value="actual_ppg">Active-game PPG</option>
        </select></label>{tab !== 'mispricing' && <label>Benchmark<select aria-label="Benchmark" value={benchmark} onChange={e => setBenchmark(e.target.value)}>
          <option value="production">Production (same saved fold)</option><option value="market">Market ECR</option>
        </select></label>}</>}
        {section === 'research' && <><label>Population<select aria-label="Population" value={population} onChange={e => choosePopulation('population', e.target.value)}>
          <option value="all">All players</option><option value="returner">Returners</option><option value="rookie">Rookies</option><option value="market_only">Market only</option>
        </select></label>
        <label>Position<select aria-label="Position" value={position} onChange={e => setPosition(e.target.value)}>
          {['ALL', 'QB', 'RB', 'WR', 'TE'].map(p => <option key={p}>{p}</option>)}
        </select></label>
        </>}
        <label>Coverage<select aria-label="Coverage" value={coverage} onChange={e => setCoverage(e.target.value)}>
          <option value="all">All players</option><option value="scored">Model scored</option><option value="missing">Missing forecast</option>
        </select></label>
        <label>Current roster<select aria-label="Current roster" value={ownership} onChange={e => setOwnership(e.target.value)} disabled={!owners.data}>
          <option value="all">Everyone</option><option value="mine">My players</option>
          <option value="rostered">Rostered</option><option value="free_agent">Available</option>
        </select></label>
        {section === 'research' && <label>Search players<input type="search" value={search} onChange={e => setSearch(e.target.value)} placeholder="Name…" /></label>}
      </div>
      {owners.isError && <p className="legend">Current roster ownership is unavailable; player rankings remain available.</p>}
      {owners.data && <p className="legend faint">Roster labels reflect the current league snapshot, even for historical seasons. Your players are highlighted.{owners.data.stale && ' Ownership snapshot is stale.'}</p>}
      {players.isError && <div className="notice">{players.error.message}</div>}
      {players.isFetching && <p role="status">Loading player comparisons…</p>}
      {players.data && <><p className="legend">{players.data.outcome_status}</p>
        <p className="legend faint">{players.data.scored_players} / {players.data.pool_size} model-scored · {players.data.ranking_basis}</p>
        {reality === 'current' && <p className="legend">Season-total forecasts and partial points to date cover different horizons. Their rank difference is not forecast error or proof of a buying opportunity.</p>}
        <DataTable key={`${activeDataset}-${tab}-${activeModel?.id}-${activeSeason}-${position}-${population}`} rows={tab === 'mispricing' ? mismatchedPlayers : visiblePlayers} columns={tab === 'mispricing' ? mispricingColumns : playerColumns}
          defaultSort={tab === 'mispricing' ? 'mismatch' : 'model_rank'} rowKey={r => r.player_id} rowClass={r => ownerById.get(r.player_id)?.is_mine ? 'mine-row' : undefined} emptyMessage="No players match these filters."
          rowLabel={r => r.player_display_name}
          renderDetails={r => <><PlayerProfileLink playerId={r.player_id} /><details className="player-details"><summary>Selected forecast context</summary>
            <p>Saved preseason cutoff: {r.historical_context?.cutoff_date ?? 'Not recorded'}. Historical roster evidence: {r.historical_context?.roster_evidence?.replaceAll('_', ' ') ?? 'Unknown'}; historical status: {r.historical_context?.roster_status ?? 'Unknown'}.</p>
            <p>Prior-season sample: {number(r.historical_context?.prior_games)} games · {number(r.historical_context?.prior_ppg)} active-game PPG · observed offensive snap share {r.historical_context?.prior_snap_share == null ? 'unknown' : `${(100 * r.historical_context.prior_snap_share).toFixed(0)}%`}.</p>
            <p>Current saved health: {ownerById.get(r.player_id)?.injury_status ?? 'Unknown'}. Verify latest news and role before acting; current ownership and health do not update this preseason forecast.</p>
          </details></>} />
      </>}
    </>}
    {tab === 'evidence' && <><MetricReportView key={activeDataset} available={!!source?.report_available} dataset={activeDataset || undefined} source={source} />
      <details className="reference-section"><summary>Original preseason draft list</summary>
        <p>The list we saved before the draft, including manual adjustments. Kept as a reference for checking how our analysis changed.</p>
        {status.metric_versions.v1?.available && <ReferenceBoard version="v1" />}
      </details></>}
  </>
}
