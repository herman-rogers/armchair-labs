import { RankMovement } from './RankMovement'
import { PlayerRankHistory } from './PlayerRankHistory'
import { rankingHistoryQuery, leagueQuery, rankingsQuery, rankingEvidenceQuery } from '../api/queries'
import { Link } from 'react-router'
import { intelligencePath, playerPath, researchPath, to, useUrlPage, useUrlParams, useUrlState } from '../navigation'
import { useQuery } from '@tanstack/react-query'
import { getResponse } from '../api/client'
import type { Ranking } from '../api/nextgen'
import { useDataRelease } from '../dataRelease'
import { DataTable, type Column } from './DataTable'
import { consensusColumns } from './RankComparison'
import { AttentionBadges, PlayerStatus } from './LeagueAttention'
import { Pager, PositionOptions, QueryError } from './Controls'
import { useDownload } from '../download'
import { fixed } from '../format'

/** `/intelligence/rankings` */
export function NextGenRankings() {
  const { token } = useDataRelease()
  const [horizon, setHorizon] = useUrlState('horizon', 'rest_of_season', { resets: ['page'] })
  const [position, setPosition] = useUrlState('position', 'ALL', { resets: ['page'] })
  const [search, setSearch] = useUrlState('q', '', { replace: true, resets: ['page'] })
  const [ownership, setOwnership] = useUrlState('pool', 'all', { resets: ['page'] })
  const [, update] = useUrlParams()
  const page = useUrlPage(50)
  const exporter = useDownload()
  const params = new URLSearchParams({ horizon, position, search })
  const query = useQuery(rankingsQuery(horizon, token))
  const historyQuery = useQuery(rankingHistoryQuery(horizon, token))
  const history = historyQuery.data?.version === query.data?.version ? historyQuery.data : undefined
  const previous = history?.snapshots.at(-1)
  const priorRanks = new Map(previous?.rankings.map(r => [r.player_id, r]))
  const league = useQuery(leagueQuery(token))
  const evidence = useQuery(rankingEvidenceQuery(new URLSearchParams({ horizon, position }), token))
  const owners = new Map(league.data?.players.filter(p => p.player_id).map(p => [p.player_id, p]))
  const rows = (query.data?.rankings ?? []).filter(r => (position === 'ALL' || r.position === position) && r.player_display_name.toLowerCase().includes(search.toLowerCase())).filter(r => ownership === 'all' ||
    (ownership === 'free' ? owners.get(r.player_id)?.availability === 'free_agent' : owners.get(r.player_id)?.is_mine))
  const columns: Column<Ranking>[] = [
    { key: 'overall_rank', label: 'NextGen rank', title: 'Raw forecast points across positions; not scarcity-adjusted draft or waiver value.', initial: 'asc' },
    { key: 'rank_movement', label: previous ? `Change · W${previous.through_week} → W${query.data?.report.through_week}` : 'Rank change', title: 'Overall rank movement since the previous published weekly cutoff, before filters. Positive means a better rank.', value: r => { const prior = priorRanks.get(r.player_id)?.overall_rank; return prior != null && r.overall_rank != null ? prior - r.overall_rank : null }, render: r => <RankMovement current={r.overall_rank} previous={priorRanks.get(r.player_id)?.overall_rank} /> },
    { key: 'player_display_name', label: 'Player', title: 'Open the player profile in a full page.', align: 'left' },
    { key: 'position_rank', label: 'NextGen · position', title: 'Predicted points rank within this position and horizon, before any filters. Equal predictions share rank.', initial: 'asc', render: r => r.position_rank == null ? 'Unranked' : `${r.position}${r.position_rank}` },
    ...consensusColumns<Ranking>({ showDates: false }),
    { key: 'prediction', label: horizon === 'next4' ? 'Next 4 weeks · points' : 'Remaining points', title: 'Expected league points during the displayed horizon, including scheduled byes.', render: r => fixed(r.prediction) },
    { key: 'current_points', label: 'Points so far', title: 'Observed current-season league points through the stated cutoff.', render: r => fixed(r.current_points) },
    { key: 'ownership', label: 'Ownership', title: 'Separately refreshed ESPN ownership; unmatched players are not assumed free agents.', align: 'left',
      value: r => { const p = owners.get(r.player_id); return p ? p.owner_team_name ?? 'Unrostered' : null },
      render: r => { const p = owners.get(r.player_id); return p ? p.owner_team_name ?? 'Unrostered' : 'Not captured' } },
    { key: 'injury_status', label: 'Status', title: 'Captured ESPN status and current alerts, separate from forecast evidence.', align: 'left',
      value: r => owners.get(r.player_id)?.injury_status,
      render: r => { const p = owners.get(r.player_id); return <div className="player-notes">
        {p ? <><PlayerStatus player={p} /><AttentionBadges flags={p.attention?.filter(flag => flag.kind !== 'injury' && flag.kind !== 'unknown_status')} /></> : <span className="badge rostered">Not captured</span>}
        {r.constraint && <span className="badge out">Reported season-ending absence</span>}
      </div> } },
  ]
  const exportRows = () => exporter.download(`nextgen-${horizon}.csv`,
    async () => (await getResponse(`/api/nextgen/rankings?${params}&format=csv`, token)).blob())
  return <section aria-label="NextGen rankings">
    <div className="analysis-controls">
      <label>Ranking horizon<select value={horizon} onChange={e => setHorizon(e.target.value)}><option value="rest_of_season">Rest of regular season</option><option value="next4">Next four weeks</option></select></label>
      <label>Ranking position<select value={position} onChange={e => setPosition(e.target.value)}><PositionOptions /></select></label>
      <label>Find ranked player<input type="search" value={search} onChange={e => setSearch(e.target.value)} /></label>
      <label>Ranking pool<select value={ownership} onChange={e => setOwnership(e.target.value)}><option value="all">All candidates</option><option value="free">Available in my league</option><option value="mine">My roster</option></select></label>
      <button type="button" className="button" onClick={() => update({ q: null, position: null, pool: null, page: null })}>Clear ranking filters</button>
    </div>
    <QueryError query={query} />
    {query.isFetching && <p role="status">Loading published NextGen rankings…</p>}
    {league.isError && <p className="notice">League ownership unavailable. All-player rankings remain available.</p>}
    {query.data && <>
      <p className="legend"><strong>{query.data.report.season} · Weeks {query.data.report.through_week + 1}–{horizon === 'next4' ? Math.min(18, query.data.report.through_week + 4) : 18}</strong> · Production through Week {query.data.report.through_week}. Published {new Date(query.data.report.published_at ?? query.data.report.generated_at).toLocaleString()}.</p>
      <p className="legend">NextGen ranks expected points for this horizon. ECR is a separately dated preseason reference. Overall points rank does not account for positional scarcity. These are not FAAB prices or weekly matchup recommendations.</p>
      {query.data.report.qb_passing_integration?.filter(d => d.horizon === horizon).map(d => <p key={d.horizon} className="legend"><strong>QB passing integration:</strong> {d.approved ? 'The historical passing experiment passed the separate fantasy tests and now contributes to these QB point forecasts.' : 'The passing experiment did not pass the separate fantasy tests; QB point forecasts retain their existing reference.'} <Link to={researchPath('qb-experiments')}>Review QB experiments and ranking evidence</Link>.</p>)}
      {horizon !== 'next4' && query.data.report.qb_passing_integration?.some(d => d.horizon === 'next4' && d.approved) && <p><Link to={to(intelligencePath('rankings'), { horizon: 'next4', position: 'QB' })}>View the improved four-week QB rankings</Link></p>}
      {league.data && <p className="legend">Ownership/status captured {new Date(league.data.captured_at).toLocaleString()}{league.data.stale ? ' · Saved snapshot; refresh in League.' : ''}</p>}
      {league.data && league.data.week > query.data.report.through_week + 1 && <p className="notice" role="alert">Rankings need a new completed-week build. League observations are newer than the model cutoff.</p>}
      {!!query.data.excluded.length && <p className="notice" role="alert">Some ranking scopes are withheld: {[...new Set(query.data.excluded.map(e => e.reason))].join('; ')}</p>}
      {historyQuery.isError && <p className="notice">Ranking history unavailable. Current ranks remain available.</p>}
      {history && <p className="legend">Rank movement compares published snapshots through completed weeks, for the same horizon. Recipe changes and the shrinking forecast window can affect ranks. {previous ? `Comparing Week ${previous.through_week} with Week ${query.data.report.through_week}.` : 'No earlier published weekly snapshot is available.'} Open row details for the weekly history.</p>}
      <DataTable rows={rows} columns={columns} pagination={page} sortParam="sort" defaultSort="overall_rank" rowNumbers rowKey={r => r.player_id} rowLabel={r => r.player_display_name}
        rowClass={r => owners.get(r.player_id)?.is_mine ? 'mine-row' : undefined}
        emptyMessage={ownership !== 'all' && !league.data ? 'League ownership is required for this filter.' : 'No players match these filters.'}
        renderDetails={r => <PlayerRankHistory ranking={r} history={history} />}
        profileHref={r => playerPath(r.player_id)} />
      <Pager page={page} total={rows.length} noun="rankings" summary={`${rows.length} matching players`}>
        <button type="button" className="button" onClick={() => void exportRows()}>Export rankings · all ownership</button></Pager>
      {exporter.error && <p role="alert">{exporter.error}</p>}
      <details className="reference-section"><summary>Ranking evidence and model decisions</summary>
        <p>Every historical forecast and recipe selection uses earlier seasons only. Modern tests cover 2019–2025; the full selection-policy check begins in 2011. All earlier training seasons are retained. Publication also checks ranking point capture, squared error, small samples and rookies.</p>
        <QueryError query={evidence} />
        {evidence.data?.evaluations.map(e => <section key={e.position}><h4>{e.position} · {e.serving === 'approved' ? 'Validated forecast' : 'Reference forecast'}</h4><p>{e.serving_reason}</p><p>Modern selection-policy MAE: {fixed(e.modern.error)} versus {fixed(e.modern.baseline_error)} reference points. Improvement: {fixed(e.modern.improvement)} points; 95% interval {fixed(e.modern.ci_low)} to {fixed(e.modern.ci_high)}. Adjusted q: {fixed(e.q_value, 3)}. Full-history improvement: {fixed(e.all_history.improvement)} points.</p></section>)}
        <p>{query.data.report.limitations.join(' ')}</p>
      </details>
    </>}
  </section>
}
