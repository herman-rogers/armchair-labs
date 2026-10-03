import { RankMovement } from './RankMovement'
import { weeklyForecastsQuery, rankingHistoryQuery, leagueQuery, matchupsQuery, rankingsQuery } from '../api/queries'
import { Link, Outlet, useLocation, useParams } from 'react-router'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { refreshObservations } from '../api/nextgen'
import type { WeeklyForecastSide, LeagueObservations, MatchupSide, ObservedLineup, RankingsResponse } from '../api/nextgen'
import { useDataRelease } from '../dataRelease'
import { DataTable, type Column } from './DataTable'
import { LeagueAttention, PlayerStatus, PlayerNotes } from './LeagueAttention'
import { leaguePath, leagueWeekPath, matchupPath, playerPath, to, useUrlFlag, useUrlParams, useUrlState } from '../navigation'
import { LeagueOverview } from './LeagueOverview'
import { forecastFor, forecastBasis, leagueSummary } from '../leagueSummary'
import { PlayerLink } from './PlayerLink'
import { PositionOptions, QueryError } from './Controls'
import { LEAGUE_POSITION_FILTERS } from '../positions'
import { fixed, rank } from '../format'
import { currentWeeklyForecasts } from '../weeklyForecasts'

type LeaguePlayer = LeagueObservations['players'][number]
const rosterColumns: Column<LeaguePlayer>[] = [
  { key: 'player_display_name', label: 'Player', title: 'Captured ESPN roster or free-agent entry.', align: 'left', initial: 'asc' },
  { key: 'position', label: 'Position', title: 'ESPN position.', align: 'left' },
  { key: 'espn_team', label: 'NFL team', title: 'Team in the current snapshot.', align: 'left' },
  { key: 'owner_team_name', label: 'Fantasy team', title: 'Current ownership.', align: 'left' },
  { key: 'lineup_slot', label: 'Slot', title: 'Current roster slot; historical matchups use the actual lineup for that week.', align: 'left' },
  { key: 'injury_status', label: 'Status', title: 'Captured ESPN provider tag; ACTIVE does not certify health.', align: 'left', render: r => <PlayerStatus player={r} /> },
  { key: 'attention', label: 'Notes', title: 'Expandable injury reports and other current alerts.', align: 'left', render: r => <PlayerNotes player={r} /> },
]

function Roster({ rows, rankings, sortParam }: { rows: LeaguePlayer[]; rankings?: RankingsResponse; sortParam?: string }) {
  const { token } = useDataRelease()
  const history = useQuery(rankingHistoryQuery('rest_of_season', token))
  const previous = history.data?.version === rankings?.version ? history.data?.snapshots.at(-1) : undefined
  const priorRanks = new Map(previous?.rankings.map(r => [r.player_id, r.overall_rank]))
  const byId = new Map(rankings?.rankings.map(r => [r.player_id, r]))
  const ranked = rows.map(r => ({ ...r, nextgen_overall: byId.get(r.player_id ?? '')?.overall_rank, nextgen_rank: byId.get(r.player_id ?? '')?.position_rank, nextgen_points: byId.get(r.player_id ?? '')?.prediction, nextgen_basis: byId.get(r.player_id ?? '')?.evidence_status }))
  const columns: Column<typeof ranked[number]>[] = [...rosterColumns.slice(0, 2),
    { key: 'nextgen_overall', label: 'NextGen · overall', title: 'Published rest-of-season points rank across the full skill-player pool, before ownership or search filters.', initial: 'asc', render: r => rank(r.nextgen_overall) },
    { key: 'rank_movement', label: 'Rank change', title: previous ? `Overall rank movement from the published Week ${previous.through_week} cutoff to Week ${rankings?.report.through_week}.` : 'No previous published rank available.', value: r => { const prior = priorRanks.get(r.player_id ?? ''); return prior != null && r.nextgen_overall != null ? prior - r.nextgen_overall : null }, render: r => <RankMovement current={r.nextgen_overall} previous={priorRanks.get(r.player_id ?? '')} /> },
    { key: 'nextgen_rank', label: 'NextGen · position', title: 'Rest-of-season points rank across the full position pool, before ownership filtering.', initial: 'asc', render: r => r.nextgen_rank == null ? '—' : `${r.position}${r.nextgen_rank}` },
    { key: 'nextgen_points', label: 'Remaining points', title: 'Published NextGen forecast; not a weekly lineup estimate or bid.', render: r => fixed(r.nextgen_points) },
    ...rosterColumns.slice(2),
    { key: 'nextgen_basis', label: 'Forecast basis', title: 'Historically tested model or explicitly labelled reference.', align: 'left', render: r => r.nextgen_basis === 'validated_forecast' ? 'Historically validated' : r.nextgen_basis === 'reference' ? 'Reference' : 'Unavailable' },
  ]
  return <>
    <DataTable rows={ranked} columns={columns} defaultSort="nextgen_points" sortParam={sortParam} rowKey={r => r.espn_id} rowLabel={r => r.player_display_name}
    emptyMessage="No captured players match these filters." rowClass={r => r.attention?.some(f => f.severity === 'urgent') ? 'attention-row' : undefined} profileHref={r => r.player_id ? playerPath(r.player_id) : undefined} />
  </>
}

function Lineup({ side, forecast, players, currentPlayers, rankings }: { side: MatchupSide; forecast?: WeeklyForecastSide; players?: LeaguePlayer[]; currentPlayers?: LeaguePlayer[]; rankings?: RankingsResponse }) {
  const playersByEspnId = new Map(players?.map(p => [p.espn_id, p]))
  const ranksByPlayerId = new Map(rankings?.rankings.map(r => [r.player_id, r.overall_rank]))
  const nextgenRank = (row: ObservedLineup) => ranksByPlayerId.get(playersByEspnId.get(row.espn_id)?.player_id ?? '')
  const columns: Column<ObservedLineup>[] = [
    { key: 'slot', label: 'Slot', title: 'The lineup slot recorded for this week, colored by player position.', align: 'left', initial: 'asc', render: r => <span className={`pos ${r.position ?? ''}`} title={r.position ? `Position: ${r.position}` : undefined}>{r.slot}</span> },
    { key: 'player_display_name', label: 'Player', title: 'Recorded weekly lineup; includes kickers and defenses.', align: 'left', render: r => <PlayerLink playerId={currentPlayers?.find(p => p.espn_id === r.espn_id)?.player_id}>{r.player_display_name}</PlayerLink> },
    { key: 'nextgen_overall', label: 'NextGen · overall', title: 'Current published remaining-season points rank across the full skill-player pool, not a rank for the selected week.', initial: 'asc', value: nextgenRank, render: r => rank(nextgenRank(r)) },
    ...(currentPlayers ? [
      { key: 'injury_status', label: 'Status', title: 'Captured ESPN status from the current snapshot only.', align: 'left' as const,
        value: (r: ObservedLineup) => currentPlayers.find(p => p.espn_id === r.espn_id)?.injury_status,
        render: (r: ObservedLineup) => { const p = currentPlayers.find(p => p.espn_id === r.espn_id); return p ? <PlayerStatus player={p} /> : <span className="badge rostered">Not captured</span> } },
      { key: 'attention', label: 'Notes', title: 'Expandable injury reports and other current alerts.', align: 'left' as const,
        render: (r: ObservedLineup) => { const p = currentPlayers.find(p => p.espn_id === r.espn_id); return p ? <PlayerNotes player={p} /> : null } },
    ] : []),
    { key: 'points', label: 'Points', title: 'ESPN fantasy points recorded for this week.', render: r => fixed(r.points) },
    { key: 'model_points', label: 'NextGen weekly points', title: 'One-week model points for this week; reference scopes are labeled. Missing forecasts remain unknown.', value: r => forecast?.forecasts.find(p => p.espn_id === r.espn_id)?.prediction, render: r => fixed(forecast?.forecasts.find(p => p.espn_id === r.espn_id)?.prediction) },
    { key: 'pro_opponent', label: 'Opponent', title: 'NFL opponent recorded for the selected week.', align: 'left', render: r => r.on_bye ? 'Bye' : r.pro_opponent ?? '—' },
  ]
  const benchColumns: Column<ObservedLineup>[] = [
    { key: 'position', label: 'Position', title: 'Player position recorded for this week.', align: 'left', initial: 'asc', render: r => <span className={`pos ${r.position ?? ''}`} title={`Roster slot: ${r.slot}`}>{r.position ?? '—'}</span> },
    ...columns.slice(1),
  ]
  return <section><h4>{side.team_name}</h4>
    <p>{fixed(side.score)} recorded points · {fixed(forecast?.model_projection)} NextGen weekly points</p>
    {side.lineup_available ? <><h5>Starters</h5><DataTable rows={side.lineup.filter(r => r.started)} columns={columns} defaultSort="slot" rowKey={r => r.espn_id} />
      <h5>Bench & reserve</h5><DataTable rows={side.lineup.filter(r => !r.started)} columns={benchColumns} defaultSort="points" rowKey={r => r.espn_id} emptyMessage="No recorded bench or reserve entries." /></>
      : <p>No lineup captured for this week. Current rosters are available under Rosters.</p>}
  </section>
}

function Matchups({ overview, rankings }: { overview?: LeagueObservations; rankings?: RankingsResponse }) {
  const { token } = useDataRelease()
  const modelQuery = useQuery(weeklyForecastsQuery(token))
  const { matchupWeek, homeId, awayId } = useParams()
  const validPath = /^[1-9]\d*$/.test(matchupWeek ?? '') && Number(matchupWeek) <= 25 && /^\d+$/.test(homeId ?? '') && /^\d+$/.test(awayId ?? '')
  const week = validPath ? Number(matchupWeek) : null
  const query = useQuery({ ...matchupsQuery(week, token), enabled: validPath })
  const schedulePath = validPath ? leagueWeekPath(week!) : leaguePath('overview')
  if (!validPath) return <section className="notice"><h3>Matchup not found</h3><Link to={leaguePath('overview')}>Back to league overview</Link></section>
  if (query.isError) return <section className="notice" role="alert"><h3>Matchups unavailable</h3><p>{query.error.message}</p><Link to={schedulePath}>Back to league overview</Link></section>
  if (!query.data) return <p role="status">Loading matchup…</p>
  const data = query.data
  const model = overview ? currentWeeklyForecasts(overview, modelQuery.data, data) : undefined
  const forecastForSide = (id: number) => model?.matchups.flatMap(g => [g.home, g.away]).find(s => s.team_id === id)
  const gameIndex = data.matchups.findIndex(game => String(game.home.team_id) === homeId && String(game.away.team_id) === awayId)
  const game = data.matchups[gameIndex]
  const gamePath = (game: typeof data.matchups[number]) => matchupPath(data.requested_week, game.home.team_id, game.away.team_id)
  const teamForecasts = overview ? leagueSummary(overview, rankings) : []
  const currentPlayers = data.requested_week === data.current_week && overview?.week === data.current_week && overview.captured_at === data.captured_at ? overview.players : undefined
  if (!game || data.requested_week !== week) return <section className="notice"><h3>Matchup not found</h3><p>These teams have no captured matchup for Week {matchupWeek}.</p><Link to={schedulePath}>Back to the weekly schedule</Link></section>
  return <section aria-label="Matchup detail">
    <nav className="page-links" aria-label="Matchup breadcrumb"><Link to={schedulePath}>← Week {data.requested_week} matchups</Link><Link to={leaguePath('overview')}>Current league overview</Link></nav>
    <div className="section-heading"><div><span className="eyebrow">{data.season} · Week {data.requested_week} · {game.status}</span>
      <h3>{game.home.team_name} vs {game.away.team_name}</h3></div>{game.involves_me && <span className="badge mine">Your matchup</span>}</div>
    <div className="matchup-scoreboard">{[game.home, game.away].map(side => <section key={side.team_id}>
      <h4>{side.team_name}</h4><strong>{fixed(side.score)}</strong><p>Recorded points</p>
      <p>{fixed(forecastForSide(side.team_id)?.model_projection)} NextGen weekly points</p>
      <Link to={to(leaguePath('rosters'), { team: side.team_id })}>View current roster →</Link>
    </section>)}</div>
    <QueryError query={modelQuery} label="NextGen weekly forecasts unavailable" />
    <h3>Recorded lineups</h3>
    <p className="legend">Patron ranks use today's remaining-season forecasts. {data.requested_week === data.current_week ? 'Current-week scores may be incomplete.' : 'Current injury tags are not applied to historical or future lineups.'}</p>
    {data.requested_week === data.current_week && !currentPlayers && <p className="notice">Current injury details are unavailable for this matchup snapshot. Review the roster alerts or refresh the league.</p>}
    <div className="league-lineups matchup-detail-lineups">{[game.home, game.away].map(side => <Lineup key={side.team_id} side={side} forecast={forecastForSide(side.team_id)} players={overview?.players} currentPlayers={currentPlayers} rankings={rankings?.report.season === data.season ? rankings : undefined} />)}</div>
    {data.requested_week === data.current_week && <p className="legend">Current roster forecasts: {[game.home, game.away].map(side => { const t = teamForecasts.find(t => t.team_id === side.team_id); return `${side.team_name}: ${rank(t?.nextgen_team_rank)} · ${fixed(t?.forecast_points)} remaining player points` }).join(' / ')}. These roster totals include the bench.</p>}
    <nav className="page-links" aria-label="Other matchups">
      {gameIndex > 0 && <Link to={gamePath(data.matchups[gameIndex - 1])}>← Previous matchup</Link>}
      <Link to={schedulePath}>All Week {data.requested_week} matchups</Link>
      {gameIndex < data.matchups.length - 1 && <Link to={gamePath(data.matchups[gameIndex + 1])}>Next matchup →</Link>}
    </nav>
  </section>
}
/** League observations and the matching-season forecasts, shared by every league route through the query cache. */
function useLeague() {
  const { token } = useDataRelease()
  const query = useQuery(leagueQuery(token))
  const rankingQuery = useQuery(rankingsQuery('rest_of_season', token))
  const data = query.data
  const rankings = data && rankingQuery.data?.report.season === data.season ? rankingQuery.data : undefined
  return { query, rankingQuery, data, rankings }
}

/** Layout for `/league/*`: snapshot context, refresh and alerts. */
export function LeagueWorkspace() {
  const { token } = useDataRelease()
  const { pathname } = useLocation()
  const isMatchups = pathname.startsWith('/league/matchups')
  const { query, rankingQuery, data, rankings } = useLeague()
  const client = useQueryClient()
  const refresh = useMutation({ mutationFn: () => refreshObservations(token), onSuccess: () => client.invalidateQueries({ queryKey: ['league-observations'] }) })
  const context = <>
    <div className="section-heading"><div><h3>{data?.league_name ?? 'League'}</h3></div>
      <button type="button" className="button" disabled={refresh.isPending} onClick={() => refresh.mutate()}>{refresh.isPending ? 'Refreshing…' : 'Refresh league'}</button></div>
    {data && <p className="legend">{data.season}, Week {data.week} · Captured {new Date(data.captured_at).toLocaleString()}{data.stale ? ' · saved snapshot is stale' : ''}</p>}
    {data && <LeagueAttention data={data} />}
    <QueryError query={query} /><QueryError query={refresh} />
    <QueryError query={rankingQuery} label="Patron forecasts unavailable">. League results remain available.</QueryError>
    {rankingQuery.isLoading && <p role="status">Loading Patron roster forecasts…</p>}
    {data && rankingQuery.data && !rankings && <p className="notice">The forecast season does not match this league season; ranks are withheld.</p>}
    {rankings && <p className="legend">Patron remaining-season forecasts · {rankings.report.season}, production through Week {rankings.report.through_week} · forecast through Week {Math.max(...rankings.rankings.map(r => r.end_week), rankings.report.through_week)} · published {new Date(rankings.report.published_at ?? rankings.report.generated_at).toLocaleString()}. Reference forecasts are labelled; current injury tags may be newer. Refresh league updates ESPN, not the published forecast.</p>}
    {!data && !query.isError && <p role="status">Loading league observations…</p>}
  </>
  return <section aria-label="League workspace">
    {isMatchups ? <>
      <Outlet />
      <details className="league-recent-activity"><summary>League snapshot & forecast context</summary>{context}</details>
    </> : <>{context}<Outlet /></>}
  </section>
}

/** `/league/overview` */
export function LeagueOverviewPage() {
  const { data, rankings } = useLeague()
  return data ? <LeagueOverview data={data} rankings={rankings} /> : null
}

/** Weekly schedule and dedicated game pages share the same cached observations. */
export function LeagueMatchups() {
  const { data, rankings } = useLeague()
  return <Matchups overview={data} rankings={rankings} />
}

/** Search, position and alert filters shared by the roster and free-agent pages. */
function usePlayerFilters() {
  const [search, setSearch] = useUrlState('q', '', { replace: true })
  const [position, setPosition] = useUrlState('position', 'ALL')
  const [attentionOnly, setAttentionOnly] = useUrlFlag('attention')
  const { data } = useLeague()
  const rows = data?.players.filter(r => (position === 'ALL' || r.position === position) && r.player_display_name.toLowerCase().includes(search.toLowerCase()) && (!attentionOnly || !!r.attention?.length)) ?? []
  const controls = <>
    <label><input type="checkbox" checked={attentionOnly} onChange={e => setAttentionOnly(e.target.checked)} /> Needs attention only</label><label>Find league player<input type="search" value={search} onChange={e => setSearch(e.target.value)} /></label>
    <label>League position<select value={position} onChange={e => setPosition(e.target.value)}><PositionOptions positions={LEAGUE_POSITION_FILTERS} /></select></label>
  </>
  return { rows, controls }
}

/** `/league/rosters?team=<id|all>&compare=<id>`; the default team is yours. */
export function LeagueRosters() {
  const { data, rankings } = useLeague()
  const { rows, controls } = usePlayerFilters()
  const [owner] = useUrlState('team', 'mine')
  const [comparison, setComparison] = useUrlState('compare', '')
  const [, update] = useUrlParams()
  if (!data) return null
  const selectedOwner = owner === 'mine' ? data.my_team_id : owner === 'all' ? null : Number(owner)
  const roster = rows.filter(r => r.owner_team_id != null && (owner === 'all' || r.owner_team_id === selectedOwner))
  const teams = data.teams
  return <>
    <div className="analysis-controls">{controls}
      <label>Roster<select value={owner} onChange={e => update({ team: e.target.value === 'mine' ? null : e.target.value, compare: e.target.value === comparison ? null : comparison })}><option value="mine">My team</option><option value="all">All rosters</option>{teams.map(t => <option key={t.team_id} value={t.team_id}>{t.team_name}</option>)}</select></label>
      <label>Compare with<select value={comparison} onChange={e => setComparison(e.target.value)}><option value="">No comparison</option>{teams.filter(t => t.team_id !== selectedOwner).map(t => <option key={t.team_id} value={t.team_id}>{t.team_name}</option>)}</select></label>
    </div>
    <h3>{owner === 'all' ? 'All rosters' : teams.find(t => t.team_id === selectedOwner)?.team_name ?? 'My roster'}</h3>
    {owner === 'mine' && data.my_team_id == null && <p>Select your team from the Roster menu.</p>}
    <Roster rankings={rankings} rows={roster} sortParam="sort" />
    {comparison && <><h3>{teams.find(t => t.team_id === Number(comparison))?.team_name}</h3><Roster rankings={rankings} rows={rows.filter(r => r.owner_team_id === Number(comparison))} /></>}
  </>
}

/** `/league/free-agents` */
export function LeagueFreeAgents() {
  const { data, rankings } = useLeague()
  const { rows, controls } = usePlayerFilters()
  if (!data) return null
  return <>
    <div className="analysis-controls">{controls}</div>
    <h3>Captured free agents</h3><p className="legend">Ownership is from ESPN; recently dropped players may still require waivers. NextGen ranks expected remaining points, not acquisition prices.</p>
    <Roster rankings={rankings} rows={rows.filter(r => r.availability === 'free_agent')} sortParam="sort" />
  </>
}

/** `/league/transactions` */
export function LeagueTransactions() {
  const { data } = useLeague()
  if (!data) return null
  return <><h3>Recent transactions</h3><DataTable rows={data.transactions ?? []} columns={[
    {key:'date',label:'Date',title:'Recorded transaction time.',align:'left'}, {key:'kind',label:'Action',title:'ESPN transaction type.',align:'left'},
    {key:'player_name',label:'Player',title:'Player involved.',align:'left',render:r => {
      const matches = data.players.filter(p => p.player_display_name === r.player_name)
      return <PlayerLink playerId={matches.length === 1 ? matches[0].player_id : undefined}>{r.player_name ?? '—'}</PlayerLink>
    }}, {key:'team_name',label:'Team',title:'Recorded fantasy team.',align:'left'}, {key:'bid_amount',label:'Bid',title:'Reported bid; absent bids remain unknown.'},
  ]} defaultSort="date" sortParam="sort" rowKey={r => `${r.date}-${r.kind}-${r.team_name}-${r.player_name}`} emptyMessage="No transactions captured." /></>
}

/** `/league/draft` */
export function LeagueDraft() {
  const { data, rankings } = useLeague()
  if (!data) return null
  const draftRows = data.draft.map(p => { const playerId = p.player_id ?? data.players.find(r => r.espn_id === p.espn_id)?.player_id; const f = forecastFor(playerId, rankings); return {...p, player_id: playerId, nextgen_overall: f?.overall_rank, nextgen_position: f ? `${f.position}${f.position_rank ?? '—'}` : '—', nextgen_points: f?.prediction, nextgen_basis: forecastBasis(f)} })
  return <><h3>Draft recap</h3><p className="legend">Original draft order alongside today’s published remaining-season NextGen ranks. These are current forecasts, not the rankings available on draft day or retrospective draft grades. Uncovered players, kickers and defenses remain unranked.</p><DataTable rows={draftRows} columns={[
    {key:'overall',label:'Pick',title:'Actual overall selection.',initial:'asc'}, {key:'round',label:'Round',title:'Draft round.'},
    {key:'player_display_name',label:'Player',title:'Drafted player.',align:'left'}, {key:'team_name',label:'Team',title:'Drafting team.',align:'left'},
    {key:'nextgen_overall',label:'NextGen · overall',title:'Current published remaining-season rank across the full skill-player pool.',initial:'asc',render:r=>rank(r.nextgen_overall)},
    {key:'nextgen_position',label:'NextGen · position',title:'Current published position rank.',align:'left'},
    {key:'nextgen_points',label:'Remaining points',title:'Current remaining-season forecast, not the original draft-day projection.',render:r=>fixed(r.nextgen_points)},
    {key:'nextgen_basis',label:'Forecast basis',title:'Historically validated or explicitly labelled reference.',align:'left'},
    {key:'bid_amount',label:'Auction bid',title:'Recorded bid, when applicable.'}, {key:'keeper',label:'Keeper',title:'ESPN keeper designation.',render:r => r.keeper ? 'Yes' : '—'},
  ]} defaultSort="overall" sortParam="sort" rowKey={r => r.overall} rowClass={r => data.my_team_id != null && r.team_id === data.my_team_id ? 'mine-row' : undefined} profileHref={r => r.player_id ? playerPath(r.player_id) : undefined} rowLabel={r => r.player_display_name} emptyMessage="No draft captured." /></>
}
