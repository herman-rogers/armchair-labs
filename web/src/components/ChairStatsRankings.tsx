import { useQuery } from '@tanstack/react-query'
import { leagueQuery, rankingsQuery } from '../api/queries'
import { useCampaignChoice } from '../api/queries/hooks'
import { describeHorizon, type Ranking } from '../api/chairstats'
import { useDataRelease } from '../dataRelease'
import { playerPath, useUrlPage, useUrlState } from '../navigation'
import { DataTable, type Column } from './DataTable'
import { CampaignSelect, VintageNote } from './CampaignSelect'
import { PlayerStatus } from './LeagueAttention'
import { Pager, PositionOptions, QueryError } from './Controls'
import { toCsv, useDownload } from '../download'
import { fixed } from '../format'

/** `/intelligence/rankings` */
export function ChairStatsRankings() {
  const { token } = useDataRelease()
  const choice = useCampaignChoice()
  const { campaignId, horizon, campaign } = choice
  const [position, setPosition] = useUrlState('position', 'ALL', { resets: ['page'] })
  const [search, setSearch] = useUrlState('q', '', { replace: true, resets: ['page'] })
  const [ownership, setOwnership] = useUrlState('pool', 'all', { resets: ['page'] })
  const page = useUrlPage(50)
  const exporter = useDownload()
  const query = useQuery(rankingsQuery(campaignId, horizon))
  const league = useQuery(leagueQuery(token))
  const owners = new Map(league.data?.players.filter(p => p.player_id).map(p => [p.player_id, p]))
  const rows = (query.data?.rankings ?? []).filter(r => (position === 'ALL' || r.position === position)
    && (r.player_display_name ?? '').toLowerCase().includes(search.toLowerCase()))
    .filter(r => ownership === 'all' || (ownership === 'free' ? owners.get(r.player_id)?.availability === 'free_agent' : owners.get(r.player_id)?.is_mine))
  const targets = [...new Set(rows.flatMap(r => Object.keys(r.predictions)))].filter(t => t !== 'points')
  const columns: Column<Ranking>[] = [
    { key: 'overall_rank', label: 'Rank', title: 'Predicted points across every position in this campaign and horizon; not scarcity-adjusted value.', initial: 'asc' },
    { key: 'player_display_name', label: 'Player', title: 'Open the player profile.', align: 'left' },
    { key: 'position_rank', label: 'Position rank', title: 'Rank within the model that forecast this position.', initial: 'asc', render: r => r.position_rank == null ? '—' : `${r.position}${r.position_rank}` },
    { key: 'team', label: 'Team', title: 'Team at the forecast cutoff.', align: 'left', render: r => r.team ?? '—' },
    { key: 'prediction', label: `${describeHorizon(horizon)} points`, title: 'Predicted fantasy points over the horizon under the campaign’s scoring contract.', render: r => fixed(r.prediction) },
    { key: 'points_per_game', label: 'Per game', title: 'Horizon points divided by scheduled games; not a weekly matchup forecast.', render: r => fixed(r.points_per_game) },
    { key: 'scheduled_games', label: 'Games', title: 'Scheduled games in the horizon at the cutoff.' },
    { key: 'availability_status', label: 'Status at cutoff', title: 'Dated roster status the model saw; later news is not applied.', align: 'left', render: r => r.availability_status ?? '—' },
    { key: 'owner', label: 'League', title: 'Current ESPN ownership from the league snapshot.', align: 'left', value: r => owners.get(r.player_id)?.owner_team_name ?? '', render: r => { const p = owners.get(r.player_id); return p ? <span>{p.owner_team_name ?? 'Free agent'} <PlayerStatus player={p} /></span> : '—' } },
  ]
  const exportRows = () => exporter.download(`chairstats-${campaignId}-${horizon}.csv`, async () => {
    const header = ['overall_rank', 'player_id', 'player', 'position', 'position_rank', 'team', 'season', 'origin_week', 'horizon', 'scheduled_games', 'points', 'points_per_game', ...targets]
    const body = rows.map(r => [r.overall_rank, r.player_id, r.player_display_name, r.position, r.position_rank, r.team, r.season, r.origin_week, r.horizon, r.scheduled_games, r.prediction, r.points_per_game, ...targets.map(t => r.predictions[t] ?? '')])
    return new Blob([toCsv([header, ...body])], { type: 'text/csv' })
  })
  return <section aria-label="ChairStats rankings">
    <div className="analysis-controls">
      <CampaignSelect choice={choice} />
      <label>Position<select value={position} onChange={e => setPosition(e.target.value)}><PositionOptions positions={['ALL', ...(campaign?.positions ?? [])]} /></select></label>
      <label>Search players<input type="search" value={search} onChange={e => setSearch(e.target.value)} /></label>
      <label>League pool<select value={ownership} onChange={e => setOwnership(e.target.value)}><option value="all">Everyone</option><option value="free">Free agents</option><option value="mine">My roster</option></select></label>
    </div>
    <QueryError query={query} label="Rankings unavailable" />
    {query.isLoading && <p role="status">Loading rankings…</p>}
    {query.data && <>
      <VintageNote models={query.data.models} />
      {exporter.error && <p role="alert">{exporter.error}</p>}
      <DataTable rows={rows.slice(page.offset, page.offset + page.limit)} columns={columns} defaultSort="overall_rank" sortParam="sort" pagination={page} serverSide rowKey={r => `${r.model_id}-${r.player_id}`} rowLabel={r => r.player_display_name ?? r.player_id} profileHref={r => playerPath(r.player_id)} emptyMessage="No forecasts match these filters."
        renderDetails={r => <div className="player-details"><h4>{r.player_display_name} · {query.data?.campaign_name} {r.model_id}</h4>
          <dl className="profile-facts">{Object.entries(r.predictions).map(([target, value]) => <div key={target}><dt>{target.replaceAll('_', ' ')}</dt><dd>{fixed(value)}</dd></div>)}</dl>
          <p className="legend">{r.candidate_basis ?? 'Candidate'} · status {r.availability_status ?? 'unknown'}{r.availability_as_of ? ` as of ${new Date(r.availability_as_of).toLocaleDateString()}` : ''} · {r.calendar_weeks ?? '—'} calendar weeks.</p>
        </div>} />
      <Pager page={page} total={rows.length} noun="players"><button type="button" className="button" onClick={() => void exportRows()}>Export CSV</button></Pager>
      <details className="outlook-method"><summary>Limitations</summary><ul>{query.data.limitations.map(l => <li key={l}>{l}</li>)}</ul></details>
    </>}
  </section>
}
