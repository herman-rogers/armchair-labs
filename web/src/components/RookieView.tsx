import { useQuery } from '@tanstack/react-query'
import { rankingsQuery, rookiesQuery } from '../api/queries'
import { useCampaignChoice } from '../api/queries/hooks'
import { useDataRelease } from '../dataRelease'
import { DataTable, type Column } from './DataTable'
import { playerPath, type UrlPage } from '../navigation'
import { CampaignSelect } from './CampaignSelect'
import { PlayerLink } from './PlayerLink'
import { Pager, QueryError } from './Controls'
import { fixed } from '../format'
import { forecastRank, rankingsPeriod } from '../forecastDisplay'

export function RookieView({ search, position, page }: { search: string; position: string; page: UrlPage }) {
  const { token } = useDataRelease()
  const choice = useCampaignChoice(['season', 'remaining', 'weekly'])
  const query = useQuery(rookiesQuery(token))
  const rankings = useQuery(rankingsQuery(choice.campaignId, choice.horizon, {}, token))
  const byId = new Map(rankings.data?.rankings.map(r => [r.player_id, r]))
  const rows = (query.data?.rookies ?? []).filter(r => (position === 'ALL' || r.position === position) && r.player_display_name.toLowerCase().includes(search.toLowerCase()))
    .map(r => ({ ...r, forecast: byId.get(r.player_id) }))
  type Row = typeof rows[number]
  const columns: Column<Row>[] = [
    { key: 'player_display_name', label: 'Player', title: 'Current rookie cohort from the observation tables.', align: 'left', initial: 'asc' },
    { key: 'position', label: 'Position', title: 'Recorded position.', align: 'left' },
    { key: 'team', label: 'Team', title: 'Current NFL team.', align: 'left', render: r => r.team ?? '—' },
    { key: 'draft_round', label: 'Round', title: 'NFL draft round; unknown stays blank.', initial: 'asc' },
    { key: 'draft_pick', label: 'Pick', title: 'Overall NFL draft pick, not a model rank.', initial: 'asc' },
    { key: 'college_name', label: 'College career', title: 'Linked college identity; only accepted links join histories.', align: 'left', render: r => r.college_id ? <PlayerLink collegeId={r.college_id}>{r.college_name ?? r.college_id}</PlayerLink> : 'No accepted link' },
    { key: 'forecast_rank', label: `${rankingsPeriod(rankings.data)} · overall`, title: 'Overall rank for the named forecast period, when the model forecast this rookie.', initial: 'asc', value: r => r.forecast?.overall_rank, render: r => forecastRank(r.forecast) },
    { key: 'forecast_points', label: `${rankingsPeriod(rankings.data)} points`, title: 'Predicted points for the named forecast period.', value: r => r.forecast?.prediction, render: r => fixed(r.forecast?.prediction) },
  ]
  return <section aria-label="Rookie analysis"><h3>{query.data?.season ?? ''} rookie class</h3>
    <p>Draft capital and linked college histories from the observation tables, alongside the chosen campaign’s forecast. Open a player to follow their college-to-NFL progression.</p>
    <div className="analysis-controls"><CampaignSelect choice={choice} /></div>
    <QueryError query={query} /><QueryError query={rankings} label="Campaign forecasts unavailable" />
    {query.isFetching && <p role="status">Loading rookies…</p>}
    {query.data && <>
      <p className="legend">{query.data.total} rookies in the {query.data.season} class. Rookies without a campaign forecast show no rank.</p>
      <DataTable rows={rows} pagination={page} sortParam="sort" columns={columns} defaultSort="draft_pick" rowKey={r => r.player_id} rowLabel={r => r.player_display_name} profileHref={r => playerPath(r.player_id)} emptyMessage="No rookies match these filters." />
      <Pager page={page} total={rows.length} noun="rookies" />
    </>}
  </section>
}
