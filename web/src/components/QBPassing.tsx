import { Link } from 'react-router'
import { useQuery } from '@tanstack/react-query'
import { playerForecastsQuery, targetsQuery } from '../api/queries'
import { useCampaignChoice } from '../api/queries/hooks'
import { describeHorizon, type Ranking } from '../api/chairstats'
import { intelligencePath, playerPath, useUrlPage, useUrlState } from '../navigation'
import { DataTable, type Column } from './DataTable'
import { CampaignSelect } from './CampaignSelect'
import { Pager, QueryError } from './Controls'
import { fixed, grouped } from '../format'

/** `/intelligence/qb-passing`: every target of the chosen campaign's QB model. */
export function QBPassing() {
  const choice = useCampaignChoice()
  const { campaignId, horizon, campaign } = choice
  const [search, setSearch] = useUrlState('q', '', { replace: true, resets: ['page'] })
  const page = useUrlPage(100)
  const model = campaign?.models.find(m => m.position === 'QB' && m.horizon === horizon)
  const query = useQuery(targetsQuery(campaignId, model?.model_id ?? ''))
  const rows = (query.data?.rows ?? []).filter(r => (r.player_display_name ?? '').toLowerCase().includes(search.toLowerCase()))
  const targets = query.data?.model.targets ?? []
  const columns: Column<Ranking>[] = [
    { key: 'position_rank', label: 'Rank', title: 'Rank by predicted points within the QB model.', initial: 'asc' },
    { key: 'player_display_name', label: 'Quarterback', title: 'Open the player profile.', align: 'left' },
    { key: 'team', label: 'Team', title: 'Team at the forecast cutoff.', align: 'left', render: r => r.team ?? '—' },
    ...targets.map((t): Column<Ranking> => ({ key: `target_${t}`, label: t.replaceAll('_', ' '), title: `Predicted ${t.replaceAll('_', ' ')} over the horizon.`, value: r => r.predictions[t], render: r => t.endsWith('yards') ? grouped(r.predictions[t]) : fixed(r.predictions[t]) })),
    { key: 'scheduled_games', label: 'Games', title: 'Scheduled games in the horizon at the cutoff.' },
  ]
  return <section aria-labelledby="qb-passing-heading">
    <h3 id="qb-passing-heading">QB production forecasts</h3>
    <p>Every quantity the quarterback model predicts, from attempts to passing and rushing production, under the campaign’s declared scoring contract.</p>
    <div className="analysis-controls">
      <CampaignSelect choice={choice} />
      <label>Find a quarterback<input type="search" value={search} onChange={e => setSearch(e.target.value)} /></label>
    </div>
    {campaign && !model && <p className="notice">{campaign.name} has no QB model for the {describeHorizon(horizon)} horizon.</p>}
    <QueryError query={query} label="QB forecasts unavailable" />
    {query.isLoading && <p role="status">Loading QB forecasts…</p>}
    {query.data && <>
      <p className="legend">{query.data.campaign_name} · {query.data.model.model_id} · {query.data.season}, as of week {query.data.origin_week} · trained through {query.data.model.training_through_season}{query.data.generated_at ? ` · generated ${new Date(query.data.generated_at).toLocaleString()}` : ''}.{query.data.model.scoring ? ` Scoring: ${query.data.model.scoring}.` : ''}</p>
      <DataTable rows={rows.slice(page.offset, page.offset + page.limit)} columns={columns} defaultSort="position_rank" sortParam="sort" pagination={page} serverSide rowKey={r => r.player_id} rowLabel={r => r.player_display_name ?? r.player_id} profileHref={r => playerPath(r.player_id)} emptyMessage="No quarterbacks match this search." />
      <Pager page={page} total={rows.length} noun="QBs" />
    </>}
  </section>
}

/** Profile section: a quarterback's forecasts from every campaign model. */
export function PlayerPassing({ playerId }: { playerId: string }) {
  const query = useQuery(playerForecastsQuery(playerId))
  const rows = (query.data?.forecasts ?? []).filter(f => f.position === 'QB')
  if (query.isError) return <p className="notice">QB forecasts unavailable: {query.error.message}</p>
  if (!rows.length) return null
  return <section><h4>Quarterback production</h4>
    <DataTable rows={rows} columns={[
      { key: 'campaign_name', label: 'Campaign', title: 'Campaign bundle.', align: 'left' },
      { key: 'horizon', label: 'Horizon', title: 'Forecast horizon.', align: 'left', render: r => describeHorizon(r.horizon) },
      { key: 'attempts', label: 'Attempts', title: 'Predicted pass attempts.', value: r => r.predictions.attempts, render: r => fixed(r.predictions.attempts, 0) },
      { key: 'passing_yards', label: 'Pass yards', title: 'Predicted passing yards.', value: r => r.predictions.passing_yards, render: r => grouped(r.predictions.passing_yards) },
      { key: 'passing_tds', label: 'Pass TD', title: 'Predicted passing touchdowns.', value: r => r.predictions.passing_tds, render: r => fixed(r.predictions.passing_tds) },
      { key: 'rushing_yards', label: 'Rush yards', title: 'Predicted rushing yards.', value: r => r.predictions.rushing_yards, render: r => grouped(r.predictions.rushing_yards) },
    ]} defaultSort="campaign_name" rowKey={r => `${r.campaign_id}-${r.model_id}`} />
    <Link to={intelligencePath('qb-passing')}>Compare quarterbacks</Link>
  </section>
}
