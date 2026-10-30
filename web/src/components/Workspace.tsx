import { Link, Outlet } from 'react-router'
import { useQuery } from '@tanstack/react-query'
import { useCampaigns } from '../api/queries/hooks'
import { metricsQuery } from '../api/queries'
import { describeHorizon, type CampaignModel, type CampaignSummary } from '../api/chairstats'
import { DataReleaseStatus } from './DataRelease'
import { DataTable } from './DataTable'
import { intelligencePath, to, useUrlPage, useUrlState } from '../navigation'
import { useCurrentPage } from '../pageNavigation'
import { RookieView } from './RookieView'
import { Pager, PositionOptions, QueryError } from './Controls'

/** Page heading and observation status shared by analysis, league and model pages. */
export function DashboardLayout() {
  const page = useCurrentPage()
  return <main className="dashboard-workspace">
    <div className="section-heading"><div><span className="eyebrow">{page?.group}</span><h2>{page?.title}</h2>
      <p>{page?.description}</p></div></div>
    <DataReleaseStatus />
    <Outlet />
  </main>
}

/** Analysis pages render against the published tables and whichever campaign is chosen. */
export function AnalysisSection() {
  const campaigns = useCampaigns()
  return <>
    <QueryError query={campaigns} label="Campaign bundles unavailable" />
    <Outlet />
    {campaigns.data && <p className="legend faint">{campaigns.data.campaigns.length} campaign bundle{campaigns.data.campaigns.length === 1 ? '' : 's'} available. Every bundle is queryable; pages choose one in the URL.</p>}
  </>
}

/** `/intelligence/rookies` */
export function IntelligenceRookies() {
  const [search, setSearch] = useUrlState('q', '', { replace: true, resets: ['page'] })
  const [position, setPosition] = useUrlState('position', 'ALL', { resets: ['page'] })
  const page = useUrlPage(100)
  return <>
    <div className="analysis-controls">
      <label>Search rookies<input type="search" value={search} onChange={e => setSearch(e.target.value)} /></label>
      <label>Position<select value={position} onChange={e => setPosition(e.target.value)}><PositionOptions /></select></label>
    </div>
    <RookieView search={search} position={position} page={page} />
  </>
}

function ModelMetrics({ campaign, model }: { campaign: string; model: string }) {
  const query = useQuery(metricsQuery(campaign, model))
  return <div className="player-details">
    <QueryError query={query} label="Metrics unavailable" />
    {query.isPending && <p role="status">Loading saved metrics…</p>}
    {query.data && <pre className="metrics-json">{JSON.stringify(query.data.metrics, null, 2)}</pre>}
  </div>
}

function CampaignDetail({ campaign }: { campaign: CampaignSummary }) {
  return <div className="player-details">
    {campaign.description && <p>{campaign.description}</p>}
    <p className="legend">{campaign.campaign_id}{campaign.base_campaign ? ` · base ${campaign.base_campaign}` : ''} · manifest <code>{campaign.manifest_sha256.slice(0, 12)}</code>{campaign.disposition ? ` · ${campaign.disposition.replaceAll('_', ' ')}` : ''} · <Link to={to(intelligencePath('rankings'), { campaign: campaign.campaign_id })}>Open rankings</Link></p>
    <DataTable<CampaignModel> rows={campaign.models} columns={[
      { key: 'model_id', label: 'Model', title: 'Model identifier inside the campaign bundle.', align: 'left', initial: 'asc' },
      { key: 'position', label: 'Position', title: 'Position the model forecasts, when it forecasts players.', align: 'left', render: m => m.position ?? '—' },
      { key: 'horizon', label: 'Horizon', title: 'Forecast horizon declared by the model.', align: 'left', render: m => describeHorizon(m.horizon) },
      { key: 'targets', label: 'Targets', title: 'Every quantity the model predicts.', align: 'left', render: m => m.targets.join(', ') },
      { key: 'training_through_season', label: 'Trained through', title: 'Last season in the training data.' },
      { key: 'latest', label: 'Latest forecast', title: 'Most recent saved forecast season and cutoff.', align: 'left', value: m => m.latest_forecast ? m.latest_forecast.season * 100 + m.latest_forecast.origin_week : null, render: m => m.latest_forecast ? `${m.latest_forecast.season} · week ${m.latest_forecast.origin_week}` : 'None saved' },
    ]} defaultSort="model_id" rowKey={m => m.model_id} rowLabel={m => m.model_id} renderDetails={m => <ModelMetrics campaign={campaign.campaign_id} model={m.model_id} />} />
    {campaign.limitations.length > 0 && <details><summary>Limitations</summary><ul>{campaign.limitations.map(l => <li key={l}>{l}</li>)}</ul></details>}
  </div>
}

const latestForecast = (c: CampaignSummary) => Math.max(0, ...c.models.map(m => m.latest_forecast ? m.latest_forecast.season * 100 + m.latest_forecast.origin_week : 0))

/** `/research/models`: every campaign bundle, one row each; expand a row for its models and metrics. */
export function ResearchModels() {
  const campaigns = useCampaigns()
  const [search, setSearch] = useUrlState('q', '', { replace: true, resets: ['page'] })
  const [status, setStatus] = useUrlState('status', '', { resets: ['page'] })
  const page = useUrlPage(25)
  const all = campaigns.data?.campaigns ?? []
  const defaultId = campaigns.data?.default_campaign_id
  const statuses = [...new Set(all.map(c => c.status).filter((x): x is string => !!x))].sort()
  const needle = search.toLowerCase()
  const rows = all.filter(c => (!status || c.status === status)
    && [c.name, c.campaign_id, c.description ?? ''].some(text => text.toLowerCase().includes(needle)))
  return <>
    <p className="legend">Every published campaign bundle in the campaign directory. {defaultId ? <>The dashboard default is <code>{defaultId}</code>; </> : ''}open rankings for any campaign to compare it, or pick one with the model selector on each page.</p>
    <QueryError query={campaigns} label="Campaign bundles unavailable" />
    {campaigns.isPending && <p role="status">Loading campaign bundles…</p>}
    {campaigns.data && !all.length && <p className="notice">No campaign bundles are published. Train and score a campaign, then publish its bundle.</p>}
    {all.length > 0 && <>
      <div className="analysis-controls">
        <label>Find a campaign<input type="search" value={search} onChange={e => setSearch(e.target.value)} /></label>
        {statuses.length > 1 && <label>Status<select value={status} onChange={e => setStatus(e.target.value)}><option value="">Any</option>{statuses.map(x => <option key={x} value={x}>{x.replaceAll('_', ' ')}</option>)}</select></label>}
      </div>
      <DataTable<CampaignSummary> rows={rows} columns={[
        { key: 'name', label: 'Campaign', title: 'Campaign name; expand the row for its models.', align: 'left', initial: 'asc', render: c => <>{c.name}{c.campaign_id === defaultId ? <small className="faint"> · default</small> : null}</> },
        { key: 'status', label: 'Status', title: 'Status the campaign manifest declares.', align: 'left', render: c => c.status?.replaceAll('_', ' ') ?? '—' },
        { key: 'generated_at', label: 'Generated', title: 'When the bundle was generated.', align: 'left', render: c => c.generated_at ? new Date(c.generated_at).toLocaleDateString() : '—' },
        { key: 'latest', label: 'Latest forecast', title: 'Most recent saved forecast across the campaign’s models.', align: 'left', value: latestForecast, render: c => { const v = latestForecast(c); return v ? `${Math.floor(v / 100)} · week ${v % 100}` : 'None saved' } },
        { key: 'positions', label: 'Positions', title: 'Positions covered.', align: 'left', value: c => c.positions.join(','), render: c => c.positions.join(', ') || '—' },
        { key: 'horizons', label: 'Horizons', title: 'Forecast horizons covered.', align: 'left', value: c => c.horizons.join(','), render: c => c.horizons.map(describeHorizon).join(', ') },
        { key: 'models', label: 'Models', title: 'Models in the bundle.', value: c => c.models.length, render: c => c.models.length },
      ]} defaultSort="generated_at" pagination={page} sortParam="sort" rowKey={c => c.campaign_id} rowLabel={c => c.name}
        renderDetails={c => <CampaignDetail campaign={c} />} emptyMessage="No campaigns match these filters." />
      <Pager page={page} total={rows.length} noun="campaigns" />
    </>}
  </>
}
