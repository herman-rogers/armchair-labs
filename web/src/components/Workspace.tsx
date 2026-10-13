import { Outlet } from 'react-router'
import { useQuery } from '@tanstack/react-query'
import { useCampaigns } from '../api/queries/hooks'
import { metricsQuery } from '../api/queries'
import { describeHorizon, type CampaignModel, type CampaignSummary } from '../api/chairstats'
import { DataReleaseStatus } from './DataRelease'
import { DataTable } from './DataTable'
import { useUrlPage, useUrlState } from '../navigation'
import { useCurrentPage } from '../pageNavigation'
import { RookieView } from './RookieView'
import { PositionOptions, QueryError } from './Controls'

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

function CampaignCard({ campaign }: { campaign: CampaignSummary }) {
  return <section className="reference-section" aria-label={campaign.name}>
    <h3>{campaign.name} <small className="faint">{campaign.campaign_id}</small></h3>
    {campaign.description && <p>{campaign.description}</p>}
    <p className="legend">{campaign.generated_at ? `Generated ${new Date(campaign.generated_at).toLocaleString()} · ` : ''}{campaign.positions.join(', ') || 'No positions'} · horizons {campaign.horizons.map(describeHorizon).join(', ')} · manifest <code>{campaign.manifest_sha256.slice(0, 12)}</code>{campaign.disposition ? ` · ${campaign.disposition.replaceAll('_', ' ')}` : ''}</p>
    <DataTable<CampaignModel> rows={campaign.models} columns={[
      { key: 'model_id', label: 'Model', title: 'Model identifier inside the campaign bundle.', align: 'left', initial: 'asc' },
      { key: 'position', label: 'Position', title: 'Position the model forecasts, when it forecasts players.', align: 'left', render: m => m.position ?? '—' },
      { key: 'horizon', label: 'Horizon', title: 'Forecast horizon declared by the model.', align: 'left', render: m => describeHorizon(m.horizon) },
      { key: 'origin_week', label: 'Cutoff week', title: 'Observation cutoff the model forecasts from; 0 is preseason.' },
      { key: 'targets', label: 'Targets', title: 'Every quantity the model predicts.', align: 'left', render: m => m.targets.join(', ') },
      { key: 'training_through_season', label: 'Trained through', title: 'Last season in the training data.' },
      { key: 'latest', label: 'Latest forecast', title: 'Most recent saved forecast season and cutoff.', align: 'left', value: m => m.latest_forecast ? m.latest_forecast.season * 100 + m.latest_forecast.origin_week : null, render: m => m.latest_forecast ? `${m.latest_forecast.season} · week ${m.latest_forecast.origin_week}` : 'None saved' },
    ]} defaultSort="model_id" rowKey={m => m.model_id} rowLabel={m => m.model_id} renderDetails={m => <ModelMetrics campaign={campaign.campaign_id} model={m.model_id} />} />
    {campaign.limitations.length > 0 && <details><summary>Limitations</summary><ul>{campaign.limitations.map(l => <li key={l}>{l}</li>)}</ul></details>}
  </section>
}

/** `/research/models`: every campaign bundle, its models and saved metrics. */
export function ResearchModels() {
  const campaigns = useCampaigns()
  return <>
    <p className="legend">Opening Drive v3 is the default dashboard model. Choose Opening Drive v2 to compare forecasts with the preserved baseline.</p>
    <QueryError query={campaigns} label="Campaign bundles unavailable" />
    {campaigns.isPending && <p role="status">Loading campaign bundles…</p>}
    {campaigns.data && !campaigns.data.campaigns.length && <p className="notice">No campaign bundles are published. Train and score a campaign, then publish its bundle.</p>}
    {campaigns.data?.campaigns.map(c => <CampaignCard key={c.campaign_id} campaign={c} />)}
  </>
}
