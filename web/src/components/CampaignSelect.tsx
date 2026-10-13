import { describeHorizon } from '../api/chairstats'
import { useCampaignChoice } from '../api/queries/hooks'
import { QueryError, SelectField } from './Controls'

/** Current campaign label and forecast horizon. */
export function CampaignSelect({ choice, showHorizon = true }: { choice: ReturnType<typeof useCampaignChoice>; showHorizon?: boolean }) {
  const { campaigns, list, campaign, horizon, horizons, setHorizon, setCampaign } = choice
  return <>
    {list.length > 1 ? <SelectField label="Forecast model" value={campaign?.campaign_id ?? ''} onChange={e => setCampaign(e.target.value)}>
      {list.map(c => <option key={c.campaign_id} value={c.campaign_id}>{c.name}{c.campaign_id === campaigns.data?.default_campaign_id ? ' (default)' : ''}</option>)}
    </SelectField> : <span className="legend">{campaign?.name ?? "Opening Drive"}</span>}
    {showHorizon && <SelectField label="Horizon" value={horizon} onChange={e => setHorizon(e.target.value)} disabled={!horizons.length}>
      {horizons.map(h => <option key={h} value={h}>{describeHorizon(h)}</option>)}
    </SelectField>}
    <QueryError query={campaigns} label="Campaigns unavailable" />
  </>
}

/** Which campaign supplied each position, at which cutoff, and whether it was re-scored later. */
export function VintageNote({ models, generatedAt }: { models: { campaign_name?: string; model_id: string; position: string | null; season: number; origin_week: number; rolling_origin?: boolean }[]; generatedAt?: string | null }) {
  if (!models.length) return <p className="legend">No saved forecast for this horizon.</p>
  const parts = models.map(m => `${m.position ?? m.model_id}: ${m.campaign_name ?? ''} ${m.season} wk ${m.origin_week}${m.rolling_origin ? ' (re-scored)' : ''}`.replace(/\s+/g, ' ').trim())
  return <p className="legend">Saved forecasts by position: {parts.join(' · ')}{generatedAt ? ` · generated ${new Date(generatedAt).toLocaleString()}` : ''}. Re-scored rows reuse weights trained at an earlier cutoff; forecasts never update with news after their cutoff.</p>
}
