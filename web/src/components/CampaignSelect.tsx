import { Link } from 'react-router'
import { describeHorizon } from '../api/chairstats'
import { useCampaignChoice } from '../api/queries/hooks'
import { researchPath } from '../navigation'
import { QueryError, SelectField } from './Controls'

/** Newest campaigns offered inline; the rest are one click away on the models page. */
const RECENT = 8

/** Current campaign label and forecast horizon. */
export function CampaignSelect({ choice, showHorizon = true }: { choice: ReturnType<typeof useCampaignChoice>; showHorizon?: boolean }) {
  const { campaigns, list, campaign, defaultId, horizon, horizons, setHorizon, setCampaign } = choice
  // The list arrives default-first, then newest; keep the current choice even when it is older.
  const shown = list.filter((c, i) => i < RECENT || c.campaign_id === campaign?.campaign_id)
  return <>
    {list.length > 1 ? <SelectField label="Forecast model" value={campaign?.campaign_id ?? ''} onChange={e => setCampaign(e.target.value === defaultId ? '' : e.target.value)}>
      {shown.map(c => <option key={c.campaign_id} value={c.campaign_id}>{c.name}{c.campaign_id === defaultId ? ' (default)' : ''}</option>)}
    </SelectField> : <span className="legend">{campaign?.name ?? 'Opening Drive'}</span>}
    {list.length > shown.length && <Link className="legend" to={researchPath('models')}>All {list.length} models</Link>}
    {showHorizon && <SelectField label="Horizon" value={horizon} onChange={e => setHorizon(e.target.value)} disabled={!horizons.length}>
      {horizons.map(h => <option key={h} value={h}>{describeHorizon(h)}</option>)}
    </SelectField>}
    <QueryError query={campaigns} label="Campaigns unavailable" />
  </>
}

/** A short note when the page shows a campaign other than the default. */
export function CampaignNotice({ choice }: { choice: ReturnType<typeof useCampaignChoice> }) {
  const { campaign, defaultId } = choice
  if (!campaign || campaign.campaign_id === defaultId) return null
  return <p className="notice">{campaign.name} is not the default model{campaign.status ? ` (${campaign.status.replaceAll('_', ' ')})` : ''}.{campaign.description ? ` ${campaign.description}` : ''}</p>
}

/** Which campaign supplied each position, at which cutoff, and whether it was re-scored later. */
export function VintageNote({ models, generatedAt }: { models: { campaign_name?: string; model_id: string; position: string | null; season: number; origin_week: number; rolling_origin?: boolean }[]; generatedAt?: string | null }) {
  if (!models.length) return <p className="legend">No saved forecast for this horizon.</p>
  const parts = models.map(m => `${m.position ?? m.model_id}: ${m.campaign_name ?? ''} ${m.season} wk ${m.origin_week}${m.rolling_origin ? ' (re-scored)' : ''}`.replace(/\s+/g, ' ').trim())
  return <p className="legend">Saved forecasts by position: {parts.join(' · ')}{generatedAt ? ` · generated ${new Date(generatedAt).toLocaleString()}` : ''}. Re-scored rows reuse weights trained at an earlier cutoff; forecasts never update with news after their cutoff.</p>
}
