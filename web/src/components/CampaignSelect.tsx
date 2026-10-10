import { describeHorizon } from '../api/chairstats'
import { useCampaignChoice } from '../api/queries/hooks'
import { QueryError, SelectField } from './Controls'

/**
 * Campaign and horizon pickers. Every published campaign is offered; the choice is a
 * URL parameter, never a server-side selection.
 */
export function CampaignSelect({ choice, showHorizon = true }: { choice: ReturnType<typeof useCampaignChoice>; showHorizon?: boolean }) {
  const { campaigns, list, campaign, horizon, horizons, setCampaign, setHorizon } = choice
  return <>
    <SelectField label="Model campaign" value={campaign?.campaign_id ?? ''} onChange={e => setCampaign(e.target.value)} disabled={!list.length}>
      {!list.length && <option value="">No campaigns published</option>}
      {list.map(c => <option key={c.campaign_id} value={c.campaign_id}>{c.name}{c.disposition === 'research_preview' ? ' · research preview' : ''}</option>)}
    </SelectField>
    {showHorizon && <SelectField label="Horizon" value={horizon} onChange={e => setHorizon(e.target.value)} disabled={!horizons.length}>
      {horizons.map(h => <option key={h} value={h}>{describeHorizon(h)}</option>)}
    </SelectField>}
    <QueryError query={campaigns} label="Campaigns unavailable" />
  </>
}

/** One line describing which saved forecasts a response came from. */
export function VintageNote({ models, generatedAt }: { models: { model_id: string; position: string | null; season: number; origin_week: number }[]; generatedAt?: string | null }) {
  if (!models.length) return <p className="legend">No saved forecast for this horizon.</p>
  const origins = [...new Set(models.map(m => `${m.season} · as of week ${m.origin_week}`))]
  return <p className="legend">Saved forecasts: {origins.join('; ')} · models {models.map(m => m.position ?? m.model_id).join(', ')}{generatedAt ? ` · generated ${new Date(generatedAt).toLocaleString()}` : ''}. Forecasts do not update with news after their cutoff.</p>
}
