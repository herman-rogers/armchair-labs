import { useQuery } from '@tanstack/react-query'
import { campaignsQuery } from '.'
import { type CampaignSummary } from '../chairstats'
import { useUrlState } from '../../navigation'

/** Every published campaign bundle. */
export const useCampaigns = () => useQuery(campaignsQuery())

/** Use the dashboard default unless the URL names an available comparison. */
export function useCampaignChoice(preferred: string[] = ['remaining', 'season', 'weekly'], allowHorizonSelection = true) {
  const campaigns = useCampaigns()
  const [requestedCampaign, setCampaign] = useUrlState('campaign', '', { resets: ['page'] })
  const [requestedHorizon, setHorizon] = useUrlState('horizon', '', { resets: ['page'] })
  const list = campaigns.data?.campaigns ?? []
  const campaign: CampaignSummary | undefined = list.find(c => c.campaign_id === requestedCampaign)
    ?? list.find(c => c.campaign_id === campaigns.data?.default_campaign_id)
    ?? list.find(c => c.campaign_id === 'opening_drive_v2')
  const horizons = campaign?.horizons ?? []
  const horizon = allowHorizonSelection && horizons.includes(requestedHorizon) ? requestedHorizon : preferred.find(h => horizons.includes(h)) ?? preferred[0] ?? ''
  return { campaigns, list, campaign, campaignId: campaign?.campaign_id ?? '', horizon, horizons, setHorizon, setCampaign }
}
