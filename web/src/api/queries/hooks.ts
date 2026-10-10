import { useQuery } from '@tanstack/react-query'
import { campaignsQuery } from '.'
import { newestCampaign, type CampaignSummary } from '../chairstats'
import { useUrlState } from '../../navigation'

/** Every published campaign bundle. */
export const useCampaigns = () => useQuery(campaignsQuery())

/**
 * The campaign and horizon a page is looking at. Both live in the URL so links share
 * the same view; the newest campaign is the default when the URL names none. Nothing
 * is stored server-side: switching is a query change, not a promotion.
 */
export function useCampaignChoice(preferred: string[] = ['remaining', 'season']) {
  const campaigns = useCampaigns()
  const [requested, setCampaign] = useUrlState('campaign', '', { resets: ['page'] })
  const [requestedHorizon, setHorizon] = useUrlState('horizon', '', { resets: ['page'] })
  const list = campaigns.data?.campaigns ?? []
  const campaign: CampaignSummary | undefined = list.find(c => c.campaign_id === requested) ?? newestCampaign(list)
  const horizons = campaign?.horizons ?? []
  const horizon = horizons.includes(requestedHorizon) ? requestedHorizon : preferred.find(h => horizons.includes(h)) ?? horizons[0] ?? ''
  return { campaigns, list, campaign, campaignId: campaign?.campaign_id ?? '', horizon, horizons, setCampaign, setHorizon, requested }
}
