import { get } from './client'

/** One model inside a campaign bundle, as `/api/chairstats/campaigns` summarizes it. */
export interface CampaignModel {
  model_id: string; position: string | null; horizon: string; origin_week: number
  targets: string[]; training_through_season: number
  latest_forecast: { season: number; origin_week: number } | null
}
export interface CampaignSummary {
  campaign_id: string; name: string; description: string | null; kind: string | null
  disposition: string | null; generated_at: string | null; manifest_sha256: string
  positions: string[]; horizons: string[]; limitations: string[]; models: CampaignModel[]
}
/** One player's row in a campaign ranking: every target the model declared is in `predictions`. */
export interface Ranking {
  campaign_id: string; model_id: string; player_id: string; player_display_name: string | null
  position: string | null; team: string | null; season: number; origin_week: number; horizon: string
  scheduled_games: number | null; calendar_weeks: number | null
  prediction: number | null; points_per_game: number | null
  position_rank: number | null; overall_rank: number | null
  predictions: Record<string, number>; candidate_basis: string | null
  availability_status: string | null; availability_as_of: string | null
}
export interface ModelVintage {
  model_id: string; position: string | null; season: number; origin_week: number
  generated_at: string | null; artifact_sha256: string; rows: number
}
export interface RankingsResponse {
  campaign_id: string; campaign_name: string; manifest_sha256: string; disposition: string | null
  horizon: string; models: ModelVintage[]; limitations: string[]
  total: number; limit: number; offset: number; rankings: Ranking[]
}
export type PlayerForecast = Ranking & { campaign_name: string; candidates: number }
export interface PlayerForecasts { player_id: string; forecasts: PlayerForecast[] }
export interface TargetsResponse {
  campaign_id: string; campaign_name: string; manifest_sha256: string
  model: { model_id: string; position: string | null; horizon: string; targets: string[]; scoring: string | null; training_through_season: number }
  season: number; origin_week: number; generated_at: string | null; artifact_sha256: string; rows: Ranking[]
}

export const fetchCampaigns = (signal?: AbortSignal) =>
  get<{ campaigns: CampaignSummary[] }>('/api/chairstats/campaigns', undefined, signal)
export const fetchRankings = (params: URLSearchParams, signal?: AbortSignal) =>
  get<RankingsResponse>(`/api/chairstats/rankings?${params}`, undefined, signal)
export const fetchPlayerForecasts = (playerId: string, signal?: AbortSignal) =>
  get<PlayerForecasts>(`/api/chairstats/players/${encodeURIComponent(playerId)}`, undefined, signal)
export const fetchTargets = (campaign: string, model: string, signal?: AbortSignal) =>
  get<TargetsResponse>(`/api/chairstats/campaigns/${encodeURIComponent(campaign)}/models/${encodeURIComponent(model)}/targets`, undefined, signal)
export const fetchMetrics = (campaign: string, model: string, signal?: AbortSignal) =>
  get<{ campaign_id: string; model_id: string; metrics: unknown }>(`/api/chairstats/campaigns/${encodeURIComponent(campaign)}/models/${encodeURIComponent(model)}/metrics`, undefined, signal)

/** Newest campaign first; the request-time default when no campaign is in the URL. */
export const newestCampaign = (campaigns: CampaignSummary[]) =>
  [...campaigns].sort((a, b) => (b.generated_at ?? '').localeCompare(a.generated_at ?? ''))[0]
export const describeHorizon = (horizon: string) =>
  ({ remaining: 'Remaining season', season: 'Full season', next4: 'Next four weeks' } as Record<string, string>)[horizon] ?? horizon.replaceAll('_', ' ')
