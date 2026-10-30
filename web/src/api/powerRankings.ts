import { get } from './client'

export interface PowerWeek {
  week: number; points: number; filled: number; starters: number[];
  opponent_team_id: number | null; win_probability: number | null;
}
export interface PowerTeam {
  team_id: number; team_name: string; wins: number; losses: number; ties: number; points_for: number;
  rating: number | null; rating_rank: number | null; this_week: number | null;
  unfilled_weeks: number; unmatched_players: string[]; weekly: PowerWeek[];
  projected_wins?: number; projected_losses?: number; average_seed?: number;
  first_seed_odds?: number; playoff_odds?: number | null;
}
export interface PowerRankingsResponse {
  status: 'ok' | 'unavailable'; issue: string | null;
  campaign_id: string | null; campaign_name: string | null; horizon: string | null;
  season: number; week: number; captured_at: string; playoff_teams: number | null;
  origin_weeks?: number[]; weeks?: number[]; slots?: string[]; limitations?: string[];
  simulations?: number; score_sd?: number | null; score_sd_samples?: number;
  stale: boolean; teams: PowerTeam[];
}
export const fetchPowerRankings = (campaign: string, token?: string, signal?: AbortSignal) =>
  get<PowerRankingsResponse>(`/api/league/power-rankings?${new URLSearchParams({ campaign })}`, token, signal)
