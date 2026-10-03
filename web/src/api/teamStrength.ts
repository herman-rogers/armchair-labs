import { get } from './client'

export interface StrengthPlayer {
  espn_id: number; name: string; position: string; slot: string | null;
  prediction: number | null; basis: string | null; status: string | null;
  started: boolean; usable: boolean; reason: string | null;
}
export interface TeamStrength {
  correlation_risk?: CorrelationRisk;
  team_id: number; team_name: string; faab_remaining: number | null;
  results: {
    weeks: number; through_week: number; record: string; points: number | null;
    ppg: number | null; ppg_change: number | null; points_against: number | null;
    average_margin: number | null; all_play: number | null; high: number | null;
    low: number | null; volatility: number | null; season_pace: number | null;
    breakdown_weeks: number; middle_ppg: number | null; weakest_ppg: number | null;
    top2_share: number | null; cumulative_top2_share: number | null;
    different_top_scorers: number; ppg_rank: number | null; all_play_rank: number | null;
  };
  current: {
    adjustments: string[]; omitted: { name: string; reason: string | null }[];
    total: number | null; top2: number | null; other_starters: number | null;
    middle: number | null; weakest: number | null; median: number | null;
    top2_share: number | null; effective_contributors: number | null;
    weekly_total: number | null; complete: boolean; covered: number; required: number;
    best_legal: number | null; usable_bench: number; replaceable: number;
    mean_drop: number | null; worst_drop: number | null;
  };
  positions: { position: string; actual_ppg: number | null; historical_weeks: number;
    starters: number; weekly_points: number | null; usable_backups: number;
    best_backup: string | null; unavailable: number; rank: number | null }[];
  history: { week: number; score: number; opponent_score: number | null;
    opponent_team_id: number; outcome: string; margin: number | null;
    all_play: number | null; scoring_rank: number | null; top2_share: number | null;
    contributions: { espn_id: number; name: string; position: string; points: number }[] | null }[];
  depth: { starter: string; espn_id: number; position: string; starter_points: number | null;
    replacement: string | null; replacement_points: number | null; replacement_status: string | null;
    moves: string[]; drop: number | null; conservative_drop: number | null; reason: string | null }[];
  players: StrengthPlayer[];
  lineup: StrengthPlayer[];
}
export interface HistoricalConnection {
  status: string; n: number; seasons: number[]; r: number | null; interval: number[] | null;
  sd: number | null; independent_sd: number | null; variance: number | null;
  independent_variance: number | null; covariance_effect: number | null;
  variance_share_pct: number | null; variance_change_pct: number | null;
  variance_change_interval: number[] | null;
}
export interface CorrelationRisk {
  status: string; issue?: string; history_start?: number; history_end?: number;
  unmapped_starters?: number; unknown_team_starters?: number;
  source?: { table: string; source_version: string; version: string };
  observed?: { n: number; covariance_effect: number | null; team_variance_share_pct: number | null; covered_connections: number };
  connections?: {
    id: string; nfl_team: string;
    a: { espn_id: number; name: string; position: string };
    b: { espn_id: number; name: string; position: string };
    historical: HistoricalConnection & { by_season: (HistoricalConnection & { season: number })[] };
    observed: { n: number; status: string; r: number | null; covariance_effect: number | null; team_variance_share_pct: number | null };
  }[];
}
export interface TeamStrengthResponse {
  season: number; week: number; captured_at: string; version: string | null;
  forecast_available: boolean; forecast_label: string; forecast_issue: string | null;
  stale: boolean; teams: TeamStrength[];
}
export const nextgenTeamStrength = (token?: string, signal?: AbortSignal) =>
  get<TeamStrengthResponse>('/api/nextgen/league/team-strength', token, undefined, signal)
