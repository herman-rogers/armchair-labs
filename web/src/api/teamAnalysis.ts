import { get } from './client'

export type TeamPlayer = {
  player_id: string; name: string; position: 'QB' | 'WR' | 'TE'; games: number;
  starter_games: number; mean: number; sd: number | null; first_season: number;
  last_season: number; last_game: string;
}
export type TeamPair = {
  a: string; b: string; n: number; df: number; r: number | null;
  interval: [number, number] | null; covariance: number | null; years: number[];
  points: { game_id: string; season: number; week: number; a: number; b: number }[];
}
export type TeamRisk = {
  player_ids: string[]; n: number; df: number;
  status: 'available' | 'empty_selection' | 'insufficient_overlap';
  mean: number | null; sd: number | null; variance: number | null;
  independent_variance: number | null; independent_sd: number | null;
  covariance_effect: number | null; variance_change_pct: number | null;
  sd_interval: [number, number] | null; variance_change_interval: [number, number] | null;
  quantiles: { p10: number; p25: number; p50: number; p75: number; p90: number } | null;
  contributions: { player_id: string; mean: number; variance: number; covariance: number; total: number }[];
  covariance: number[][]; games: { game_id: string; season: number; week: number; total: number }[];
}
export type TeamAnalysisData = {
  team: string; start: number; end: number; qb: string; sample: 'starter' | 'active';
  basis: 'season' | 'raw'; team_games: number; players: TeamPlayer[]; pairs: TeamPair[];
  quarterbacks: { player_id: string; name: string; games: number; last_game: string }[];
  risk: TeamRisk; teams: { id: string; name: string }[];
  report: TeamAnalysisCatalog;
}
export type TeamAnalysisCatalog = {
  table: string; table_version: string; version: string; source_version: string | null;
  built_at: string; season: number; through_week: number; earliest_season: number;
  scoring: string; evidence: string; season_type: string;
}
export const fetchTeamAnalysisCatalog = (token?: string, signal?: AbortSignal) =>
  get<TeamAnalysisCatalog>('/api/nextgen/team-analysis/catalog', token, undefined, signal)
export const fetchTeamAnalysis = (params: URLSearchParams, token?: string, signal?: AbortSignal) =>
  get<TeamAnalysisData>(`/api/nextgen/team-analysis?${params}`, token, undefined, signal)
