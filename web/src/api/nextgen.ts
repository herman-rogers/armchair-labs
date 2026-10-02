import { get } from './client'
import { allPages } from './pagination'

export interface Entry {
  id: string; label: string; kind: string; target: string | null; horizon: string;
  unit: string | null; positions: string[]; validity: string; evidence: string;
  serving: string; allowed_uses: string[]; reason: string; path?: string; source_manifest_sha256?: string | null; definition?: string; alias_of?: string | null;
}
export interface Catalog {
  version: string; archived_count: number; entries: Entry[];
  gold: { version: string; manifest_sha256: string };
  report: { season: number; players: number; targets: number; training: string; limitations: string[] };
}
export type Measurement = Record<string, string | number | boolean | null | undefined> & {
  ecr_overall?: number | null; ecr_position?: number | null; ecr_overall_date?: string | null; ecr_position_date?: string | null;
  player_id: string; player_display_name: string; position: string; population: string;
  period: string; observed_weeks: number;
}
export interface Forecast {
  ecr_overall?: number | null; ecr_position?: number | null; ecr_overall_date?: string | null; ecr_position_date?: string | null;
  player_id: string; player_display_name: string; position: string; population: string;
  target: string; model: string; season: number; cutoff: string; fallback_source: string; prior_observed_weeks: number; career_observed_weeks: number; prediction: number | null; unit: string;
  overall_rank: number | null; position_rank: number | null; population_rank: number | null;
}
export interface Ranking {
  player_id: string; player_display_name: string; position: string; population: string; team: string | null;
  season: number; through_week: number; horizon: string; end_week: number; scheduled_games: number; schedule_known: boolean;
  prior_weeks: number; current_weeks: number; current_points: number; prediction: number; unconstrained_prediction: number;
  recipe: string; evidence_status: string; evidence_reason: string; rank_eligible: boolean;
  overall_rank: number | null; position_rank: number | null;
  constraint: string | null; constraint_source: string | null; constraint_known_on: string | null;
  ecr_overall: number | null; ecr_position: number | null; ecr_overall_date: string | null; ecr_position_date: string | null;
}
export interface RankingReport {
  season: number; through_week: number; generated_at: string; published_at?: string; observations_saved_at: string;
  candidate_players: number; validated_scopes: string[]; rank_meaning: string; limitations: string[];
  qb_passing_integration?: { horizon: string; approved: boolean; reason: string }[];
}
export interface RankingsResponse {
  version: string; horizon: string; total: number; rankings: Ranking[]; report: RankingReport;
  excluded: { position: string; population: string; reason: string }[];
}
export interface RankingEvaluation {
  position: string; horizon: string; approved_challenger_policy: boolean; reason: string; q_value: number;
  available: boolean; serving: string; serving_reason: string;
  modern: Evaluation & { capture_improvement: number }; all_history: Evaluation & { capture_improvement: number };
}
export interface Evaluation {
  target: string; model: string; position: string; population: string; window: string;
  n: number; years: number; metric?: string; error?: number; baseline_error?: number;
  improvement?: number; ci_low?: number; ci_high?: number; positive_years?: number; mse_improvement?: number; loo_min?: number; prediction_coverage?: number; outside_training_range?: number;
  annual?: { season: number; n: number; error: number; baseline_error: number; gain: number }[];
  calibration?: { lower: number; upper: number; n: number; predicted: number | null; observed: number | null }[];
  diagnostics?: { n: number; zero_outcomes: number; note: string;
    workload_groups: { label: string; n: number; years: number; error: number; baseline_error: number; improvement: number }[] };
}
export interface Registry { version: string; scope: string; total: number; entries: Entry[] }
export interface AttentionFlag { kind?: string; severity: string; label: string }
export interface InjuryNews { summary: string; known_on: string; source_url: string; evidence_status: string }
export interface LeagueObservations {
  attention_notices?: AttentionFlag[]; injury_news_warning?: string | null;
  version: string; season: number; week: number; stale: boolean; league_name: string;
  captured_at: string; age_seconds: number; my_team_id: number | null; regular_season_weeks: number;
  decision_status: string;
  teams: { team_id: number; team_name: string; wins: number; losses: number; is_mine: boolean; faab_remaining: number | null; division_name: string | null; schedule?: { week: number; opponent_team_id: number; score: number; outcome: string }[] }[];
  players: { player_id: string | null; espn_id: number; player_display_name: string; position: string; owner_team_name: string | null;
    owner_team_id: number | null; lineup_slot: string | null; espn_team: string | null;
    availability: string; injury_status: string | null; is_mine: boolean; attention?: AttentionFlag[]; injury_news?: InjuryNews[] }[];
  transactions: { date: string | null; kind: string | null; team_name: string | null; player_name: string | null; bid_amount: number | null }[];
  draft: { player_id?: string | null; overall: number; round: number; round_pick: number; team_id: number; team_name: string; espn_id: number; player_display_name: string; bid_amount: number | null; keeper: boolean }[];
}
export interface ObservedLineup {
  espn_id: number; player_display_name: string; position: string | null; slot: string;
  points: number; projected_points: number | null; pro_opponent: string | null; on_bye: boolean; started: boolean;
}
export interface MatchupSide {
  team_id: number; team_name: string; score: number | null; espn_projection: number | null;
  lineup: ObservedLineup[]; lineup_available: boolean;
}
export interface ObservedMatchups {
  season: number; current_week: number; requested_week: number; available_weeks: number[];
  captured_at: string; stale: boolean; age_seconds: number; source: string;
  matchups: { home: MatchupSide; away: MatchupSide; status: string; involves_me: boolean }[];
}
export type Rookie = Measurement & { draft_round: number | null; draft_pick: number | null; college_linked: boolean }
export const nextgenCatalog = (token?: string, signal?: AbortSignal) => get<Catalog>('/api/nextgen/catalog', token, undefined, signal)
export const nextgenRankings = (params: URLSearchParams, token?: string, signal?: AbortSignal) =>
  allPages<RankingsResponse>(params, p => get<RankingsResponse>(`/api/nextgen/rankings?${p}`, token, undefined, signal), 'rankings', 1000)
export const nextgenRankingEvidence = (params: URLSearchParams, token?: string, signal?: AbortSignal) =>
  get<{ evaluations: RankingEvaluation[] }>(`/api/nextgen/rankings/evidence?${params}`, token, undefined, signal)
export const nextgenPlayers = (params: URLSearchParams, token?: string, signal?: AbortSignal) =>
  allPages<{ total: number; players: Measurement[] }>(params, p => get<{ total: number; players: Measurement[] }>(`/api/nextgen/players?${p}`, token, undefined, signal), 'players', 1000)
export const nextgenForecasts = (params: URLSearchParams, token?: string, signal?: AbortSignal) =>
  allPages<{ total: number; forecasts: Forecast[]; entry: Entry; note: string }>(params, p => get<{ total: number; forecasts: Forecast[]; entry: Entry; note: string }>(`/api/nextgen/forecasts?${p}`, token, undefined, signal), 'forecasts')
export const nextgenEvidence = (params: URLSearchParams, token?: string, signal?: AbortSignal) =>
  get<{ comparisons: Evaluation[] }>(`/api/nextgen/evidence?${params}`, token, undefined, signal)
export const nextgenRegistry = (params: URLSearchParams, token?: string, signal?: AbortSignal) =>
  allPages<Registry>(params, p => get<Registry>(`/api/nextgen/registry?${p}`, token, undefined, signal), 'entries')
export const nextgenLeague = (token?: string, signal?: AbortSignal) => get<LeagueObservations>('/api/nextgen/league', token, undefined, signal)
export const nextgenMatchups = (week: number | null, token?: string, signal?: AbortSignal) =>
  get<ObservedMatchups>(`/api/nextgen/league/matchups${week == null ? '' : `?week=${week}`}`, token, undefined, signal)
export const nextgenRookies = (params: URLSearchParams, token?: string, signal?: AbortSignal) =>
  allPages<{ total: number; players: Rookie[]; season: number; through_week: number }>(params, p => get<{ total: number; players: Rookie[]; season: number; through_week: number }>(`/api/nextgen/rookies?${p}`, token, undefined, signal), 'players', 1000)
export async function refreshObservations(token?: string) {
  const response = await fetch('/api/nextgen/league/refresh', { method: 'POST', headers: token ? { 'X-Data-Catalog': token } : {} })
  if (!response.ok) {
    if (response.headers.get('X-Data-Catalog-Stale') === 'true') window.dispatchEvent(new Event('data-catalog-changed'))
    const body = await response.json().catch(() => ({}))
    throw new Error(body.detail ?? 'League refresh failed')
  }
  return response.json()
}
