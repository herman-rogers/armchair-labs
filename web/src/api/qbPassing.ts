import { get } from './client'

export interface PassingForecast {
  player_id: string; player_display_name: string; team: string | null;
  season: number; through_week: number; end_week: number; scheduled_games: number;
  horizon: string; prediction: number; unconstrained_prediction: number;
  lower: number | null; upper: number | null; reference_recipe: string;
  execution_ypa: number; observed_career_ypa: number | null; evidence_attempts: number;
  effective_attempts: number; prior_weight: number; current_attempts: number; current_yards: number;
  prior_attempts: number; prior_yards: number; role_group: string; roster_state: string;
  roster_source: string | null; roster_known_on: string | null; medical_status: string;
  constraint_reason: string | null; constraint_source: string | null; news_known_on: string | null;
  retired_known: boolean; schedule_known: boolean; issued_at: string;
  production_recipe?: string; execution_reference?: number; rate_recipe?: string;
  rate_evidence_status?: string; evidence_status?: string;
}
export interface PassingReport {
  version: string; generated_at: string; season: number; through_week: number;
  observations_saved_at: string; candidates: number; training_history: string;
  prospective_status: string; prospective_started_at: string; limitations: string[];
  variation_research?: string; production_approved_horizons?: string[];
  rate_approved_horizons?: string[]; ranking_approved_horizons?: string[];
}
export interface PassingEvaluation {
  model: string; horizon: string; window: string; origin_type: string;
  n: number; years: number; rmse: number; mae: number; mse: number; bias: number;
  mse_gain: number; ci_low: number; ci_high: number;
  interval_coverage: number | null; interval_width: number | null; interval_n: number;
  groups: (PassingEvaluation & { group: string })[];
  annual: { season: number; n: number; mse: number; mae: number; mse_gain: number }[];
}
export const passingForecasts = (params: URLSearchParams, token?: string, signal?: AbortSignal) =>
  get<{ version: string; total: number; report: PassingReport; forecasts: PassingForecast[] }>(`/api/nextgen/qb-passing?${params}`, token, undefined, signal)
export const passingEvidence = (params: URLSearchParams, token?: string, signal?: AbortSignal) =>
  get<{ version: string; comparisons: PassingEvaluation[]; limitations: string[];
    decisions: { reason: string; retrospective_checks_passed: boolean }[];
    component_scores: { horizon: string; n: number; primary_fraction_mse: number; execution_observed_n: number; execution_ypa_mae: number; execution_attempt_weighted_mae: number }[];
  }>(`/api/nextgen/qb-passing/evidence?${params}`, token, undefined, signal)

export interface VariationEvaluation {
  model: string; status: string; n: number; years: number; mae: number; rmse: number | null;
  improvement_pct: number; ci_low: number; ci_high: number;
  annual: { season: number; n: number; mae?: number; mse?: number; error?: number; gain?: number }[];
}
export interface VariationDecision {
  target: string; horizon: string; approved: boolean; reason: string; q_value: number;
  modern: { improvement_pct?: number; improvement: number; ci_low: number; ci_high: number };
  all_history: { improvement_pct?: number; improvement: number; ci_low: number; ci_high: number };
}
export const passingVariations = (params: URLSearchParams, token?: string, signal?: AbortSignal) =>
  get<{ version: string; scope: string; target: string; horizon: string; decision: VariationDecision | null;
    comparisons: VariationEvaluation[];
    report: { version: string; generated_at: string; rate_variations: number; yard_variations: number;
      ranking_variations: number; approved_scopes: string[]; limitations: string[] };
  }>(`/api/nextgen/qb-passing/variations?${params}`, token, undefined, signal)
