import type { ScoringSeries } from './api/scoring'

/** Shared axes preserve comparisons across cards, including negative actual scores. */
export function scoringDomain(series: ScoringSeries[]) {
  const values = series.flatMap(s => s.weeks.flatMap(w => [w.actual, w.predicted])).filter((v): v is number => v != null)
  return [Math.floor(Math.min(0, ...values) / 5) * 5, Math.ceil(Math.max(10, ...values) * 1.1 / 5) * 5] as [number, number]
}
