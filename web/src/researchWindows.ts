/** Evaluation slices, not distinct source datasets or quality grades. */
export const EVIDENCE_WINDOWS: Record<string, { label: string; purpose: string; advanced?: boolean }> = {
  modern: { label: 'Modern seasons · default', purpose: 'Primary contemporary comparison. More recent is not automatically cleaner; check each source’s coverage and the actual model folds.' },
  long_horizon: { label: 'Long history · stability check', purpose: 'Larger sample across older NFL environments. Useful for robustness, but some inputs and models cover only the later years.' },
  market: { label: 'Market archive · consensus comparison', purpose: 'Compare with the original dated consensus archive. Fewer seasons; consensus coverage and snapshot timing still matter.' },
  recent: { label: 'Recent seasons · sensitivity check', purpose: 'Overlaps the modern window. Retain to check whether changing the start year changes the conclusion, not as independent confirmation.', advanced: true },
  market_full: { label: 'Extended market · includes backfill', purpose: 'Adds older reconstructed market snapshots. Keep for sensitivity analysis; audit source dates and coverage rather than treating backfill as uniformly equivalent to the original archive.', advanced: true },
}
