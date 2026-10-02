# Queue closure: additional diagnostics

Declared after Protocol 2 results, before fitting these variants. These are explicitly
exploratory follow-ups, not untouched confirmation. The source review also asks for age,
recency, exposure weighting, support and dependence checks. Keep all results in the lab.

* Compare a fixed five-season training window with all earlier seasons for QB/RB/WR/TE
  season points, with identical market-control inputs and fixed learners.
* Compare with/without age for an at-least-eight-scoring-appearances event, and separately
  for conditional yards/attempt, yards/carry or yards/target. The event is meaningful
  recorded scoring participation, not injury or biological aging. Three earlier labeled
  seasons are required. Report age bands <25, 25–29, and 30+ without dropping exits.
* The first quantile trial only allowed CQR expansion. Its overcoverage motivates testing
  signed CQR scores `max(lower-y, y-upper)` with global, population, and workload groups.
  Reuse stored uncalibrated OOF bounds; five earlier years, at least three years/100 rows,
  groups 60 rows/three years. Thus this follow-up starts in 2013, with matched comparisons.
  Negative corrections can shrink; if bounds cross, collapse to their midpoint and record
  the event. This is not a claim of conditional coverage under temporal drift.
* Report efficiency MSE both equally per player and weighted by the realized denominator;
  the latter changes the estimand, not the evaluation cohort. Do not filter on future
  workload thresholds. Add cutoff-defined team-change, low-prior-volume/high-market,
  missing-market, position-change and old-age slices of saved predictions.
* For all nonidentity signal comparisons, the team-allocation comparisons, and weekly
  roles, compare season-bootstrap evidence with player-career cluster and two-consecutive-
  season block bootstraps. Check franchise clusters for team allocation and workload-
  calibrated season-point interval scores. Use 2,000 bootstrap draws, fixed seed.
  These are dependence sensitivities, not a universal correction for every correlation.
* Numerically reconcile published NGS rates/counts and compare counts with gold; audit
  trajectories, all-player routes, source dates, college exits, and executable decision
  histories. Distinguish missing historical evaluation data from a source never existing.

Apply one Holm family to the new all-population age/recency MSE and signed-CQR interval-
score comparisons. Slices, dependence checks and diagnostics are descriptive. A future
prospective test remains required even when historical checks agree.
