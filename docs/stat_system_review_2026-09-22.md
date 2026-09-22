# Stat-system review — 2026-09-22

Scope: the current working tree, generated boards, retained forecast predictions,
and saved metric report. This is a review, not a model promotion or board rebuild.
Existing uncommitted changes were treated as the current application.

## Recommendation

Preserve the current V1 draft board as an immutable reference. Consolidate the
forward-looking systems around one forecast interface, with an explicitly approved
production model and separately versioned experiments. Keep the frozen 2026 Adaptive
forecast intact for its prospective evaluation.

The immediate priority is consistency: training populations, evaluation populations,
artifact provenance, and the meaning of the numbers displayed together. Adding more
features before resolving these issues would make model selection harder to trust.

## What exists

| System | Actual behavior | Current saved coverage | Recommended role |
|---|---|---|---|
| V1 | Last-season league-scored PPG minus positional replacement, with manual VOR adjustments | 610 players | Preserved draft reference |
| V2 / Projection | Historical and component projections feed fitted PPG and games models; overall ranking uses fitted season-points VOR | 610 model-rated players plus 162 market fallbacks | Production forecast, after consistency repairs |
| Adaptive | Frozen 2026 selector scores joined onto the V2 player pool; selected model differs by position | 610 frozen model scores plus 162 market fallbacks | Frozen experimental comparison |
| Research | Next-generation returner/rookie models, availability and market challengers, rich weekly features, feature discovery | 50 configured fit specifications, 7 marked for live application | Separate research workflow |

V2 is already more than the original hand-built projection. Its position ranks use
fitted PPG for QB/RB, fitted season points for WR, and last-season PPG for TE. Its
overall order uses fitted season points for every position. Those two rankings can
legitimately disagree, but their different objectives need to be visible.

Adaptive does not update its selector as current-season results arrive. Its
`fitted_adaptive_ppg_hybrid` name describes the selector's evaluation target; the
selected values are season points. The saved source choices are athletic models for
QB/TE, full-stack RB, and cutoff-rostered WR. The snapshot contains 611 players; the
live modeled pool contains 610 after the existing removal override.

References: `src/patron/config/league.yaml`,
`src/patron/config/metric_report.yaml`, `src/patron/pipeline.py`,
`src/patron/config/experimental_freeze_2026.yaml`.

## Findings, in priority order

### 1. V1 currently reproduces the draft, but is not actually frozen

The saved V1 board passes the existing fixture comparison: all 160 published players
are present, all PPG and VOR values are within tolerance, top-25 overlap is 25/25, and
rank correlation rounds to 1.000. The advisory flag comparison matches 159/160.
`board.json` and `board_v1.json` are byte-identical.

However, `patron board` rewrites both files on every build. Shared scoring,
aggregation, configuration, and overrides can change the result. The full build also
requires the Adaptive snapshot and manifest before it returns any board; a missing
experimental artifact can therefore prevent a V1/V2 rebuild.

Preserve the exact current board, its overrides, scoring configuration, and digest in
a dedicated draft snapshot. Keep the existing V1 selector/API behavior compatible.
Separate optional experiment builds from the ability to serve or rebuild production
artifacts. Preserving a version name alone does not preserve the draft's numbers.

Evidence: `src/patron/cli.py:73`, `src/patron/pipeline.py:412`,
`data/static/2026_draft_list.md`, `src/patron/validation.py`.

### 2. Rookie research data enters production-model training

The backtest appends rookie rows to the returner rows. The production `fitted_ppg`
and `fitted_games` specifications have no population restriction. Missing historical
features are imputed with training means, so these models both train on and predict
rookies despite having no rookie-specific information.

In the saved completed folds, the production PPG model's eligible training population
includes 2,045 rookie rows; the games model includes 4,114. All 238 pending 2026
rookies receive production fitted forecasts. Several rookie QBs with no historical
inputs receive exactly the same 10.292 PPG and 6.149 games estimates.

The next-generation returner model already demonstrates the intended boundary through
`require_features: [returning_indicator]`; the incumbent lacks it. Consequently,
adding a report-only rookie experiment can change production coefficients. This is a
confirmed coupling, although it is not sufficient by itself to explain every change
in reported performance.

Give every model an explicit training and prediction population. Evaluate a separate
rookie model or a declared market fallback, then combine the two populations at the
forecast-output boundary. Production artifacts should depend only on the approved
model and its dependencies; the current fit fingerprint includes all research specs.

Evidence: `src/patron/metrics/backtest.py:318`,
`src/patron/config/metric_report.yaml:85`, `src/patron/metrics/fit.py:324`,
`src/patron/metrics/fit.py:340`, `src/patron/metrics/fit.py:914`.

### 3. The headline ranking comparisons use different player populations

`_rank_fold` drops players lacking a ranker value before constructing the actual top
K. Each ranker therefore gets its own definition of the successful outcome pool.
`_versus` pairs seasons, but does not reconcile those player pools.

For the saved 2025 overall fold, historical PPG is evaluated on 589 players, ECR on
521, and fitted season points on 776. A minimal diagnostic confirms that a model
which entirely omits the best actual player can still score a perfect hit rate,
because that player is removed before the ideal ranking is computed.

Use two distinct evaluations:

- A common-player comparison to isolate ranking quality among players both methods
  can score.
- A complete draft-pool comparison of the actual deployed policy, including rookies,
  missing scores, market fallbacks, and coverage failures.

Freeze the outcome universe for each comparison. The existing market-disagreement
analysis already intersects player coverage and is a useful foundation. It still
needs coverage reporting and a separate complete-pool evaluation.

Evidence: `src/patron/metrics/backtest.py:706`,
`src/patron/metrics/backtest.py:813`, `src/patron/metrics/backtest.py:1113`.

### 4. Saved boards, model artifacts, and documented evidence have drifted

Reapplying the latest saved fitted models to the saved V2 board's own inputs changes
PPG for all 610 model-rated players beyond 0.001. The largest PPG change is about
2.68; the largest season-points change is about 30.26. For Christian McCaffrey,
the saved board carries 241.633 season points, while the latest models applied to
those same inputs produce about 220.975. This diagnostic did not rebuild or write
the board.

The latest report's fit fingerprint matches the current fit configuration. It does
not establish that the separately saved board was built with that report. Exported
boards need an artifact identifier tying together model coefficients, training
population, feature/scoring definitions, data cutoff, and overrides. Report and board
publication should be coordinated, with stale or degraded states visible in the UI.

The older documentation's claims of 63.8% modern overall hit rate for production
and 65.2% for Adaptive describe an earlier evaluation. The latest saved report,
generated 2026-08-31, reports the following:

| Ranker in latest report | Modern overall hit rate, 2019–2025 |
|---|---:|
| Historical PPG prior | 60.0% |
| Fitted season points | 59.3% |
| Recomputed Adaptive-family selector | 60.5% |
| Next-generation season points | 61.2% |
| Market ECR | 63.3% |

These are reported diagnostics, not a fair model-selection verdict: the populations
above differ, and the recomputed Adaptive-family results do not grade the frozen
2026 forecast. Re-evaluating only returning players changes fitted season points to
62.6% and the Adaptive-family selector to 64.3%, illustrating the scope sensitivity.

The common-player market-disagreement analysis also warrants caution: over its
2011–2025 window, the production model's hit-rate difference versus overall ECR is
-4.89 percentage points and its model-only selections recover 49 actual top-60
players versus 93 for the market-only selections. This supports withholding claims
of demonstrated market outperformance pending a corrected deployment-level test.

Evidence: `data/outputs/metric_report.json`,
`data/outputs/metric_backtest_predictions.parquet`, `data/outputs/board_v2.json`,
`docs/v2_metrics_review.md`, `src/patron/metrics/fit.py:929`.

### 5. The interface mixes several forecast definitions

V2's displayed `Proj PPG` and `Exp G` are the hand-built estimates, while its season
points and roster simulation use fitted PPG and fitted games. Multiplying the two
displayed values does not reproduce the displayed season forecast. For example,
McCaffrey's displayed hand-built values are 16.358 PPG and 14.587 games, while the
fitted values underlying his saved season forecast are 18.247 and 13.243.

Adaptive changes VOR rankings and season-equivalent waiver values but inherits V2's
fitted PPG, fitted games, and projected volatility. All 772 saved rows share those
fields with V2. Its roster expected-points simulation therefore still uses V2
forecasts, even when its power ranking uses Adaptive scores.

Even historical `ppg` changes meaning between V1 and V2: V1 counts production rows;
V2 adds participation-observed games to the denominator. This changes PPG for 222 of
the 610 common players. Keep the legacy definition for V1 and label both definitions
explicitly when comparing systems.

Expose a canonical forecast with its source and units. Either supply a coherent
Adaptive PPG/games decomposition from the selected frozen models, or clearly label
the retained V2 simulation as a separate estimate. A frozen season total alone
cannot identify both PPG and expected games.

Evidence: `web/src/components/RosterPanel.tsx:77`,
`web/src/components/BoardTable.tsx:190`, `src/patron/pipeline.py:210`,
`src/patron/espn/reports.py:293`, `src/patron/metrics/enrichment.py:527`.

### 6. Waiver replacement does not match its stated meaning

The wire documentation says replacement is the best freely available player. The
implementation instead selects QB12/RB25/WR35/TE12 within the already-free-agent
pool. When the pool is smaller, it selects its last player, despite the docstring
claiming a best-player fallback.

A diagnostic with 30 free-agent RBs valued from 30 down to 1 produces replacement
6 instead of the best available value of 30. This changes cross-position waiver
comparisons and any interpretation of positive wire VOR.

Define draft replacement, waiver replacement, and improvement over a specific roster
player as separate quantities. For add/drop decisions, marginal lineup benefit is
more useful than applying preseason positional ranks to the remaining wire.

Evidence: `src/patron/espn/reports.py:94`,
`src/patron/api/league_routes.py:214`.

## Statistical critique and simplification

Keep the strong foundations: exact league scoring, audited touchdown bonuses,
historical production, explicit overrides, position-aware replacement value,
walk-forward fitting, and an immutable prospective forecast.

Use active-game PPG, expected games, season points, and decision-specific replacement
value as the central forecast outputs. Keep targets, carries, role, age, and historical
production as supporting explanations. Historical floor and volatility are useful
descriptions, but their projected versions need calibration before they should drive
strong risk claims.

`projection_confidence` is a sample-support heuristic, not a calibrated probability
that the forecast or rank is correct. Rename it accordingly. Develop prediction
intervals and tiers from held-out residuals, separately checking positions and player
populations. Display model/market disagreement as an uncertainty signal, not as
automatic evidence that the model found a bargain.

The production games model includes both `expected_games` and
`projected_availability`, which are equivalent apart from the season-length scale in
the current construction. They add no independent information. Consolidate that
input in a future candidate and evaluate the change, preserving the frozen forecast.

Do not automatically merge age, injury, role, market price, and confidence into one
extra score. Availability already enters season points. In particular, the roster
risk simulation samples availability while summing a VOR built from
availability-adjusted season points; that extra availability treatment needs its own
objective and validation.

Adaptive's original frozen evidence showed a modest modern hit-rate improvement
(65.24% versus 63.81% for the incumbent). Against ECR in its original five market
folds, its hit rate was 65.67% versus 65.00%, but its NDCG was lower
(0.6701 versus 0.6921). That is useful experimental evidence, not a clear promotion
case. Preserve the snapshot and its planned post-season grade. Also preserve the
grading rules: the snapshot is hashed, but the grader currently reads ranking
settings from the mutable report configuration.

All three selectable systems remain historical/preseason views. Live ownership and
injury displays do not turn them into current-week or rest-of-season forecasts.

## Consolidation sequence

1. Archive the exact current V1 draft artifact and its dependencies; retain V1
   compatibility and independently verify that future work leaves it intact.
2. Separate production training from research populations and model specifications.
   Add board/report provenance and make experimental artifacts optional dependencies.
3. Repair common-pool and complete-pool evaluation, then evaluate the actual deployed
   market-fallback policy. Regenerate model evidence before choosing any new winner.
4. Introduce one forecast contract: active-game PPG, expected games, season points,
   value over the appropriate replacement, uncertainty, source, and as-of date.
   Keep old API fields as compatibility aliases during migration.
5. Use the same selected forecast consistently across board, roster, comparison, and
   waiver views. Put component models and research metrics behind explanation/details
   views. Keep V1 available and Adaptive accessible as a frozen comparison.
6. Pursue improvement through separately evaluated availability and rookie models,
   calibrated uncertainty, and roster-specific decision value. Require new evidence
   before promoting any research result.

## Verification and limits

- Offline test suite: 385 passed, 35 deselected.
- Saved V1 versus the existing draft fixture: all acceptance stages passed.
- Adaptive snapshot digest verified; live model scores match the snapshot within
  the exported three-decimal rounding.
- Board/report drift, training population eligibility, differing evaluation pools,
  and waiver replacement behavior were checked with read-only diagnostics.
- No network data refresh, full statistical refit, board rebuild, model promotion,
  or application-code change was performed for this review.
