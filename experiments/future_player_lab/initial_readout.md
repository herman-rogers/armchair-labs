# Initial experiment results

The later [completed deep feature and ensemble search](deep_readout.md) expands
this initial study and adds matched comparisons against our published forecasts.

Completed on 2026-09-24. These are retrospective research results and a dated
preseason reconstruction, not production model promotions.

## Completed work

| Run | Scope | Pipeline fits | Status |
|---|---|---:|---|
| `exploration_001` | 10 families, 20 recipes, four positions, four horizons; test 2023–2025 after two earlier validation seasons | 1,600 | Complete, no candidate failures |
| `history_001` | Eight families, 16 recipes; preseason tests 2009–2025 after three earlier validation seasons | 1,280 | Complete, no candidate failures |

The breadth pass uses preseason and week-8 origins. The historical pass adds
longer evaluation for ridge, extra trees, approximate kernels, MLP, PLS, interaction
bases, neural-tree representations and hurdle models. It does not repeat exact
SVR or histogram boosting over all 17 historical test years.

Both runs use NFL observations beginning in 2001 and verified preseason candidates
beginning in 2004. The breadth panel contains 69,584 examples from 4,770 players,
with 1,244 input columns. Its 7,233 unknown workload labels stay unknown; other
observed outputs from those examples remain usable. These rows and their overlapping
horizons are not independent observations.

Across the two runs, 516,980 model/policy forecast rows are saved, each with five
outputs. Source and input hashes passed. All 17,396 reconstructed season-point
labels reconcile exactly with the gold season outcomes.

## What the first breadth pass found

The predeclared ensemble chooses its three constituent pipelines using earlier
points-MSE results. On the later 2023–2025 tests:

- It improves points MSE over persistence on all 16 position/horizon tasks.
- It improves points MSE over the fixed histogram-boosting anchor on 11 of 16 tasks.
- Selecting one pipeline by earlier points MSE improves over that anchor on 5 of 16 tasks.
- The points selector chooses histogram boosting in 42 of 48 yearly decisions;
  the other choices are five hurdle models and one extra-trees model.
- Selecting by earlier NDCG@24 improves later NDCG@24 over the fixed booster on
  only 3 of 16 tasks and loses on points MSE on all 16.

These are task counts, not significance claims. Each breadth comparison has only
three outer years. The ensemble's unweighted mean task-level MSE change against
the booster is about −2.8%; that is not a pooled error reduction or a guarantee
that the difference will persist. Paired annual results and descriptive bootstrap
intervals are saved for inspection.

The experiment therefore provides some support for blending candidates, while
showing that automatic search has not consistently displaced the tree controls.
It does not establish that neural networks, kernels, or learned representations
are universally unhelpful: this is a finite search with limited optimizer budgets.
Several neural fits reach their epoch limits; some PLS fits also emit convergence
or constant-residual warnings. No claim of full optimizer convergence is made.

## Read the results

- [Breadth dashboard with matched-row differences](runs/exploration_report_001/index.html)
- [Breadth comparisons against persistence and fixed boosting](runs/exploration_comparison_001/report.md)
- [2009–2025 preseason dashboard](runs/history_report_001/index.html)
- [Matched historical comparisons](runs/history_comparison_001/report.md)
- [2026 research forecast CSV](runs/forecast_2026_001/forecasts.csv)
- [Verified implementation archive for that export](runs/forecast_sources_001/manifest.json)

The dashboards display each model's available sample count and its matched
comparison count. All Δ MSE comparisons in these dashboards use identical rows.
The original fit-run dashboards are preserved with their original reporting code;
use the linked report versions for relative comparisons involving missing workload.

The longer study illustrates why the objective matters. For example, the QB
ensemble reduces equally weighted annual RMSE from 84.39 to 78.11 points, while
MAE rises from 51.26 to 56.52. Avoid calling that an improvement on every definition
of reliability. The annual, ranking and cohort files expose these tradeoffs.

## 2026 inference

The export contains 839 players under three policies, or 2,517 rows. It refits
selected pipelines on completed earlier seasons; every 2026 label is explicitly
unknown. The output records the actual issuance time and calls itself a preseason
reconstruction. It does not incorporate the observed opening weeks of 2026 and
must not be represented as a forecast originally issued before the season.

Numeric calibration of the ranking-selected policy is particularly poor. It emits
79 negative season-point estimates, including approximately −216.7 for Malik
Cunningham. Negative fantasy outcomes can occur, but that extreme estimate is an
extrapolation warning, not a credible expected point total. Its raw outputs are
retained for research rather than silently clipped. The points-MSE policy has 13
negative estimates (minimum −8.05); the ensemble has one (minimum −1.41).
Inspect point accuracy and calibration before using any ranking-selected score as
a numeric forecast. No policy is promoted by this experiment.

## Validation and next investigations

Twenty-nine focused tests pass. They cover cutoff dates, duplicates, missing and
unlabeled outcomes, retained binary inputs, all ten model families, chronological
selection, matched rows, checkpoint integrity, and a full-run test proving that
changing 2025 labels cannot change its forecasts, choices or current intervals.
Both final dashboards also pass browser filter checks without JavaScript errors.

The main and extended configurations are ready for wider origins, more trials and
longer evaluation; those larger campaigns have not been executed. Useful subsequent
experiments include more thorough optimizer/capacity searches, matched ablations
of discovered features, position-transition robustness, ranking-score calibration,
and evaluation of a protocol frozen before genuinely later outcomes arrive.

Historical roster coverage, retrospective data vintages and unavailable inputs
remain explicit data constraints. Model discovery does not remove them. See the
[protocol](protocol.md) for the exact search space and estimands.
