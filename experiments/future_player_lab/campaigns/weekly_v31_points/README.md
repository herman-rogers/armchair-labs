# Opening Drive v3.1: uncertainty feedback into point predictions

This extension tests whether information learned from uncertainty and earlier
forecast errors improves expected points, ranking decisions and error variance.
It is part of v3.1 research. The earlier interval-only v3.1 run stays sealed so
its forecasts, identity and prospective comparison remain reproducible.

## Reproduce

```sh
.venv/bin/python -m experiments.future_player_lab.campaigns.weekly_v31_points.run
.venv/bin/python -m pytest experiments/future_player_lab/tests/test_weekly_v31_points.py
```

Default sealed run:
`experiments/future_player_lab/runs/opening_drive_v31_points_20261013_002`.
Interactive report: `data/outputs/opening_drive_v31_points_2026/index.html`.
Use fresh `--output` and `--report` directories for another run. Existing outputs
are never overwritten. Every fold and final 2026 fit is serialized; inputs,
source, local dependencies, protocol, predictions, decisions and checksums are
retained. The protocol is written before evaluating candidates.

To regenerate the presentation from sealed results without refitting:

```sh
.venv/bin/python -m experiments.future_player_lab.campaigns.weekly_v31_points.report
```

The public report receipt identifies the sealed run and current renderer.

## Candidate families

There are 27 fixed candidates and two earlier-season selectors:

- Three point-identical controls: original v3, interval-only v3.1, and v3 with
  the same residual-interval procedure as the changed means.
- Position bias correction; workload corrections with 24/100/400 prior rows;
  player corrections with 8/24/64 prior rows, shrunk toward the workload group.
- Regularized linear and tree residual corrections, with and without added
  uncertainty; matched half-strength corrections.
- Direct point regressors, with and without uncertainty.
- Appearance/conditional-points mixtures, with and without uncertainty; a
  matched half-strength mixtures blended with the original v3 forecast.
- Seven-member model blends: equal weights, historical inverse MSE,
  nonnegative covariance-aware convex weights, and conditional inverse MSE
  weights with and without uncertainty. Conditional weights are half-shrunk
  toward the convex blend. Inverse MSE is a tested heuristic, not an optimality
  claim for correlated model errors.

The extra inputs are original spread and quantile asymmetry, original
distribution calibration bias, v3.1 padding, and separately calibrated tail
shifts/widths. Tail corrections use fixed finer workload groups. Every one is
computed from forecasts and outcomes available before the forecast season.
Raw historical variance features remain available in both matched arms.

The mixture uses `appeared`, meaning a recorded box-score row, rather than
`points > 0`. Zero-point and negative-point appearances are included. Completed
nonappearance rows are checked to have zero scores. This is a participation
proxy, not a recovered official active/inactive roster label. Missing future
labels are never interpreted as nonappearance training observations.

## Evaluation contract

Use the frozen v3 panel and matched base predictions. Fits use up to five
earlier seasons beginning in 2018. Validation predictions start in 2019;
evaluation covers 2021–2025. Fits are frozen throughout each evaluation season.
Historical within-season feature updates use only already completed games.
Base model policies were originally selected in 2018–2020; these years and
the later evaluation years have already been used in development. This is a
retrospective extension, not an untouched holdout.

Selection uses the preceding two validation seasons and equal-year RMSE. The
designated `select_guarded` policy also requires prior RMSE for base projections
at least 5, and QBs at 1–5, to stay within 1% of the baseline when support reaches
100 rows. It chooses per position. The unguarded selector is a control. Exact
ties prefer v3.1. Both may select the unchanged baseline.

Primary: point RMSE. Secondary: MAE, mean bias, error variance, interval
coverage/width/score, and weekly ranking outcomes. MSE equals centered error
variance plus squared mean bias. A narrow interval is not evidence of a better
point prediction.

Ranking evaluation selects the top 10 QBs/TEs and top 20 RBs/WRs each week from
identical eligible universes. It records realized points and regret relative
to hindsight top-k. This is a ranking diagnostic, not actual-roster start/sit
or a league simulation. Cohorts are fixed by original v3 predictions so models
cannot improve their apparent subgroup scores by moving rows between cohorts.
Outcome-conditioned appearance groups are diagnostics only.

Changed means get 80%/90% intervals from earlier out-of-time residuals of that
same method, using fixed finer workload groups with a 100-row fallback to
position. Selected policies use the selected candidate's calibrated intervals.
The matched interval control distinguishes a change of interval recipe from
an improvement in point forecasts. No distribution variance is inferred merely
from two endpoints, and no full predictive or joint-lineup distribution is
claimed.

Paired season bootstrap intervals use 5,000 draws across five seasons. They
are descriptive and do not correct for testing multiple methods. The report
distinguishes the predesignated selector, matched uncertainty ablations, and
exploratory individual-method results.

## Boundaries and preservation

This bounded study covers the proposed mechanisms and controls, not every
possible estimator. K/DST, a new joint-lineup model and API deployment are out
of scope. Final 2026 fits are research artifacts based on the frozen snapshot;
they are not advertised as fresh forecasts. No existing v3/v3.1 model, API
bundle or pregame capture is changed. Before/after checksums and exact replay
of each serialized 2025 candidate verify preservation and inference behavior.

Primary technical references:
[out-of-sample stacking](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingRegressor.html)
and [LightGBM objectives](https://lightgbm.readthedocs.io/en/latest/Parameters.html).

## Revision history

Initial run `_001` is retained. Audit found that computing the tail shift as
`(new_lo + new_hi - old_lo - old_hi) / 2` introduced cancellation noise around
zero in the first training year. Standardizing this noise made the early
linear uncertainty control unstable and inflated its later calibrated ranges.
Revision `_002` computes the exactly equivalent endpoint differences first,
so unchanged endpoints produce exactly zero. A regression test covers this.
It also adds two matched half-strength basic controls after inspecting `_001`,
to distinguish the effect of partial blending from the effect of uncertainty
inputs. This expansion is disclosed in the protocol; it is not a fresh holdout.

## Completed revision 002 findings

The designated earlier-season selector changed pooled RMSE versus the unchanged
v3/v3.1 point prediction by +0.183% QB, +0.064% RB, −0.218% WR and −0.418% TE.
WR improved RMSE in all five evaluation seasons; TE in four. Selected TE MAE
slightly worsened (+0.128%), so even its RMSE gain is not a win on every metric.
The ranking diagnostic improved for RB/WR/TE and worsened for QB; ranking and
RMSE are distinct objectives.

Adding uncertainty inputs improved 16 of 28 matched position/family RMSE
comparisons, with small and inconsistent differences. Half-strength basic
mixtures already improved WR RMSE by 0.181% and TE by 0.421%; uncertainty increased
those improvements to 0.218% and 0.433%. Much of the gain therefore comes from
the mixture/blend architecture rather than the additional uncertainty features.

These results support a targeted future WR/TE mixture comparison, not replacing
every position with the adaptive selector. The prospective log and API remain
unchanged. The sealed run contains 61,046 evaluation player-weeks, 29 methods,
and 108 exact serialized-model replay checks; 1,356 protected files were
unchanged. The full related test suite passed 42 tests, including a pipeline
test proving that changing 2025 labels cannot alter any 2021–2025 forecast or
selection decision.
