# QB boosted-tree capacity study — September 24, 2026

Frozen before fitting. Research only; no catalog, ranking or serving changes.
Scope is all nine canonical **preseason, full-season QB targets**, not the separate
weekly passing-horizon experiment. Use the published, hash-verified analysis,
identical candidate keys, labels, features, availability constraints and units.
Retain backups, nonappearances, injuries and roster exits. Undefined rate labels
remain missing. Provider-vintage and missing dated role/retirement limitations of
the canonical inputs remain; this isolates model settings rather than data repairs.

## Fixed search

Histogram gradient boosting, squared-error training objective, seed 20260923,
63 bins, no random validation split or automatic early stopping. Tree-count
checkpoints: 30, 60, 120, 240, 480. Fit each path once and score staged predictions.
Depth/maximum-leaf pairs: (1,2), (2,4), (3,4), (3,8), (5,8), (5,15),
(5,31), (unlimited,15), (unlimited,63), crossed with learning rates .02/.05/.10.
Base minimum leaf size 30 and L2 10. At unlimited depth/15 leaves/.05 learning
rate, cross minimum leaf sizes 10/30/60 and L2 0/10/100. Also test 50% and 75%
feature subsampling at the base setup, and two deliberate high-capacity stress
paths: unlimited depth/63 leaves/minimum leaf 5/L2 0 at .05 and .10.
Deduplicate paths. The exact original 120-tree booster is included.
No result-driven search expansion. Depth bounds interaction length; leaf count
is the tree capacity/width proxy, not a neural-network width.

## Temporal evaluation and selection

Expanding annual training: completed candidate seasons from 2004 through Y-1;
test Y=2007–2025. Current 2026 labels never enter training, tuning or scoring.
Drop constant columns using training data alone, matching the saved implementation.
First reproduce baseline, ridge and fixed boost in every target/fold, failing on
key, label, coverage or prediction mismatches. Reuse saved comparators thereafter.

Two predeclared complete policies per target: choose by canonical loss (MAE,
except Brier for appearance), and by MSE for conditional-mean forecasting.
For each test year use only earlier annual out-of-fold scores. Require three
earlier folds, otherwise use the original booster. Minimize equal-season loss;
among candidates within one paired standard error of the empirical winner, choose
the lowest leaf-capacity × tree-count bound, then stronger regularization and a
stable ID. Eligible canonical-loss choices must have nonworse earlier mean MSE
than the fixed booster; MSE choices must have earlier MAE <=102% of fixed boost.
Both must have nonworse earlier primary loss than fixed boost. Fixed boost always
remains eligible. No training-error based stopping or test-year choice.
Separate 2023–2025 diagnostic freezes settings chosen from 2007–2022, while
refitting on each year's available history. These are reused historical seasons,
**not an untouched holdout**. All-history headline includes fallback years;
report selection-active 2010–2025 as well.

## Errors and overfitting

Save all candidate row predictions, equal-season MAE/MSE/RMSE/bias and training
errors, fold sizes, configuration choices, and provenance. Compare complete
policies with canonical reference, fixed booster and ridge on identical finite
rows, with an explicit coverage audit. Report all-history and modern windows,
annual gains, 10,000 paired season-bootstrap intervals, two-year moving-block
bootstrap sensitivity, leave-one-season-out gain, and modern exact annual
sign-flip p-values. Apply Holm correction across 18 modern policy-versus-fixed
comparisons (nine targets × two policies). Other comparisons are descriptive.
Season/block uncertainty does not fully address repeated-player dependence or
researcher reuse of this dataset; historical significance is not promotion.

Report prior passing workload (300+, 1–299, zero, unknown), population and small
prior-history groups using cutoff inputs. Report hindsight-best configuration
separately, never as the achieved performance of the selection policy. Show
capacity/tree-count curves with training and future-season errors, selection
stability, and the recent frozen-settings diagnostic. Publish negative findings.
No prospective claim or automatic promotion, regardless of results.

Implementation reference: scikit-learn's official HistGradientBoostingRegressor
documentation (https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html).
