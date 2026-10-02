# Longitudinal discovery protocol

## Research question

Can learned descriptions of player history and context improve future production
and ranking forecasts over persistence and fixed nonlinear models? Which methods
transfer across positions, horizons, years and player populations?

Separate the feature representation, estimator, training-history policy, forecast
target and evaluation metric. Optimize complete pipelines using earlier data.
No single globally selected feature list is imposed on every estimator.

## Data and population

The pinned `canonical_targets_20260923_r3` manifest and each loaded table must match
their SHA-256 hashes. Gold preprocessing is reused as a data-quality contract;
older forecasting systems, registry statuses and discovered correlations are not.

The completed preseason candidate roster is joined one-to-one to its outcome
contract. It includes zero future production, rookies, returning players and
market-only candidates. Selection does not require an outcome-season appearance
or minimum attempts/targets. The roster's upstream historical coverage remains a
limitation: in-season discoveries outside preseason membership are not evaluated.

For a season/origin pair, an observation is a feature only if its season/week is
at or before the completed origin. Labels use strictly later weeks, ending at the
specified horizon. A model predicting year Y trains on examples ending before Y.
Thus overlapping horizons and repeated players never split randomly across a
training/test boundary. Seeing an established player's earlier career in training
is intentional. Rookie cohort results measure a different generalization problem.

The current release has observations from 2001, candidate contracts from 2004,
and complete outcomes through 2025. Earlier observations are useful warmup history.
No fabricated 2001–2003 preseason population or complete 2026 labels are introduced.

Inputs:

- 27 production/efficiency channels from raw player weeks, plus all numeric/boolean
  weekly-usage observation fields except season/week keys. Provider-fitted EPA/CPOE
  and related fields are retrospective vintages, not certified original model vintages.
- Dated preseason age, experience, draft, measured weight, depth, roster, transaction,
  known absence and team-opportunity context. Their availability dates are checked.
  In-season models retain these as preseason context; they are not relabeled as current.
- The last three observed college annual records strictly before the forecast season
  and with a recorded season end before the preseason cutoff, using linked identities.
  Unknown coverage stays explicit. NFL outcomes never enter college inputs.
- Each history window computes per-channel mean, standard deviation, slope and
  observation count. Window 0 retains the full observed career. Other windows count
  recorded observations, not scheduled games. Slopes use season*32 + week spacing;
  that preserves order and season gaps but is not a count of elapsed calendar days.
- Raw sequences keep the last configurable number of recorded observations for
  13 core production channels. Gaps and total observed history are separate context.

Missing source values remain NaN. Training-only medians, scales and missing flags
handle predictors. A training-constant value column cannot be learned from and is
removed within that fit; its missing flag remains. Binary inputs have no special
exclusion. Nulls inside a future production record make that aggregate label
unknown. An eligible player with no future records has zero recorded production.
This does not establish why the player was absent or that they were medically unavailable.

## Targets and horizon units

| Output | QB | RB | WR / TE |
|---|---|---|---|
| Points | League scoring | League scoring | League scoring |
| Workload | Pass attempts | Carries + targets | Targets |
| Yards | Passing | Rushing + receiving | Receiving |
| Touchdowns | Passing | Rushing + receiving | Receiving |
| Observed weeks | Count of future player-week records | Same | Same |

The windows are calendar NFL weeks, including byes, not the next N appearances.
Targets are fit per calendar week of the forecast horizon and converted back to
totals for reporting. Count outputs are nonnegative and predicted record weeks
cannot exceed the horizon. Points and yards may legitimately be negative.
Unknown labels never remove otherwise observed targets from training.

## Representations and search

The random generator creates recipes before outcomes are scored. Each family has
an all-history summary-input anchor with modest compute. Additional recipes search
summary/raw-sequence/combined views, expanding or 5/10-year training windows,
no decay or 3/7-year half-lives, regularization, kernel bandwidth, representation
dimension, tree size and neural capacity/epochs. Fixed choices and sampled settings
are fully recorded. Change `models.py` to expand these spaces; no result is claimed
outside the executed space.

- Ridge is a regularized linear control.
- Histogram boosting fits each target; extra trees share multivariate splits.
- RBF SVR consumes training-fitted PCA representations. Nyström kernel regression
  tests nonlinear similarity on the full standardized input without exact SVR's
  quadratic sample scaling.
- PLS learns supervised projections shared across observed targets, then ridge fits
  those projections.
- Interaction discovery ranks inputs with training-only extra trees, generates
  pair products/squares among 24 selected transformed inputs and adds them to all
  original transformed inputs for ridge. This is a bounded interaction search,
  not exhaustive symbolic regression. Pair names are saved as hypotheses.
- MLP learns nonlinear shared hidden features directly. Neural-tree uses a supervised
  MLP bottleneck followed by extra trees. Representation and prediction are evaluated
  together on later data. No random validation early stopping is used. Reaching an
  optimizer budget is recorded and is not evidence of convergence.
- Hurdle models estimate any future recorded participation, then conditional production.
  Both components use training observations only; predictions average over that uncertainty.

All target scaling, feature scaling, PCA, PLS, kernel maps, selectors and neural
representations fit inside their own chronological training fold. Supervised
representations can overfit their training examples; outer-year performance tests
whether that learning transfers.

## Selection and evaluation

For an outer year Y, candidate forecasts on each of the preceding K validation years
were themselves trained using only still earlier years. The selector reads those
past scores, chooses a pipeline, refits on eligible pre-Y examples and predicts Y.
Cached fixed-recipe fits implement this nested chronology without redundant refitting.
No choice reads Y's labels. Persistence is also eligible.

Three adaptive policies are evaluated:

1. Lowest equally weighted earlier-season points MSE.
2. Highest earlier-season points NDCG@24.
3. Equal-weight ensemble of the three lowest earlier-season points-MSE candidates.

All five output losses are reported for each policy. Selection is currently based
on points, so a selected model is not claimed to be optimal for every other output.
Changing the selection target is a separate experiment. SVR's epsilon-insensitive
loss also differs from squared-loss mean estimation; compare its outcomes without
calling all estimators conditional-mean models.

MSE, MAE, signed bias, within-origin Spearman correlation, NDCG@24 and top-24 overlap
are saved annually. NDCG uses nonnegative actual production as gain. Tied ranks and
very small populations limit top-K interpretation. Reported RMSE is the square root
of mean annual MSE, giving each season equal weight. Cohort files describe points
performance for rookie/returning/market-only groups without dropping low performers.

Intervals use absolute residuals from earlier completed forecasts of the same
adaptive policy, in per-calendar-week units. The first outer year has no calibrated
intervals. Nominal 80% coverage is an empirical target, not a guarantee under
dependence or distribution shift. Paired MSE differences against persistence are
bootstrapped by season. These intervals are descriptive and unadjusted for the
many comparisons. A three-year breadth pass is especially weak evidence about
long-term stability; use the longer profiles and inspect fixed-tree comparisons.

Repeatedly studied historical seasons are retrospective evidence. Freeze a selected
protocol before evaluating genuinely later outcomes. No model is automatically
registered, promoted or published on the strength of this run.

## Research basis

- [Forecasting: rolling-origin evaluation](https://otexts.com/fpp3/tscv.html)
- [Cawley & Talbot: model-selection overfitting](https://jmlr.org/papers/v11/cawley10a.html)
- [Bengio et al.: representation learning](https://arxiv.org/abs/1206.5538)
- [TabReD: time splits can change model rankings](https://arxiv.org/abs/2406.19380)
- [nflWAR: contextual player evaluation and uncertainty](https://arxiv.org/abs/1802.00998)
- [Brill et al.: dependence in football observations](https://arxiv.org/abs/2406.16171)
- [Gneiting: align forecast objectives and scoring](https://arxiv.org/abs/0912.0902)
