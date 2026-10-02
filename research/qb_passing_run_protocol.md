# QB passing implementation experiment · September 24, 2026

Frozen into each create-only run before fitting. This is retrospective development,
not a new untouched holdout. The earlier named cases have been inspected.

Revision 2 corrects two implementation defects identified in revision 1: no-history
fallbacks use earlier outcome means within the candidate population (one preseason
row per year, as in the legacy benchmark), and passing outcomes are retained even
when a QB candidate's later position changes to TE/WR. A career forecast with no
NFL attempts uses the same explicit population fallback. Revision 1 is preserved
as superseded research; these repairs are not independent replication or tuning.

Revision 3 corrects a second application of partial availability: known unavailable
game fraction is now an explicit model input. An unconditional forecast is never
multiplied by that fraction afterward. A fully unavailable horizon still forces
zero for all methods. Revision 2 remains preserved, superseded research. No
hyperparameters, model family or selection threshold changed after viewing results.

## Population, origins, labels

All QB preseason candidates from 2004 onward, plus QBs first observed before each
weekly cutoff. Passing history starts in 2001. Evaluate 2007–2025 with expanding
training on all earlier completed seasons; show 2019–2025 separately. Current
2026 outcomes are never training labels. No injury/zero-production exclusions.
Origins are preseason and every completed NFL week. A weekly origin is the day
after the last scheduled game in that week. No same-week injury reports enter an
earlier origin. Historical provider snapshots are reconstructed, not vintage data.

Predict next scheduled team game, next four scheduled team games, and remaining
regular season. Unknown teams use a labelled league-week schedule fallback.
Labels sum the player's production over the corresponding future week interval;
byes are not zeros in the exposure denominator. Freeze schedule and team at issue.
Schedule slots are frozen at issue. A later cancellation is not removed using
future knowledge; its zero production remains in the total-yardage target.

Separate three observed workload states: no pass attempts, 1–14 attempts, and
15+ attempts. These are production states, not medical diagnoses or confirmed
starting assignments. Predict state fractions over the horizon, and conditional
passing production. An absent observation has no measured YPA.

## Fixed candidates (no outcome-driven hyperparameter search)

1. Prior-season yards per scheduled game (legacy reference; earlier population
   mean for no history).
2. Current-season pace, falling back to prior-season at preseason.
3. Career reference: recency-weighted career passing yards / attempts, shrunk
   toward earlier league YPA with 100 prior attempts; season half-life 3 years.
   Career workload probabilities update with recent scheduled games. An injury
   fragment supplies only its actual attempts to execution evidence.
4. Adaptive ridge reference: imputed/scaled features, alpha 100; direct mean
   yards per future game. Multi-year and current opportunity/efficiency inputs.
5. Direct boosting, same features: 100 iterations, learning rate .05, 15 leaves,
   minimum leaf 60, L2 10, max bins 63, no early stopping, fixed seed 20260924.
6. Conditional boosting: same settings; workload-state probabilities and
   conditional attempts, with conditional yardage residual correction around
   the exposure-weighted execution estimate. This correction estimates the
   remaining opportunity/execution dependence; it is not an independence claim.

Regressors fit yards per scheduled game on the combined horizon panel and restore
the row's known schedule length for total-yardage scoring. This prevents long
horizons from dominating the fit; the declared primary evaluation remains total
yardage MSE. Multi-year source corrections are a hash-bound QB-specific input view
over the existing gold release; unrelated products retain their input release.

Age, draft capital and missingness enter as profile priors. College features are
not added in this fixed first experiment. Compare all methods on identical rows
and dated constraints. Also preserve the original saved preseason booster for
an explicitly labelled historical comparison, without rewriting its forecasts.

## Evidence, selection and serving

Primary score: equal-season mean squared error of total yards. Report RMSE, MAE,
bias, season-bootstrap intervals, and cutoff-defined sparse-history, established,
current-low-workload and current-substantial-workload groups. Intervals are
exploratory, unadjusted; overlapping horizons and careers are not independent.
Report workload probability squared error and execution errors only where their
denominators exist. Conditional error does not replace unconditional total error.

Choose a reference using only earlier out-of-fold seasons (minimum 3); before
that use the career reference. Challenger policy also selects only on earlier
folds. To replace a reference retrospectively require >=2% MSE improvement,
positive 95% season-bootstrap MSE gain interval, no >2% MAE deterioration,
and no >10% MSE deterioration in any group with >=100 rows across >=3 seasons,
both full history and modern. Three horizons are a single exploratory family;
no challenger receives validated serving status until prospective shadow review.

Publish the nested selected reference as a reference forecast, its evidence and
separate execution/opportunity measurements. Failed/inconclusive challengers
remain research-only. This experiment cannot promote a fantasy-point ranking.

Calibrate a descriptive 80% residual range using only earlier out-of-fold
errors, separately by horizon and current-workload group (>=100 examples, else
horizon fallback). Record actual coverage and width. Do not call it a medical
risk estimate or a guaranteed interval. No historical interval is calibrated
using that season's labels.

Current production and news have separate cutoffs. Preserve raw and constrained
forecasts, source links and issue times. Dated retirement/return and announced
absence evidence updates opportunity, never erases career execution. Unknown
medical status remains unknown. Preserve a current prospective forecast snapshot
before its target games and evaluate it later, not during this run.
