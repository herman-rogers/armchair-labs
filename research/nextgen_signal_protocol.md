# Enriched full-history signal audit — fixed before fitting

Final replay pins published gold `canonical_20260923_r5`, derived from enriched
`canonical_20260923_r4`. Verify its entire manifest dependency closure. No direct cache inputs,
saved model predictions, current player metadata, or 2026 outcomes enter fitting.
Use all completed admitted 2004–2025 candidate rows for coverage. Report the 54
quarantined candidates separately. Do not reconstruct earlier candidate lists
from eventual NFL appearances. This is retrospective exploratory research.

The immediate question is which observation families add predictive information,
not which historical model can be selected as a new production winner. Existing
Next-gen, Adaptive, starter, outlook and college results remain separately dated
evidence. Do not mix their old predictions into the new training table.

Fit separate position/population models for returners and rookies. Market-only
candidates remain in the coverage inventory but have no own-stat forecast in this
experiment. Train on every earlier completed candidate year, requiring three
earlier years and at least 30 outcomes. Missing feature values retain candidates.
NFL histories use every earlier captured season (observations begin 2001), with
completed-career and recent-three-season summaries kept distinct. College history
begins 2004; accept linked, single-team, complete college seasons strictly before
the NFL entry/forecast year and ending before the forecast cutoff.

Fixed learner: ridge alpha 100, training-only median imputation and missing
indicators, training-only standardization. Target is season league points divided
by nominal NFL season length (16 through 2020, 17 subsequently). Convert back to
points, floor at zero. Force zero for a documented full-season absence in every
model. Partial absence counts are features, not a second discount. No tuning,
feature selection, blend optimization, or refitting after viewing results.

Returner baseline ladder: calibrated prior PPG and games; then add prior per-game
attempts, carries, targets and receptions plus target/air-yard shares. Test each
predefined family separately above usage: completed career/recent-three history;
age/experience; snap level/trend; weekly production/usage shape; red-zone/goal-line
work; per-opportunity efficiency; team environment; dated availability/roster
context; route proxies; NFL tracking statistics; draft/combine measurements.
Profile role-band rates belong to the snap family. They describe participation,
not a starter designation. Rookies compare position mean, draft capital, then
separately age/combine and college production/shares. Also fit the union of all
families as a declared diagnostic, without choosing it as a winner.

Repeat every incremental comparison above the same baseline plus cutoff-safe ECR
(log and inverse positional and overall ranks, dispersion). A regression using ECR
is a market-informed baseline, not the original consensus ranking. Evaluate that
lane only on ECR-observed test candidates, after at least three earlier seasons
with market observations. This is an incremental-information test, not draft
profit or a claim to beat raw ECR rankings.

A family can start changing forecasts only after three earlier seasons with at
least ten candidates observing one of its substantive values. Coverage counters
and missingness flags do not establish substantive observation. Before admission,
copy the relevant baseline forecast. Record admission per fold; primary family
comparisons include only admitted folds. Preserve all earlier years in training.
This prevents reporting a short tracking feed as decades of evidence.

Report coverage by year, position and population; positive/true observations of
Boolean facts, not merely non-null counts. Report all evaluable folds plus fixed
eras 2007–2012, 2013–2018, 2019–2025. Use identical candidates within every paired
comparison, equal-season MAE differences, MSE guardrail, 10,000 fixed-seed season
bootstrap 95% intervals, positive-season counts and leave-one-year-out means.
Keep returner small-prior (<10 games) diagnostics, without selecting on outcomes.
Intervals are descriptive, unadjusted for the many families and correlated
repeated players. No isolated feature causal/importance claims follow from these
family comparisons. Modern comparisons and full-history comparisons overlap.

Save a create-only release containing coverage, features, predictions, fold
admissions, complete comparisons, provenance, implementation snapshots and this
protocol. Verify production/frozen artifacts are unchanged. Test temporal
exclusion, held-out target invariance, training population separation, admission,
and missingness semantics. Publication/model promotion is outside this audit.

## Documented integrity correction after diagnostic run r1

The first run revealed extreme early forecasts. Inspection of enriched weekly
and seasonal inputs found 2003–2008 total receptions exceeding total targets;
the release records 20,570 impossible weekly pairs and 2,538 impossible admitted
preseason pairs. This is an input integrity violation, not poor predictive
performance used as a reason to drop a season. Preserve r1 and its original
protocol snapshot as a contaminated diagnostic, not selection evidence.

For r2, detect affected source seasons directly from observations: total targets
less than receptions and at least ten weekly receptions-greater-than-targets
violations. Mask targets for the entire affected source season, plus associated
target/air-yard shares, receiving-first-down rate and prior-season opportunity
shape. Never invent replacement targets. Mask incomplete career target totals;
derive all target rates after masking. Keep receptions, points, rushing/passing
production, every candidate, every outcome, and every earlier training season.
The view is derived from the same pinned enriched gold; it does not modify the
accepted release. Save its exact diagnostic and policy with results. Count only
finite numeric values as coverage; NaN is missing, not an observed statistic.

All learner choices, families, years and outcome definitions remain fixed.
Add a reporting guardrail against calibrated prior points as well as the usage
baseline on identical rows. This uses saved forecasts and introduces no extra
fitting or model selection. This integrity correction is explicitly subsequent
to viewing r1; r2 remains exploratory. Unusual position-switch histories may
still expose model extrapolation failures and must remain in evaluation.

The published catalog became available during this audit. Gold r5 shares enriched
r4 with the initial accepted gold r4; changed consumed tables only normalize NaN
shares to null. Final audit r3 replays the unchanged r2 correction/learners against
that published r5. Verify feature and prediction equality with r2 rather than
selecting between results. r1/r2 snapshots retain their exact earlier protocols.
