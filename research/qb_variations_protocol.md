# QB passing variation search · September 24, 2026

This is a new retrospective development experiment, frozen before execution.
It extends the fixed first experiment; that experiment and its prospective-only
disposition remain preserved. The user has requested systematic variations and
production integration. This protocol permits scoped retrospective approval with
prospective monitoring, never a prospective accuracy claim. Previously inspected
years are not an untouched holdout. No deletion of injuries or zero outcomes.

## Information and targets

Use the corrected, hash-verified QB panel, all earlier candidate seasons from
2004 and passing observations from 2001. Test expanding annual folds 2007–2025;
2019–2025 is a reporting slice, not a training restriction. Preserve every weekly
origin and preseason origin. Rates are yards per pass attempt (YPA); volume is
attempts per scheduled game; unconditional production is total future yards.
Undefined future YPA has no rate label, but its zero production stays in totals.
Execution fits use future attempts as training-label weights, never predictors.

Additional inputs use only earlier seasons/current completed weeks: career and
recent throwing efficiency, completions, TD/INT rates, rushing, EPA/CPOE, prior
late-season volume, current team passing competition, and draft/age profiles.
Optional prior NGS CPOE and audited college features retain missingness. Existing
preseason features are already cutoff reconstructed; vintage limitations remain.
No later injury diagnosis or eventual starting designation is a predictor.

## Fixed search space

YPA: six season half-lives (0.5, 1, 2, 3, 5, unlimited) crossed with five prior
attempt counts (0, 50, 150, 400, 1000); four trailing-game windows (4, 8, 16, 32)
crossed with 150/400 prior attempts; prior-season and current-season estimates
with 400 prior attempts; expanding league mean and existing career reference.
Fit ridge alpha 10/100/1000 on efficiency-only, profile and enriched inputs;
boosting 7/15/31 leaves on profile/enriched inputs. Include a 50/50 blend of each
fitted rate model and the fixed 2-year/400-attempt career estimate. All boosts:
120 iterations, .05 learning rate, minimum leaf 60, L2 10, 63 bins, no early
stopping, seed 20260924. No tuning after results.

Yards: preserve the old adaptive reference and both old boosters. Fit the six
profile/enriched boosters to yards/game and attempts/game, and three enriched
ridge variants to yards/game. Multiply attempts forecasts by the nested selected
YPA and include 50/50 direct/component blends. Negative totals are clipped to
zero; fully known unavailability constrains all comparators equally. No extra
partial-absence discount. Products are candidates, not independence assumptions.

Selection uses equal-season out-of-fold errors strictly before the test year,
separately by horizon and preseason/weekly origin. Minimum three earlier years;
before then use the original rate/reference. Primary YPA loss is attempt-weighted
MSE; primary production loss is total-yard MSE. Save every candidate, selections,
fold years, subgroup errors and negative findings. Evaluate complete policies,
not the retrospectively best column. Do not repeatedly search until a gate passes.

## Ranking bridge

Generate separate predictions for the exact saved ranking candidates, schedule
counts and end weeks (four calendar weeks and remaining season). Verify their
league-point labels against the saved ranking benchmark. Never substitute four
team games for four weeks. Train point models on earlier out-of-year passing
forecasts only; this intentionally uses the full available cross-fitted weekly
history, starting 2007, while upstream models retain training from 2004.

Fixed bridges: ridge alpha 10/100/1000 and enriched boosting 7/15/31 leaves to
league points/game, plus each boost on non-passing-yard points/game, restoring
predicted yards times the league's passing-yard coefficient. This retains TDs,
interceptions, rushing and scoring bonuses in the residual. Try each at weights
0.25/0.5/0.75/1 against the existing point reference. Choose by earlier equal-year
MAE, subject to nonworse MSE and top-10 capture, minimum four earlier years;
otherwise retain the reference. Evaluate the whole selected policy from 2011.
No in-sample upstream predictions in downstream training. Non-QB models stay
outside this experiment; overall ranks are recomputed if QB points change.

## Publication gates

Eight primary weekly policies: three rate horizons, three production horizons,
two QB fantasy horizons. Modern exact two-sided annual sign-flip p values receive
Benjamini-Hochberg correction across all eight (q <= .05). Equal-season bootstrap
95% lower gain must exceed zero in full history and modern; at least seven modern
and eight full-history years. Rate requires >=0.5% weighted MSE improvement;
yards >=2% MSE improvement; neither may worsen MAE >2%. Eligible cutoff-defined
groups (100 rows, three years) may not worsen primary loss >10%, in both windows.
Rankings require modern MAE gain >=max(.5 points,1%), nonworse full-history MAE,
MSE and top-10 capture in both windows, and existing rookie/small-prior guardrails.
Preseason comparisons are explicitly exploratory and do not grant approval here.

Only policies passing their own target/horizon gates can serve. Rate approval
does not imply yardage or fantasy approval. Failed/inconclusive variations remain
research-only with reasons; previous artifacts are immutable and marked archived.
Calibrate production intervals on earlier policy errors only. Publish a frozen
current snapshot, hash-bound implementation, evidence and source lineage; later
prospective scoring remains required to measure real-time performance.
