# Full-history profile experiment, fixed before fitting

Use every accepted QB/RB/WR/TE candidate with a completed season outcome,
including returners, rookies, market-only players and zero-production outcomes.
Accepted candidates begin in 2004. NFL observations begin in 2001 and college
observations in 2004. Do not manufacture pre-2004 preseason candidate cohorts
from players who subsequently appeared. The first trainable test fold is the
first season with at least 30 earlier candidate outcomes at that position:
2005 with the inspected inputs. All 2004 outcomes seed training. No future
outcomes or random train/test split. Do not restrict training to modern years.

Pin historical_backfill_20260923_r2, player_profiles_v1_20260923_r2 and
college_nfl_v1_20260922_r5. Verify each accepted dependency. Profile/college
history is pinned to historical_v2_20260922_r5; require identical candidate keys,
positions, populations, completed outcomes and market inputs across these
versions before joining. New backfill supplies restored names, dated context
and the previously missing rookie/market-only cutoffs. Preserve both saved V2
versions as benchmarks; neither defines eligibility for the new model.

Five feature specifications, fitted separately by position:
1. Basic: age, experience, era, season length, population, draft capital and
   prior-season observed production/workload/efficiency.
2. Career: basic plus all prior NFL history, exposure counts, role-band rates,
   complete career and recent-three-year rates, prior peaks, gaps and team changes.
3. Profile: career plus college history/quality flags, prior reported injury
   history, dated roster/absence context and available dated QB job evidence.
4. Basic + market: basic plus positional ECR, overall ECR and dispersion.
5. Profile + market: all the above.

Keep career totals and career-wide summaries, alongside recent summaries; the
recent features do not truncate stored history or training years. College inputs
are constructed from all earlier captured college seasons, quality-gated by
source season, with missing/partial/truncated coverage explicit. They remain
separate from NFL denominators. Add exposure-weighted college interactions that
diminish with NFL attempts/carries/targets (fixed 200 QB attempts, 100 RB carries,
100 WR/TE targets). Keep unweighted college features too. These weights are an
untuned modeling choice, not a validated reliability estimate.

Unknown features never exclude a candidate and never become confirmed healthy,
active or backup status. Zero exposure counts are distinguished from missing
historical coverage. Prior injury reports are reported statuses, not diagnoses
or a count of injury-caused absences. Current profile metadata (current club,
career endpoints, latest forecasts, eventual success) cannot enter features.
Static birth/draft facts are admitted only when dated by the forecast year;
published historical revisions remain a documented source limitation. The NFL
season/week cutoff is applied before any aggregate or participation-band label.

Two fixed learners for each specification: ridge alpha=100 with training-only
median imputation, missing indicators and scaling; histogram gradient boosting
with squared loss, 120 iterations, learning rate .05, 15 leaves, minimum leaf
30, L2=10, max_bins=63, no early stopping, fixed seed. No tuning on reported
test seasons and no selection of a winner after seeing outcomes. All learners
train on every eligible earlier year with equal row weights. The full-profile
gradient booster is the primary nonlinear challenger; matched basic/profile
comparisons separate new information from architecture.

Execution compatibility, before any completed evaluation: the installed tree
binning code cannot handle fewer than two finite distinct training values.
Remove constant/all-missing training columns for that fold, retaining a
missingness indicator when observation status varies. This uses no test values;
it never removes a candidate or an earlier training year. If all columns are
unlearnable, use the training mean.

Target: full-season league points / nominal season length. Convert predictions
back to season points, clip at zero. A documented full-season absence forces
zero for every new model. Partial known absences are inputs, not a second
multiplicative discount or an invented points cap. No games multiplier.

Report all evaluable historical folds, pre-modern 2005-2018, and modern
2019-2025 separately, by position/population. Compare matched candidate rows,
MAE and RMSE, equal-season paired differences, fixed-seed 10,000 season-cluster
bootstrap intervals and leave-one-season-out ranges. Explicitly distinguish
returners with <10 prior box-score games from <10 career observed weeks; neither
cohort is selected by future success. Report missing-source coverage by year.

For the user's training-history concern, rerun the primary profile and
profile+market models on the modern test folds using only the preceding five
training seasons, retaining the same full career feature vectors. Compare with
the all-earlier-history models on identical rows. This isolates training-year
restriction from individual player-history truncation. This sensitivity is
declared now, not chosen after results.

Market evaluation has two clearly separate pools: all ECR-covered candidates
with new forecasts, and the exact old r5 returner/V2 comparison pool. Keep the
original top-60 positional replacement convention and realized value target.
Report broad market history (2011 onward where ECR exists) and modern history.
Reconcile original V2's 87 modern calls and 0/13 small-prior-season miss recovery.
Track all 13 named misses, but do not train or select features on that subset.
Earlier missing ECR does not remove rows from the generic forecast or training;
all-missing market columns receive no learned coefficient/split until observed
in training. No modern ADP windows are backdated into earlier cutoffs.

Save a create-only research release with feature dictionary, complete predictions,
fold sizes, coverage, paired diagnostics, market misses, provenance hashes and
implementation snapshots. Temporal tests perturb outcome-season observations,
held-out targets and future college/context evidence. Test that missing profiles
and rookies remain present, old career history affects features, all earlier
training rows remain included, and market-pool reconciliation holds. Freeze
production and existing artifacts. No publication or promotion from this run.

All retrospective eras have already informed research. More historical folds
provide breadth, not a newly independent confirmation set. Era changes, manual
evidence gaps and fixed model choices remain limitations.
