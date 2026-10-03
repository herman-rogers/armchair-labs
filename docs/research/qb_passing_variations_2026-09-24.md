# QB passing variations and fantasy integration · September 24, 2026

> Dated research record. Findings and roster advice apply to the stated data and cutoff; follow-up ideas are not an active implementation commitment.

This experiment separates passing efficiency (yards per attempt), opportunity
(attempts per future scheduled game), unconditional future yards, and league-point
rankings. It extends the original fixed QB experiment. It does not erase short or
injured seasons, or turn missed opportunities into evidence of zero passing ability.

## Search and chronology

The frozen search tests 72 rate variations, 24 yardage variations and 36 fantasy
combinations. Career half-lives and prior-attempt weights, recent game windows,
regression, boosting, profile/college/tracking features and blends are explicit
alternatives. This is a bounded search, not a claim to have exhausted all models.

Training retains all earlier candidate seasons from 2004, with passing observations
from 2001. Forecast folds cover 2007–2025. Modern means evaluating 2019–2025, never
discarding older training history. A recipe for a given year is chosen using only
earlier out-of-year forecasts. Initial years use the original reference.

Future YPA is defined only when there are pass attempts. Its primary loss weights
by attempts within a season and then equally weights seasons. Zero-attempt cases
remain in unconditional production errors. Injury diagnoses without trustworthy
publication cutoffs are not predictive features. Unknown medical status stays
unknown; dated news can constrain opportunity without erasing career evidence.

The fantasy bridge trains on earlier cross-fitted passing estimates, beginning
with the first available 2007 upstream forecasts. It predicts complete league
points, or the non-passing-yard residual plus predicted passing yards multiplied
by the league scoring coefficient. TDs, interceptions, rushing and bonuses remain
part of the league-point target. Its next-four horizon uses **four calendar weeks**;
the passing display uses **four scheduled team games**. Matched fantasy outcomes
must equal the saved ranking benchmark before fitting proceeds.

The ranking comparison tests the combined integration recipe, including its
additional weekly training examples and profile inputs. It does not establish
that every gain is caused specifically by a lower standalone YPA error.

## Serving contract

The protocol freezes eight primary policy tests: three efficiency horizons, three
yardage horizons and two QB ranking horizons. Publication requires meaningful
improvement, season-level uncertainty checks, an eight-policy multiplicity
correction and subgroup guardrails. Rankings additionally test top-10 point
capture. Rate approval cannot substitute for a yardage or ranking approval.

Only approved policies update current numbers. Individual alternatives and failed
policies remain in Research → QB experiments, with their losses and status. An
approved selection policy does not automatically approve each individual recipe.
The Intelligence → QB passing view and QB profile card use the published policy
and an interval calibrated on earlier forecast errors. Fantasy rankings identify
whether their separate integration test passed. Unrelated position models retain
their current predictions; overall ranks are recomputed if QB forecasts change.

Releases are immutable and hash verified. The catalog cannot drop these policies
silently, or serve a variation experiment from a different gold/profile/week
release. An incident affecting the approved rate or yardage policy closes its
serving path instead of substituting archived output.

The original QB research and registry revalidation dispositions remain preserved.
Variation revision 1 stopped before fitting on a snap-only candidate coverage
check (Cooper Rush, 2019). Revision 2 stopped in the first fold because early
tracking columns were entirely missing. Both are quarantined incomplete research.
Revision 3 retains the snap-only ranking candidate and removes unusable columns
using training data only. No outcome-based hyperparameter changes followed.

## Reproduction and future scoring

```sh
just qb-variations NEW_RESEARCH_VERSION
just qb-variations-publish NEW_RESEARCH_VERSION NEW_DELIVERY_VERSION
just qb-variations-score NEW_DELIVERY_VERSION NEW_SCORE_VERSION
```

For a staged data refresh, pass `--analysis-version STAGED_ANALYSIS` to the build
and publisher. First build the corrected data, profile, ranking and base QB
products; then build the variation experiment and publish the complete catalog.
Every rebuild saves all candidates, selections, fold boundaries, scoring outcomes,
implementation and decisions. A later observation release scores the frozen
delivery without refitting. Date-only schedules conservatively exclude same-day
issues unless verified kickoff timing becomes available.

Historical results are retrospective development, not an untouched holdout or
prospective evidence. The expanded protocol allows scoped retrospective deployment
with prospective monitoring; it does not rewrite the first experiment's
prospective-only disposition or assert an expert-market/FAAB advantage.

## Results

Research: `qb_variations_20260924_r3`. Delivery: `nextgen_qb_variations_20260924_r2`.
The authoritative published version is always `data/current.json`.

The new policy changes 33 current QB point forecasts for the four-week horizon.
All remaining-season point predictions and non-QB point predictions are preserved
exactly. Overall four-week ranks update to account for the changed QB totals.
The original 213 registry revalidation dispositions remain intact, with no entries
returned to the pending-revalidation queue.

Five of eight policy scopes pass the frozen gates. Modern results (2019–2025):

| Target | Horizon | Reference error | Selected policy error | Decision |
|---|---|---:|---:|---|
| YPA, attempt-weighted RMSE | Next team game | 1.828 | 1.812 | Approved; 1.70% MSE reduction, q=.050 |
| YPA, attempt-weighted RMSE | Next four team games | 1.187 | 1.164 | Research only; full-history interval and multiplicity gates fail |
| YPA, attempt-weighted RMSE | Remaining season | 0.975 | 0.950 | Research only; full-history interval and multiplicity gates fail |
| Passing yards, RMSE | Next team game | 63.73 | 61.53 | Approved; 6.78% MSE reduction, q=.0417 |
| Passing yards, RMSE | Next four team games | 201.73 | 196.23 | Approved; 5.37% MSE reduction, q=.0417 |
| Passing yards, RMSE | Remaining season | 505.76 | 479.73 | Approved; 10.03% MSE reduction, q=.0417 |
| League points, MAE | Next four calendar weeks | 7.640 | 7.044 | Approved; 7.81% MAE reduction, q=.050 |
| League points, MAE | Remaining season | 33.806 | 33.242 | Research only; modern interval includes no improvement, q=.391 |

MSE reduction is not RMSE reduction. The remaining-season yardage improvement is
principally fewer large errors: its modern MAE changes only from 274.08 to 273.64
yards. Rate MAE is unweighted within a season; its RMSE weights by attempts, so
those two displayed quantities do not have the usual unweighted ordering.

The four-week ranking policy gains 0.597 points of MAE (95% season-bootstrap interval
0.312–0.881) and 3.21 percentage points of top-10 point capture. Its full-history
MAE improves from 8.115 to 7.554, with 2.68 percentage points of capture gain.
The remaining-season point policy's modern gain interval is −0.652 to +1.561;
it therefore does not replace that reference despite a lower observed mean error.

The current next-game efficiency recipe blends profile ridge (alpha 1000) with
the two-year-half-life/400-prior-attempt career estimate. Current yardage recipes
are profile boosting (15 leaves) with a component blend for next game, enriched
boosting (7 leaves) with a component blend for four team games, and a profile
boosting (31 leaves) attempts × selected-rate component for remaining season.
These choices belong to the approved *earlier-year selection policy*, not a claim
that those fixed recipes independently passed every gate.

A component can fail standalone approval while the combined forecast passes its
own outcome test. The long-horizon YPA columns therefore retain the career
reference even though an approved integrated production recipe uses learned
efficiency internally. No unsupported standalone rate is exposed as validated.

The Brady 2009 preseason case remains 2,907 yards under the selected reference,
versus the archived old booster's 1,510 and 4,398 actual yards. This experiment
does not promote a new preseason policy or claim that example is solved. Career
evidence is preserved; uncertainty about season opportunity remains in the error.

Validation: 59 focused tests and 23 data/registry/ranking-view tests pass, including
chronological features/ranges, exact ranking outcomes, API serving boundaries,
publication continuity and prospective scoring. Frontend build and targeted lint
pass. Desktop/mobile browser checks cover passing forecasts, profiles, research
filters and the live four-week QB rankings. The release audit is preserved in
`qb_variations_delivery_validation_20260924_r1` with the exact delivery hash.

The approved 80% production ranges cover 85.03% / 82.33% / 81.23% of modern
next-game / four-team-game / remaining-season outcomes, respectively. Every
historical range uses only earlier years' forecast residuals.

The initial prospective score (`qb_variations_prospective_20260924_r1`) correctly
has zero scored outcomes and 580 pending forecast horizons. It does not count
this retrospective development run as prospective evidence.
