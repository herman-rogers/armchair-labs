# Proposed dynamic QB passing-yard experiment

Status: design for the next experiment, September 24, 2026. No results or serving
approval. The motivating historical cases and booster results have already been
inspected; reusing those years is retrospective research, not independent confirmation.
See [the audit and literature review](../docs/research/qb_passing_forecasts_review_2026-09-24.md).

Implementation follow-up: [the fixed run protocol](qb_passing_run_protocol.md) and
[QB passing system report](../docs/history/qb_passing_system_2026-09-24.md) now cover the
generic history model, dated constraints, dashboard reference forecasts and
prospective scoring. Boosting remains shadow research. The richer starting-role
and medical-state ambitions below are not validated by workload proxy labels.

## Questions and outputs

1. Given information available now, what is the probability of being active and of
   holding the primary passing role? Role and medical status have distinct labels.
2. Given the role and availability scenario, what passing opportunity and execution
   are expected? Publish attempts, YPA estimates, exposure and uncertainty.
3. Across possible role/availability scenarios, what passing-yard distribution is
   expected for the next game, next four scheduled team games and remaining season?

For forecast issue time t and future game g, use a conditional mixture:

`E(Y_g | information_t) = sum_s P(state_g=s | information_t) × E(Y_g | state_g=s, information_t)`.

States distinguish primary passer, reserve/competition, unavailable and unresolved
status as data permit. Inactive/absent games have zero production but no observed
YPA. Jointly or conditionally model attempts and execution: independently multiplying
mean attempts by mean YPA generally omits their dependence. Never apply availability
twice to an unconditional production prediction.

## Information and data contract

Retain all historical player observations, including backups, injury-shortened
seasons, roster exits and years with zero production. Use expanding training windows
over all earlier completed, trustworthy labels. No blanket modern-only training.

Maintain separate timestamps for event occurrence, source publication, ingestion,
forecast issue and outcome horizon. Future roster metadata and retrospectively
revised provider features are not automatically available at an earlier cutoff.
Preserve source/version hashes and state whether a feature is reconstructed.

Generic history tier: all covered schedules, passing counts and historical player
profiles. Rich tier: matched rows with dated role, injury, depth, transaction and
context evidence. Missing rich data stays unknown; retain generic predictions and
report coverage. Source-rich gains are measured against generic models on the same
rows, then evaluated as a complete policy with explicit fallbacks.

Before fitting, audit dated retirements, releases, transfers of the starting role,
returns and long-term absences. Brady's 2023 missing retirement restriction shows
why candidate retention must be combined with correct eligibility information.
Repair the general historical source path in a new gold version, preserve originals,
and record affected cases. Do not hand-correct only the named examples.

## Candidate models

- Prior-season reference, retained for continuity.
- A lean dynamic reference using dated role probabilities, team passing volume and
  exposure-weighted multi-season execution with shrinkage. Determine regularization
  and recency weight using earlier inner validation only.
- The fixed historical booster as a preseason comparator; a separately specified
  weekly booster trained on the same available information for weekly comparisons.
- A small conditional role/availability model plus workload/execution estimates.

Do not assume that a more elaborate state model wins. Direct regression and the
conditional model receive matched information. Treat college/draft inputs as priors
for sparse NFL history, with coverage flags and a matched incremental test.

Execution learns from actual pass attempts, with suitable context adjustments and
aging/recency uncertainty. A seven-attempt injury fragment provides little evidence
against hundreds of earlier attempts. Injury or recovery can affect future execution,
but that effect must be estimated rather than assumed absent or inferred from zero
season production. Coaching changes, receivers, protection, opponent, weather and
game environment are additional candidate causes, tested in a small declared set.

Role updates may be abrupt when an assignment is announced. Scoring streaks alone
do not establish a job change. Missing injury news does not mean healthy, and a
box-score absence does not identify a medical cause. Known unavailability constrains
the affected future games while preserving the separate execution estimate.

## Forecast origins and scoring

Freeze preseason forecasts permanently. Freeze weekly forecasts before each decision
deadline and retain any later news-triggered revision as a new version. Evaluate
revisions on future games only; previously accumulated yards are displayed separately.
Full-season pace and remaining-season expectation cannot share one error label.

Primary outputs require different scores:

- Expected yardage/attempt totals: squared error for the mean, with MAE, bias and
  tail misses as complementary diagnostics. If medians are published, declare MAE
  as their primary loss rather than calling them expected totals.
- Yardage distributions: CRPS or declared quantile losses, plus interval coverage
  and width overall and for cutoff-defined role/sample groups.
- Participation and role probabilities: Brier score, log loss and calibration.
- Execution: player-weighted and attempt-weighted rate errors reported separately,
  with denominator coverage. Undefined rates remain undefined.
- Responsiveness: accuracy of forecasts issued after dated promotion, return or
  absence evidence; time from publication to revised forecast; and bias over a
  declared number of subsequent games. Stable-role false alarms remain in evaluation.

Score all candidates in the unconditional production evaluation. Do not remove
injury seasons after observing them. A conditional-performance score answers a
different question and must be labelled as such. Any use of realized future attempts,
starts or health to explain misses is an oracle diagnostic, never a deployable forecast.

Use identical cases and horizons for model comparisons, season-level paired results,
and sensitivity to player/team dependence and overlapping forecast windows. All
tuning and model choice use earlier inner folds. Stratify using only cutoff-known
information; report unobserved role/medical labels as coverage gaps.

## Decision before running

Inventory actual source coverage, then freeze the precise origin schedule, labels,
small model set, hyperparameter search, practical improvement thresholds, guardrails
and multiple-comparison family in an immutable run protocol **before fitting**.
This design intentionally does not invent promotion thresholds from the audit gains.

Require gains against the strongest matched simple reference, acceptable uncertainty
and subgroup behavior, and a prospective shadow period on future unplayed games.
Record its starting issue timestamp before any target games occur. Already observed
2026 games are not a prospective holdout. Preseason claims need a future preseason
cohort; weekly evidence does not automatically authorize preseason or fantasy claims.

Report negative and inconclusive findings alongside useful ones. Keep all artifacts;
only explicitly approved target/horizon/population uses enter everyday analysis.
