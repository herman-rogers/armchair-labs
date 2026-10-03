# Individual statistics and combinations, September 23, 2026

This audit implements the evidence portion of `docs/architecture/nextgen_system_design_2026-09-23.md`.
It does not publish a new default forecast or replace the dashboard policy. The
referenced profile study is an archived benchmark requiring revalidation, not
proof that its exact gains survive the current gold contract.

Before fitting, derive a new immutable gold release from published gold r5 and its
pinned enriched r4. Quarantine demonstrably invalid 2003–2008 receiving-target
history and all affected annual, weekly, three-season and cumulative target
derivatives. Preserve candidates, scoring outcomes and every historical year.
Verify provenance and consistency. Do not mutate r5 or the current catalog.

Inventory every numeric gold feature, the 97-entry metric catalog, numeric legacy
enriched columns, original 218 profile predictors, reconstructed named formulas,
and explicitly retired formulas in the metric graveyard. Classify outcomes,
identifiers, model outputs and invalid/unreconstructable formulas separately.
Every catalog entry must receive an explicit disposition; excluded is not harmful.
Register definitions, units, dependencies, transformations and available years.
Never feed old fitted predictions or unsafe legacy composites into the new fits.

Rebuild profiles from corrected gold weeks, identities, college and injury tables.
Keep full career history alongside recent summaries and observed participation
bands. All historical candidates and zero-production outcomes remain. Market-only
candidates stay in prediction/decision universes. Separate position fits, with
rookie/returner/market-only indicators and subgroup reporting. All training folds
use all earlier accepted years, not a rolling window. No 2026 outcomes are graded.

Primary target: full-season league points divided by nominal season length;
predictions are converted back to points, floored at zero, and constrained to zero
for a documented full-season absence. No partial-absence double discount.
This is not a universal test of four-week predictions, medical availability,
conditional active-game scoring, or trade/waiver profitability.

Individual feature tests use ridge alpha 100, training-only median imputation,
missing indicators and scaling. A numeric value and its missingness indicator are
one feature block and must be added/removed together. Require three earlier years
with at least ten finite observations and actual training variation before an
individual effect is scored. Retain all earlier rows regardless of coverage.

For each admitted statistic, run:

1. Addition to the corrected profile study's Basic baseline (production, usage,
   age, draft capital, population and era), and Basic plus cutoff-safe ECR.
2. Removal from its own coherent feature-family combination above the same
   baseline. Original profile predictors additionally receive removal from the
   full profile, so a constituent can hurt despite a beneficial family mean.
3. Exact duplicate/alias accounting. Duplicated columns are not independent votes;
   record which model contexts contain aliases and compare train-deduplicated fits.

An individual feature already in a baseline receives a removal test, not a fake
zero-valued addition test. Additions and deletions answer conditional, model-specific
questions; correlated substitutes can conceal usefulness. Removal gain is defined
as full-model error minus refitted reduced-model error. All comparisons are paired.
Use algebraically exact block ridge updates for the exhaustive screen; test them
against explicit sklearn refits, including missingness and correlated inputs.

Revalidate Basic/Career/Profile and market-informed counterparts with the original
fixed ridge and histogram booster (120 iterations, .05 learning rate, 15 leaves,
minimum leaf 30, L2 10, max_bins 63, no early stopping). Test declared combination
controls: training-only exact-column deduplication; dropping cross-position raw
passing-volume features for non-QBs; dropping college exposure-weighted transforms;
and dropping raw career cumulative volume while retaining rates/exposure. These
are separate recipes, not post-result chosen winners. No global feature selection
or tuned combination is evaluated as if it had been chosen in an earlier season.
The exhaustive individual screen is linear; do not generalize its removals to the
nonlinear profile model without the corresponding refit.

Report full evaluable history first, per-position and population results, and
fixed-era sensitivity (2005–2012, 2013–2018, 2019–2025). Eras summarize results;
they never restrict the primary training data or stored player careers.
Report finite coverage, substantive variation, active folds, MAE, MSE, equal-season
paired gains, 10,000 season-bootstrap intervals and leave-one-season-out means.
Use two-sided season-sign-flip p-values and Benjamini–Hochberg q-values across the
exhaustive primary screen; their dependence/exchangeability limitations remain.

Before results: a one-season-point MAE improvement is the practical screening
threshold for preseason forecasts, not established team value. Candidate evidence
also needs at least five admitted years, a positive 95% lower endpoint, q <= .05,
nonnegative mean squared-error improvement, and positive leave-one-year-out means.
Anything passing remains a research candidate, not approved serving evidence.
Report smaller effects and uncertainty; do not hide negative/null findings. No
selection-profit claim is authorized by lower prediction error.

Save immutable registry, complete individual results/folds, model predictions,
combination diagnostics, correction audit, code and protocol snapshots, manifests
and a searchable standalone report. Preserve original studies and frozen outputs.
Tests cover cutoff filtering, missing/zero semantics, transitive target masking,
full-history training, exact block refits and held-out target invariance.

Before fitting, the user additionally requested scoring concentration and variance
checks. Retain existing volatility, floor, peak/entropy and weighted average
features. Add prior-season, recent-three-season and full-career median, population
standard deviation, coefficient of variation (positive mean only), best-two-week
share of positive points, mean excluding the best two weeks (at least three
observations), and observation count. Add between-season scoring-rate dispersion.
These summarize observed weeks, not guaranteed active games or medical risk.
Preserve zeros and negative scores; use positive points only for the concentration
denominator. Two-week concentration has a different baseline over a 17-week season
than over a multi-season career; report its period and sample size. All are tested
as individual additions/removals, not automatically promoted as draft heuristics.
