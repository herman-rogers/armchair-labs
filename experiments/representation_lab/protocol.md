# Frozen first experiment protocol

Target: preseason full-season league fantasy points, separately QB/RB/WR/TE.
Use the corrected individual-stat r4 matrix and its canonical_targets gold lineage,
with hashes pinned in config.json. All 17,396 complete candidates remain eligible,
including zero outcomes, rookies and market-only candidates. No 2026 outcomes.
Forecast dates and available evidence dates are checked before fitting. Current
source vintages are historical reconstructions, not guaranteed archived vintages.

The inventory has 1,339 admissible columns (40 two-valued columns descriptively).
Fourteen market columns are isolated; the own-data inventory has 1,325 columns.
Do not use individual-screen results, globally detected aliases, importance or
coverage thresholds. Registry admissibility is a data contract, not signal proof.
The contract was curated from the completed historical data; this qualification
also applies to this retrospective experiment. Constant and unavailable registry
entries are not counted as admissible. No supervised discovery uses future labels.

The raw arm uses 13 weekly measured statistics for three prior calendar seasons,
plus annual totals at lags 4–25 and 33 cutoff-known scalar observations. Missing
weeks and unknown totals remain missing. Annual sums require every recorded value
to be known and are not a claim that every game was observed. Source seasons must
strictly precede forecast season. Older raw records are not fed to the recent-only
control; the main raw arm retains their annual observations. Rich summaries,
weighted opportunity, age multipliers, forecast formulas and market ranks are not
inputs to the raw arm. Static fields include draft and latest measured college
production; this is not a complete raw college/tracking feed experiment.

Fixed recipes (defined in methods.py before reading results): compact Basic tree;
old top-80 Spearman gate with binary exclusion; unfiltered own-inventory trees
(7/15 leaves), ridge, PCA+SVR, tree-selected+SVR; raw trees (7/15 leaves), last-season
raw tree, ridge, generic signed-log/square basis+ridge, PCA+SVR, tree-selected+SVR;
full-inventory tree with market inputs. Raw and inventory selectors compete
separately, plus an own-data joint selector. Market is a separate benchmark.
Each representation/predictor combination earns its own chronological score.
There is no presumption that tree-selected inputs suit SVR. The PCA representation
is learned; neural networks and symbolic interaction search are outside this run.

All models fit points per scheduled game (16/17) and return nonnegative season
points. No actual games, realized role, availability or outcome determines a
prediction. Every method uses the same target transform and output bound.
Imputation, scaling, clipping relative to learned scale, PCA and supervised tree
selection are fitted afresh inside each training fold. Missing flags accompany
every feature, even if missingness was not previously observed. No coverage filter
or binary exclusion is applied except in the explicitly named legacy control.

Train on all earlier seasons from 2004. Fixed-recipe expanding-window forecasts
begin in 2005. Score outer years 2008–2025. At the start of each outer season choose
the pipeline using mean season MSE on the previous three chronological validation
folds. A validation fold itself trains only on still earlier seasons. Reusing
cached fixed-recipe fits is computationally equivalent to those nested fits.
Selection finishes before scoring current-season labels. No random row validation,
early stopping split, retrospective best-model choice, or future tuning. The grid,
seed, search family and data contract are fixed before execution.

Primary metric: equal-season MSE. Secondary: equal-season MAE. Store each model's
forecast, each annual loss, population/era breakdowns and each selection decision.
Compare against compact Basic, old gate, and unfiltered raw/inventory trees on
identical rows. Bootstrap paired annual loss changes (4,000 draws); intervals are
unadjusted and descriptive, with no automatic promotion or "proven signal" gate.
Repeated experimentation on familiar years limits independent confirmation.

Group permutation diagnostics use held-out forecasts only after model choice and
never feed back into fitting. Correlated substitutes and unrealistic shuffled
combinations limit interpretation. Tree-selected feature recurrence and adjacent
split co-occurrence generate hypotheses; they do not establish interaction gains.
No discovered interaction is declared beneficial without an independent ablation.

This lab imports only the existing safety helper as project executable code;
all forecasting inputs come from pinned artifacts. Source/config/package versions,
predictions, choices and diagnostics are saved inside a create-only run directory.
The Python guard is an accident barrier; native writes are not universally audited.
Protected inputs and publication pointers must retain their hashes at completion.
