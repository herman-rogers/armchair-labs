# NextGen data reliability and full dataset training

> Planning document. Proposed work is not a statement of shipped behavior; use the [plans index](README.md) for scope and status.

Review date: October 2, 2026. This document records the NextGen review and the
data work needed to train models on the full available dataset. The immediate
direction is to preserve valid observations, repair information loss in the
transformations, and let learners evaluate broad inputs and interactions.
Historical single-stat significance and the production serving registry should
not decide which valid measurements a research model can see.

The published historical data has strong provenance and reconciles to its
captured sources. Its limitations are selective coverage, missing original
publication vintages, discarded NGS sample counts and weekly detail, and stale
published observations. These are data engineering problems with direct
consequences for training. Increasing model complexity cannot recover observations
that never reach the training dataset.

The findings below describe the reviewed release. The proposed contracts and
acceptance criteria are implementation work, not completed repairs or new model
approvals. Training can continue on existing valid data while these repairs expand
coverage; complete recovery of every source is not a prerequisite for research.

## What the review verified

The [current catalog](../../data/current.json) selected these releases when checked:

| Dependency | Reviewed release |
| --- | --- |
| Gold | `canonical_nextgen_20260923_r1` |
| Profiles | `nextgen_products_20260923_r2_profiles` |
| Analysis | `nextgen_qb_variations_20260924_r2` |
| Football observation cutoff | 2026 Week 2, captured September 22 |
| NFL tracking coverage | Passing/receiving 2016–2025; rushing residuals 2018–2025 |

Verification during the review established:

- All 702 offline tests passed; 35 tests were deselected. This is the test result
  for that review snapshot, not a certification of subsequent workspace changes.
- The published analysis loaded successfully with its manifest and dependency
  hashes verified across raw, enriched, gold, profiles and research evidence.
- Rebuilding the four NGS metrics directly from preserved provider files matched
  all 2,063 gold player-seasons with at least one observed NGS metric.
- All 14,077 profile tracking rows matched their corresponding gold rows. Many
  rows contain no tracking observations; this count is not tracking coverage.
- Every non-null preseason NGS feature matched the recorded prior season. There
  were no source seasons at or after the forecast season in that check.
- Local HTTP checks returned the catalog, measurements, forecasts, profiles and
  both ranking horizons. Each ranking horizon contained 841 candidates; CSV
  exported all 841. Unapproved forecasts and stale catalog tokens returned 409.

These checks establish integrity of the captured values and their transformations.
They do not independently verify the NFL's underlying tracking models, certify
historical publication times, or establish predictive accuracy.

**Subsequent workflow context:** the workspace now contains
[`research/refresh_nextgen.py`](../../research/refresh_nextgen.py),
[`src/patron/data/weekly.py`](../../src/patron/data/weekly.py), and the documented
`just nextgen-refresh` workflow. It coordinates completed-week capture, rebuilding
and publication. The catalog still selected the reviewed Week 2 release when this
document was written. The collector captures weekly production, snaps, schedules
and touchdown plays; it does not ingest current NGS passing/receiving/rushing
tables. Scheduling and successful publication of newer data remain separate from
the existence of this command. The earlier finding that refresh required manual
stage coordination must be read with this addition in mind.

## Reliability and predictive evidence

The four exposed NGS metrics are a small projection of much richer preserved
provider tables. Their observed coverage and consecutive-season stability were
recalculated from the profile release:

| Metric | Observed player-seasons | Consecutive-season correlation | Paired observations |
| --- | ---: | --- | --- |
| QB completion percentage over expected | 407 | 0.35 | 261 |
| RB rushing yards over expected per attempt | 405 | 0.20 | 232 |
| WR/TE average separation | 1,251 | WR 0.59; TE 0.41 | WR 584; TE 176 |
| WR/TE YAC over expected | 1,251 | WR 0.34; TE 0.47 | WR 584; TE 176 |

These are unadjusted Pearson correlations among players observed in consecutive
seasons at the same position. They are descriptive diagnostics, affected by
selection, and are not out-of-sample forecast scores. Separation measures distance
from the nearest defender at catch/incompletion; it does not isolate receiver
talent or describe every route. Expected-production residuals depend on provider
models. Preserve their definitions and units alongside the values.
[NGS dictionary](https://nflreadr.nflverse.com/articles/dictionary_nextgen_stats.html).

NGS includes players above provider opportunity thresholds. Its missingness is
therefore selective. A missing metric does not imply average ability, zero
production, or an ingestion failure. The provider offers weekly records and
season summaries, with `week == 0` identifying regular-season summaries; current
data is updated nightly according to its documentation.
[Provider contract](https://nflreadr.nflverse.com/reference/load_nextgen_stats.html).

None of the 24 retained individual-stat tests involving these four metrics passed
the promotion screen. For example, adding RB rushing yards over expected reduced
season-point MAE by approximately 0.32 points against the basic own-data model,
but slightly worsened it after adding the market baseline. This is evidence about
those recipes, targets and samples. **It does not justify deleting NGS columns
from a new nonlinear, representation, opportunity or efficiency experiment.**
[Individual-stat results](../../data/research/nextgen_qb_variations_20260924_r2/individual_results.parquet).

The published system mostly serves reference point forecasts. It separately
approves QB next-four-week rankings, QB passing-yard forecasts at three horizons,
and next-game passing efficiency. The QB four-week policy reduced modern
historical MAE from 7.64 to 7.04 points; the season-bootstrap interval for the
improvement was 0.31–0.88 points. These are retrospective development results,
not demonstrated market advantage.
[Ranking evidence](../../data/research/nextgen_qb_variations_20260924_r2/ranking_evaluations.json).

Errors are larger in the population most relevant to active fantasy decisions.
Recalculating the served policies from retained predictions gives:

| Position | Four-week MAE across all candidates | Four-week MAE with observed workload |
| --- | ---: | ---: |
| QB | 7.04 points | 17.80 points |
| RB | 6.98 points | 18.28 points |
| WR | 6.81 points | 14.92 points |
| TE | 4.94 points | 12.48 points |

These are equal-season averages for 2019–2025 at the Week 2 origin. The workload
cohort uses the existing cutoff-defined thresholds: QB at least 10 attempts,
RB at least 5 carries plus targets, and WR/TE at least 2 targets per elapsed team
game. The calculation uses `policy` for approved QB next-four forecasts and
`reference` for the other positions. Report both population-wide and cohort
accuracy in future experiments. The saved
[prospective evaluation](../../data/research/qb_variations_prospective_20260924_r1/report.json)
had zero scored outcomes and 580 pending rows.

This review assessed the serving system, not every research model. Existing
[Future Player Lab](../../experiments/future_player_lab/README.md), its
[deep feature and ensemble campaign](../../experiments/future_player_lab/deep/README.md),
and the [representation lab](../../experiments/representation_lab/README.md) already
support broader inputs and model families. The deep campaign documents 2,929
candidate inputs and LightGBM, XGBoost, CatBoost and other learners. Those reports
have their own cohorts and validation contracts; the serving review does not
supersede their results. Extend their data adapters and reproducibility machinery
as the observation contract improves.

## Full dataset training contract

Make all valid, task-appropriate information available at each forecast cutoff.
Retain the full captured history in storage, expose broad input views to learners,
and record any learner-specific reduction. A compact baseline can remain useful
without becoming the input ceiling for every experiment.

| Concern | Required behavior |
| --- | --- |
| Predictive screening | Do not filter valid inputs by earlier p-values, single-feature correlations, top-80 lists, published model importance or serving permission. Evaluate interactions and representations within each experiment. |
| Sparse and binary inputs | Retain them with coverage and missingness information. Do not globally drop a source because it covers only recent years. |
| Constant or entirely missing fields | Preserve the field contract. A learner may omit an unusable column based only on its training fold, recording the omission and reason. |
| History | Offer all earlier eligible seasons. Recent windows, recency weights and dimensionality reduction are explicit candidate representations, selected using earlier data. |
| Candidate population | Retain cutoff-eligible rookies, returners, low-production players and players missing optional sources. In-season pools should also admit players first observed by that cutoff. Audit every exclusion. |
| Unknown versus zero | Preserve observed zero, unknown, undefined denominator, source unavailable and quarantined values distinctly where the evidence supports that distinction. Do not invent a missingness reason. |
| Labels | Store future outcomes separately, with target-specific observation masks. Unknown outcomes cannot become zero training labels. |
| Time | Exclude future observations, later roster states, future career endpoints and same-season final aggregates from earlier forecasts. |
| Source defects | Keep original bytes and quarantine only affected measurements and dependent transforms. Retain independently valid data for the player. |
| Model outputs | Keep provider estimates explicitly labeled as modeled measurements. Saved project predictions can be separate stacking inputs only when generated without the row's outcome and available under the declared timing contract. |

This separates **data validity**, **research input eligibility**, and **production
serving approval**. A failed past prediction claim changes its evidence record;
it does not make a valid observation unavailable to all future models. The current
gold feature allowlist should evolve through source definitions, units, lineage
and timing review, with every excluded field accounted for. It should not become
a permanent list of handpicked predictors. Undocumented fields remain preserved
and visible in the inventory until their semantics are resolved.

The dataset spans several grains: plays, player games/weeks, team games, seasons,
college stints and dated events. Keep normalized tables at those grains, then
build deliberate training views. Joining all tables directly into one wide frame
would multiply rows and distort sample weights. Player IDs serve joins and
grouping by default; identity embeddings require an explicit experiment and
evaluation on new players.

## Transformation repairs

### Preserve the complete captured NGS tables

[`build_nextgen_features`](../../src/patron/metrics/enrichment.py) currently filters
to regular-season summary rows and selects four metrics. Gold and
[`profile_tracking.py`](../../src/patron/metrics/profile_tracking.py) inherit that
projection. Recover the full captured passing, receiving and rushing tables
before deriving this existing convenience view.

Add normalized NGS observation tables, separated by stat type, retaining weekly
and summary records, provider identities, team/stint fields, source values,
definition versions and capture references. Candidate keys include provider
player ID, season, season type, week, stat type and capture version. Validate
actual provider cardinality; retain any necessary stint/aggregate discriminator
instead of silently keeping the first duplicate. Keep season summaries separate
from weekly facts so a query cannot add both into a total.

The preserved files already contain opportunities discarded downstream:

| Source | Recoverable opportunity fields | Observed range in captured season summaries |
| --- | --- | --- |
| Passing | Attempts and completions | Attempts 131–733 |
| Receiving | Targets and receptions | Targets 43–191; receptions 15–149 |
| Rushing | Rush attempts | Attempts 85–378 |

These ranges describe this capture, not formal qualification thresholds. Preserve
all other supplied metrics too, including expected and observed quantities,
air-yard measures, cushion, time to throw, aggressiveness and rushing context.
Their predictive usefulness remains an empirical question. These public tables
are aggregates; the repository does not thereby acquire raw player coordinates.

**Acceptance:** inventory every source column as retained, transformed or excluded
with a reason; reconcile the four existing metrics exactly; reconcile source keys
and counts; retain all recoverable weekly records and denominators; expose
coverage by season, position and metric. Validate denominator semantics before
calling a provider opportunity count the exact sample for every metric.

### Correct aggregation and missingness semantics

Store numerator, denominator and observed exposure alongside a derived rate when
the source permits it. Compute compatible pooled rates from pooled numerators and
denominators; never average weekly percentages or residuals without the appropriate
weights. Preserve provider season aggregates as their own observations. Weekly
qualification thresholds and missing weeks may prevent exact reconstruction of
those season aggregates; report the discrepancy and coverage instead of forcing
agreement.

Retain distributions, chronology, zero counts, observation counts and gaps for
learned representations. A missing player-week can mean a bye, no participation,
no eligible tracking sample or a missing capture. Infer zero participation only
when the source contract and completeness evidence support it. Never fill all
absent rows with zeros to obtain a rectangular tensor.

**Acceptance:** uneven-exposure examples produce correctly weighted rates;
unknown denominators and incomplete totals remain unknown; negative residuals
and legitimate zeros survive; joins cannot inflate row counts; traded players
and position changes preserve identity and appropriate historical context.

### Preserve timing and revisions

Record event/observation time, provider publication time when known, capture time,
and forecast issue time separately. Version corrected captures and retain their
predecessors. A current-season `week == 0` row is a season-to-date summary at its
capture, not automatically a completed season. A final season summary cannot be
joined to a midseason historical forecast merely because its season matches.

Historical event dates alone do not certify original availability. Keep missing
publication times unknown and mark reconstructed experiments accordingly. Future
captures should support true as-of selection without overwriting earlier vintages.

**Acceptance:** adding future rows or revising later snapshots cannot change
earlier candidate membership, features, preprocessing or forecasts. Every feature
has an observation window, lineage and availability rule; every target has an
end time and maturity status. Current NGS ingestion explicitly joins this capture
contract rather than relying on the historical 2016–2025 bundle.

### Bind caches to inputs and transformations

[`cached_frame`](../../src/patron/data/derived.py) keys artifacts by name and season
list. [`pipeline.py`](../../src/patron/pipeline.py) uses that cache for NGS, while
[`data_pipeline.py`](../../research/data_pipeline.py) restores historical derived
artifacts before enrichment. Unchanged season lists do not prove unchanged data.

Use a fingerprint covering input content hashes, transform implementation and
parameters, schema/definition version and relevant configuration. Write cache
results atomically with their provenance. A changed input, scoring configuration
or transform must invalidate the result. Distinguish reused verified artifacts
from computations actually replayed in build reports.

Add an offline verification mode that recomputes recoverable NGS transformations
from preserved provider tables without accepting derived-cache hits. Reconcile
results to gold and explain intentional corrections. Legacy enriched imports
without raw inputs remain explicitly labeled imports, not claimed raw replays.

**Acceptance:** changing source bytes or transformation code while keeping the
same seasons cannot return the prior cache entry. An unchanged dependency set
can reuse its verified result. Interrupted writes never appear complete.

## Dataset and model development

Build a versioned training release from the expanded observation tables. Its
manifest should bind source/transform hashes, feature definitions and units,
candidate rules, forecast origins, horizon definitions, label masks, split
assignments and column-level coverage. Keep data preparation reproducible
independently of any particular model or production publication.

Expose raw-history, broad tabular, and learned-representation views over the same
candidate/date/target keys. Recover additional source fields through their
contracts; do not describe an existing 2,929-column projection as the complete
raw dataset. Compare each view on matched rows and retain the original inputs
when adding embeddings or clusters, so compression can be evaluated explicitly.

**Model terminology:** ordinary gradient-boosted regression/classification/ranking
trees are supervised: they optimize an outcome or supplied objective. Unsupervised
learning can discover clusters or components without future production labels.
A combined workflow can learn representations first and feed those alongside
observations into boosted trees. If a tree learns a reconstruction target or
pseudo-label, record how that target was created and the objective being fitted.
[Scikit-learn unsupervised methods](https://scikit-learn.org/stable/unsupervised_learning.html),
[XGBoost objectives](https://xgboost.readthedocs.io/en/stable/parameter.html).

Proposed experiment tracks:

| Track | Purpose | Evidence to retain |
| --- | --- | --- |
| Broad boosted trees | Learn nonlinear effects and interactions across valid observation families; reuse existing histogram boosting and deeper tree engines. | Matched predictions against compact baselines and existing broad models; fit settings, failures and coverage. |
| Unsupervised representations | Learn components, player groupings or compressed histories through PCA, clustering or autoencoders. | Stability across eras/cohorts, reconstruction where applicable, and checks for representations dominated by missingness or provider era. |
| Representation plus prediction | Add learned representations to raw/tabular inputs and test whether they help future outcomes. | Matched ablations with and without the representation, fitted inside each training fold. |
| Multiple football outcomes | Learn opportunity, conditional efficiency, appearances, production and fantasy scoring with appropriate labels. | Separate target masks, units, denominators, calibration and horizon-specific results. |
| Market comparisons | Test both own-data and market-augmented models using genuinely dated market observations. | Identical-row baselines; distinguish added predictive accuracy from acquisition or lineup value. |

Use learner-appropriate missing-value handling. Histogram boosting, for example,
supports missing values natively; PCA and other learners need explicit handling
fitted on training data. Preserve coverage indicators, but check whether apparent
gains depend on historical source changes that may not generalize.
[Histogram boosting documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html).

All learned preprocessing, imputation, scaling, clustering, embeddings, feature
selection, early stopping and hyperparameter selection belong inside chronological
training/validation boundaries. Unsupervised preprocessing fitted to future test
seasons still sees future distributions. Outcome windows must mature before a
label can train a fold; overlapping horizons require appropriate temporal gaps.
Cluster uncertainty by season and disclose repeated-player/overlapping-forecast
dependence. Novel-player evaluation should complement ordinary future-season
evaluation where generalization to rookies is a goal.

Keep full earlier history available; evaluate shorter windows or recency weights
as explicit alternatives. Use earlier validation seasons to choose them. Report
the complete candidate population and workload, rookie, position, source-coverage
and era cohorts. Preserve failed trials and search budgets. Familiar historical
years support development; frozen prospective forecasts provide the next
independent evidence.

## Implementation order and completion criteria

1. **Publish a source and transformation inventory.** Map every captured field,
   grain, denominator, time rule, join and downstream loss. Reuse current gold and
   broad labs immediately; record what is missing from each training view.
2. **Recover NGS observations and repair transformations.** Add complete provider
   tables and opportunity counts, preserve weekly detail, fix aggregation semantics
   and implement source/code-based cache invalidation. Reconcile the old projection
   before building expanded training views.
3. **Release the broad training dataset.** Include all eligible history and
   candidates, optional-source masks and separate labels. Run temporal invariance,
   cardinality, reconciliation and reproducibility checks. Every omitted source
   field and excluded candidate needs an explicit reason unrelated to past
   predictive significance.
4. **Extend the existing model experiments.** Compare broad trees, representations
   and their combinations on the same examples. Run comparable old-view/new-view
   experiments to distinguish gains from recovered data from gains due to tuning.
   Save fold definitions, fitted preprocessing, settings, predictions and failures.
5. **Operate refresh and prospective scoring.** Extend the new refresh workflow
   with NGS capture and source-specific freshness checks. Surface observation age
   independently of ESPN connectivity, retain capture revisions, and score frozen
   forecasts when outcomes mature. A successful data refresh need not promote a
   different model; experimental training remains available without serving approval.

The data work is complete when every available source field is accounted for,
valid observations survive into reproducible research views, denominators and
missingness remain interpretable, future data cannot alter earlier examples, and
cache reuse follows exact dependencies. Model success is evaluated separately:
broader inputs enable a fair experiment but do not guarantee a better forecast.
