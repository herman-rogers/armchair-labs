# QB passing forecasts: screen clarification, challenger audit and dynamic forecasts

Reviewed September 24, 2026. The Evidence screen is updated. The saved passing-yard
models were reproduced and reviewed; no challenger was promoted and no forecast
release was replaced. The proposed dynamic experiment below is not implemented.

The useful next step is to forecast workload and execution conditional on current
information, preserve each forecast's issue time, and evaluate changes after new
information arrives. An unexpected injury can make a season-total forecast miss
without demonstrating a collapse in passing ability. That distinction should change
our model and evaluation design, not remove inconvenient season outcomes.

## Screen changes

- Error units are explicit: season passing yards, season points, yards/attempt, etc.
- The page explains the preseason horizon, reference recipe and equal-season MAE.
- Counts are labelled player-seasons, including backups and nonappearances.
- Baseline details omit self-comparison gains, zero-width improvement intervals and
  duplicated error columns. Challenger improvement intervals remain in Research.
- Passing yards include prior-workload groups: 300+ attempts, 1–299, observed zero
  and unknown prior attempts. These are observed-history groups, not starting-role
  labels. No future production, injury or start determines group membership.
- Probability outcomes retain their Brier-score explanation and calibration table.

The new workload diagnostics read the published, hash-verified prediction and feature
artifacts. They do not refit forecasts or change model eligibility. Original research
files and the current catalog remain unchanged.

## Reproduction and assessment of boosting

The [reproduction runner](../research/qb_passing_review.py) refit baseline, ridge and
histogram boosting using the saved implementation and exact saved features for every
completed QB passing-yard fold from 2007 through 2025. All **57 model/fold outputs
reproduced exactly** (maximum absolute difference zero). Training includes all earlier
completed candidate years, beginning in 2004. Modern is an evaluation slice.

[Preserved report](../data/research/qb_passing_review_20260924_r1/report.json) and
[manifest](../data/research/qb_passing_review_20260924_r1/manifest.json) retain source
hashes, reproduction checks, population/workload results, named cases and related
outcomes. Run on the currently published source:

```sh
.venv/bin/python research/qb_passing_review.py --version NEW_RESEARCH_VERSION
```

This is an audit of reused historical evidence, not a new independent holdout.

| Evaluation | Player-seasons | Prior-season MAE | Ridge MAE | Boosting MAE |
| --- | ---: | ---: | ---: | ---: |
| 2007–2025 | 1,960 | 788.6 | 741.2 | 707.6 |
| 2019–2025 | 794 | 786.4 | 676.4 | 661.7 |

All errors are season passing yards, with each evaluated season weighted equally.
Boosting improves on the reference by 81.0 yards over full history (95% season
bootstrap interval 44.9–120.5), and 124.7 modern (71.2–174.3). It improves all seven
modern seasons and also reduces squared error. Against ridge, the modern gain is
only 14.7 yards, with interval **−17.0 to 39.2**. These intervals are exploratory,
unadjusted for the overlapping comparisons and repeated research on these years.

The fixed booster uses 120 iterations, learning rate .05, 15 leaves, minimum leaf
30, L2=10 and squared-error loss. Inputs include prior production, career/recent
rates, age, draft position, era, population, historical participation and captured
dated absence facts. It is not a weekly updating role model. Its features do not
include a complete dated history of starting assignments, recoveries and exits.

The original protocol deliberately kept every challenger in research regardless of
its result. Therefore the correct conclusion is **promising for this outcome, still
unapproved**, not “boosting failed every target.” Current fantasy-ranking promotion
results are a separate experiment against stronger references.

The gains are uneven. Modern results by information available at the cutoff:

| Prior-season workload | n | Reference MAE | Boosting MAE | Reduction |
| --- | ---: | ---: | ---: | ---: |
| 300+ pass attempts | 204 | 1,216.4 | 1,051.7 | 164.7 |
| 1–299 pass attempts | 326 | 652.1 | 645.8 | 6.3 |
| Observed zero attempts | 27 | 211.0 | 288.7 | −77.7 |
| Prior attempts unknown | 237 | 666.4 | 382.6 | 283.7 |

These exploratory groups describe prior volume; low volume is not proof of injury
or promotion. Independently, modern rookie MAE falls from 792.2 to 453.7, whereas
returner MAE falls from 839.4 to 777.5 and its improvement interval crosses zero
(−2.4 to 123.7). Much of the overall gain is against a weak no-history fallback.
Full-history MAE actually gets worse for the 1–299 and observed-zero groups.

| Case | Reference yards | Boosting yards | Actual yards |
| --- | ---: | ---: | ---: |
| Brady 2008 | 4,806 | 3,219 | 76 |
| Brady 2009 | 76 | 1,510 | 4,398 |
| Mahomes 2018 | 284 | 888 | 5,097 |
| Jackson 2019 | 1,201 | 1,206 | 3,127 |

The Brady rebound concern is directly confirmed. Having career features in a model
does not ensure that it handles a short injury season appropriately. The promotion
examples also remain large misses despite the improved aggregate result.

A separate source gap appears among the largest errors: Brady 2023 has a boosting
forecast of 4,215.7 yards and an actual total of zero. His August 25 cutoff row has
an inferred prior team and no known-availability cap, although his retirement was
announced February 1. This is missing decision-time information, not an unforeseen
in-season event. Repair dated roster exits for all candidates in a new version,
rather than patching this one famous case or changing the saved result.
[Contemporaneous NFL announcement](https://amp.nfl.com/news/tom-brady-retirement-23-seasons-in-nfl-buccaneers-patriots).

Modern boosting also improves pass-attempt MAE (111.6 → 92.4) and passing YPA MAE
(1.560 → 1.319; 512 defined outcomes). Those are independently fitted targets,
not a causal decomposition of yardage error. Role, availability, team volume,
opponents, execution and random play outcomes can all matter. The present labels
cannot establish that all misses are caused by injury or role changes.

## Targeted field review

This follows the [broader September 23 literature review](nfl_player_research_literature_review_2026-09-23.md).
Sources were checked September 24. Original research and producer documentation
are distinguished below; no external model was independently replicated here.

**Direct passing-yard research.** Czasonis, Kee, Kritzman and Turkington's February
2026 manuscript predicts next-season passing yards per game using relevant prior
player-seasons and reports prediction-specific diagnostics. Its evaluation includes
QBs with at least 100 attempts in the outcome seasons, 2021–2025; that excludes many
availability/job failures we must forecast. Its reported ranking separation is not
a matched MAE comparison against our booster. Useful idea: show which historical
contexts support an individual forecast. Limits: selected population, different
target, and insufficient evidence here to adopt its claimed reliability measure as
a calibrated interval. [Author manuscript hosted by MIT](https://mitsloan.mit.edu/shared/ods/documents?PublicationDocumentID=10821).

**Updating rather than replacing career evidence.** Kevin Cole's 2019 PFF study
describes Bayesian updating of quarterback grades and EPA, combining prior
expectations with observed performance. This supports testing sample-aware updates
that retain substantial older evidence when the new sample is small. The accessible
article is an introduction with proprietary inputs, not a reproduced passing-yard
benchmark. [Original practitioner study](https://www.pff.com/news/nfl-pff-data-study-combining-grades-and-stats-to-enhance-quarterback-projections).

**Forecast the remaining horizon explicitly.** Synthefy's September 2026 passing-yard
example predicts remaining game yards from quarterback history, team/opponent
context and current-game state, then adds yards already recorded. It illustrates
the useful distinction between an original forecast and a forecast issued after
new information. This is a vendor-authored intragame demonstration, not evidence
for season-long role forecasting. Its 2025 betting strategy was refined on the
same season used to report returns; the authors identify the results as exploratory.
No betting-performance claim is adopted here. [Producer's methods and limitations](https://www.synthefy.com/blog/nori-nfl-quarterback-passing-yards).

**Match evaluation to the forecast's meaning.** Gneiting explains why squared error
targets the conditional mean and absolute error the conditional median. A useful
expected-total forecast can differ from the most typical outcome when injury and
role scenarios create a skewed distribution. Keep MAE as an interpretable diagnostic,
but declare whether we are forecasting a mean, median or distribution and use
appropriate losses. [Point-forecast paper](https://arxiv.org/pdf/0912.0902).

**Dynamic uncertainty is a separate task.** Gibbs and Candès develop online conformal
prediction that adapts to changing distributions. Their examples concern financial
volatility and COVID counts, not NFL yards. The method motivates testing adaptive
interval calibration after role changes; it does not guarantee coverage for a
particular injured QB or solve delayed, overlapping season-total labels.
[JMLR paper and code](https://jmlr.org/papers/v25/22-1218.html).

**Injury uncertainty should not become a claim of impossibility.** A 2021 American
football study found predictive signal in surveys and player statistics, using
one NCAA team in one season with substantial missing data. It is not an NFL
validation and its private monitoring inputs are absent here. It cautions against
both “injuries are completely unpredictable” and a claim that public box scores
can reliably forecast the timing of an individual injury.
[Original study, author manuscript](https://wmit-pages-prod.s3.amazonaws.com/wp-content/uploads/sites/13/2023/02/17142402/Pain-Free-2021.pdf).

NFL injury-label coverage is itself imperfect: Inclan and colleagues found public
studies captured an average of 66% of ACL injuries in league medical records, with
variation by position and visibility. This is a data-ascertainment result, not an
injury-prediction accuracy figure. Unrecorded injury must remain unknown rather
than be coded healthy. [Original validation study](https://pubmed.ncbi.nlm.nih.gov/34166138/).

**Update schedules constrain historical claims.** Current nflverse documentation
describes nightly stats, dated depth snapshots from 2025, and post-2023 participation
files available only after the postseason. Current injury documentation describes
daily updates; the older review's statement that the source ended after 2024 should
not be carried forward as current fact. Our published gold actually contains 90,752
injury rows spanning 2009–2025. This does not certify their original publication
times or give us complete 2026 injury coverage. Audit captured timestamps and
source revisions before using them in a historical decision-time test.
[Official availability documentation](https://nflreadr.nflverse.com/articles/nflverse_data_schedule.html).

## Recommended next experiment

The [dynamic QB protocol](../research/dynamic_qb_passing_protocol.md) proposes three
separately scored outputs: participation/role probabilities, passing workload and
efficiency conditional on opportunity, and unconditional next-game/remaining-season
yards. Update role and availability promptly when dated evidence arrives; update
execution according to sample size and context. An injury-shortened season should
contribute its actual passing opportunities to execution estimation, not replace
the player's career with a tiny total or an imputed zero rate.

Preserve the original preseason forecast. Evaluate each later revision against
only the games still ahead of its issue time. All earlier NFL history stays
available for training; source-rich modern tests are additional matched comparisons.
No future injury or realized-start filter can silently remove hard cases from the
unconditional score. Oracle workload/health analyses, if used, remain explicitly
retrospective diagnostics.

The next study should compare a strong simple dynamic reference, the existing
booster and a small role/availability/production model on identical candidates,
cutoffs and horizons. It should test adaptation after dated events and deterioration
among stable starters. A better aggregate error alone does not establish that the
role-transition problem is solved.

## Verification

- Frontend production build passed; local Node 21 produced the existing Vite
  supported-version warning.
- Fourteen focused Python tests passed, including equal-season weighting,
  cutoff-defined groups, missing histories, pending outcomes and API scope filtering.
- Dedicated evidence browser smoke passed on desktop/mobile, including probability
  calibration and challenger visibility only in Research archive.
- The broader existing NextGen browser smoke stopped before the Evidence checks
  while looking for the unchanged Lamar Jackson profile link. This review makes no
  claim that the full navigation suite passed.
- All 57 saved QB model/fold fits reproduced exactly; published predictions and
  catalog hashes were not changed.
