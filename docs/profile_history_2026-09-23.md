# Profile-informed forecasts across the historical record

> Historical result requiring revalidation: the source matrix predates the target
> and market corrections. See the [corrected-gold individual-stat audit](individual_stats_2026-09-23.md)
> for the current profile reconstruction, individual additions/removals, and limitations.

**Keeping older training data helped the primary model. Full profiles also
improved average point forecasts, especially for players with a short prior
season but substantial earlier careers. Neither result solved the original
market-miss problem.** All 13 previously identified modern small-prior-season
misses remain outside the predicted top 60.

The experiment is complete and research-only. The
[design](../research/profile_history_design.md) was saved before fitting;
[code](../research/profile_history.py),
[temporal tests](../research/test_profile_history.py),
[complete results](../data/research/profile_history_20260923_r1/report.json),
[coverage](../data/research/profile_history_20260923_r1/coverage.csv), and
[feature dictionary](../data/research/profile_history_20260923_r1/feature_dictionary.json)
are retained with immutable input hashes and implementation snapshots. No
production forecast, profile pointer, college pointer or earlier research
artifact was changed.

**The broad pass uses every completed accepted candidate, rather than requiring
complete modern enrichment.** There are 17,450 player-seasons from 2004–2025:
12,677 returners, 4,115 rookies and 658 market-only candidates across QB, RB, WR
and TE. Zero-production outcomes remain included. The 698 candidates from 2004
seed training; 16,752 forecasts are evaluated in 84 position/year folds across
2005–2025. The modern subset contains 5,955 forecasts across 2019–2025. The
2005–2018 evaluation contains 10,797.

Every primary fold trains on **all earlier accepted candidate years**, without
a five-year or modern-only training restriction. The 2025 fits use 2004–2024:
2,185 QB, 4,833 RB, 6,308 WR and 3,310 TE training rows. Training examples have
equal row weights. Each player's features retain all available earlier NFL
history alongside recent summaries; older observations are not discarded when
the model receives a recent-three-season feature.

NFL profile observations start in 2001, college observations in 2004, and the
accepted preseason candidate cohorts in 2004. Earlier NFL observations therefore
inform the earliest profiles, but there is no fabricated 2001–2003 preseason
cohort selected from players who subsequently appeared. The first evaluable
forecast is 2005 because 2004 supplies the first accepted training outcomes.
The incomplete 2026 outcomes are not graded.

**The data join uses the newer accepted backfill.** The main candidate and dated
context source is `historical_backfill_20260923_r2`. Full NFL profiles and college
history come from `player_profiles_v1_20260923_r2` and
`college_nfl_v1_20260922_r5`, which pin the earlier repaired r5 history.
Before joining, the runner verifies identical candidate IDs, positions,
populations, outcomes and benchmark market values across historical versions.
Existing non-null cutoffs must agree. The newer backfill supplies restored names,
additional dated context and the previously missing rookie/market-only cutoffs.
Both versions of saved V2 remain separate benchmarks.

Missing enrichment never removes a candidate. Counts below describe available
evidence, not a certification that every historical observation is complete.

| Evidence present | All 2004–2025 candidates | Modern 2019–2025 candidates |
| --- | ---: | ---: |
| Candidate rows | 17,450 | 5,955 |
| Any prior NFL observations | 13,098 | 4,514 |
| Any prior snap observations | 7,436 | 4,504 |
| Accepted college identity link | 12,261 | 5,404 |
| At least one usable complete college season | 10,602 | 4,911 |
| Prior injury-report observations | 8,117 | 3,847 |
| Dated roster state | 6,746 | 3,011 |
| Known absence fact | 90 | 36 |
| Dated QB job evidence | 44 | 24 |
| Positional ECR | 8,284 | 4,519 |
| Overall ECR | 5,624 | 3,202 |

No injury report means no captured report, not confirmed health. Missing snaps
are unknown participation. College quality gaps preserve earlier usable college
seasons; incomplete or ambiguous seasons do not become zero production. Current
club, eventual career endpoints, later forecasts and future success never enter
the feature matrices. Historical source revisions and retrospective identity
links remain limitations of the reconstruction.

**The feature comparison separates longer history from the wider profile.**

| Specification | Inputs | Feature columns |
| --- | --- | ---: |
| Basic | Prior-season production, workload and efficiency; age, experience, era, season length, population and draft capital | 28 |
| Career | Basic plus all prior NFL exposure, career/recent rates, participation-specific samples and rates, prior peak, gaps and observed team changes | 148 |
| Profile | Career plus college production/shares/quality, reported injury history, dated roster/absence context and available QB job evidence | 212 |
| Basic + ECR | Basic plus positional/overall rank transforms and dispersion | 34 |
| Profile + ECR | Entire profile and the same market inputs | 218 |

College and NFL production retain different denominators. College information
enters both directly and through fixed interactions whose weight declines with
NFL attempts/carries/targets. No individual college, injury or context family is
isolated by the bundled profile comparison; its gain cannot be attributed to one
of those sources alone. The dated job ledger is still limited to the previously
defined QB review cohort. Snap participation bands do not establish RB job
assignments or TE receiving routes.

Two learners were fixed before fitting: ridge regression with alpha 100, and
histogram gradient boosting with 120 iterations, learning rate .05, 15 leaves,
minimum leaf size 30, L2 penalty 10, 63 bins and no early stopping. The booster
was declared the primary nonlinear challenger before outcomes were inspected.
All five specifications run with both learners. There is no parameter search or
post-result choice of model family.

Imputation and scaling are trained on earlier rows. For trees, columns with no
varying observed training values cannot supply a numeric split; varying
missingness is retained as an indicator. This training-only transform also
handles the installed binning implementation's failure on constant/all-missing
early-history columns. It does not drop players or older training years.

The target is season league points divided by nominal season length, converted
back to season points at prediction time. This accommodates the 16/17-game
change without treating a 17-game season as greater per-game ability. Negative
forecasts are clipped to zero. An explicitly known full-season absence forces
zero in every new model. Partial absence facts are features, not another games
multiplier or an invented hard points ceiling.

**Full profiles improve generic point accuracy modestly across the wider
history.** The following results use the fixed booster and identical rows
within each column. MAE is mean absolute error in season league points, giving
each test season equal weight. Lower is better.

| Inputs | All history, 2005–2025 | Earlier, 2005–2018 | Modern, 2019–2025 |
| --- | ---: | ---: | ---: |
| Basic | 34.89 | 35.59 | 33.49 |
| Career | 34.42 | 35.18 | 32.91 |
| Profile | **34.13** | **35.03** | **32.32** |
| Basic + ECR | 32.40 | 33.82 | 29.55 |
| Profile + ECR | **32.20** | **33.74** | **29.12** |

Profile versus basic improves all-history MAE by **0.77**, season-bootstrap 95%
interval **[0.38, 1.19]**, with positive differences in 17 of 21 seasons. Modern
improvement is **1.17**, interval **[0.34, 2.03]**, positive in six of seven
seasons. Equal-season RMSE also falls: 53.50 to 52.59 over the full history and
51.68 to 50.47 in modern years.

The addition beyond ECR is smaller: **0.20** all-history MAE improvement,
interval **[-0.10, 0.49]**, and **0.43** modern improvement, interval
**[-0.06, 0.98]**. These intervals include zero. Pre-2011 folds have no ECR;
their market-feature models cannot learn a market effect. The 2011 fold also
has no earlier ECR observations for learning that relationship. Actual market
selection comparisons therefore begin with the available 2011–2025 market pool.

Modern position-level results:

| Position | Forecasts | Basic | Profile | Basic + ECR | Profile + ECR |
| --- | ---: | ---: | ---: | ---: | ---: |
| QB | 794 | 49.16 | 47.57 | 40.97 | 40.96 |
| RB | 1,571 | 35.54 | 34.03 | 31.52 | 30.72 |
| WR | 2,345 | 32.55 | 31.53 | 29.12 | 28.85 |
| TE | 1,245 | 22.99 | 22.16 | 20.86 | 20.29 |

The generic QB/RB/WR profile improvements have positive interval lower bounds;
TE's interval includes zero. Each position's incremental profile gain beyond ECR
has an interval spanning zero. These are related exploratory comparisons with
no correction for testing multiple positions, cohorts and models.

Saved V2 does not forecast the entire new candidate population, so its error
must be compared on matching rows. Among **4,244 modern returners**, original
V2 MAE is 37.44, backfilled V2 is 37.21, the generic profile booster is 36.65,
and profile + ECR is 32.84. The latter improves over backfilled V2 by 4.36,
interval [3.36, 5.34]. Its matched advantage over basic + ECR among these
returners is much smaller: 0.53, interval [0.07, 0.96].

**Restricting training to five years made the primary model worse.** This
comparison changes only the training-example window. Both versions retain the
same full career, college and other historical feature vectors for each player.

| Primary model, modern evaluation | Five training years | All earlier training years | Error reduction from older training rows |
| --- | ---: | ---: | ---: |
| Profile | 33.54 | **32.32** | 1.22 [0.63, 1.70] |
| Profile + ECR | 29.72 | **29.12** | 0.60 [0.19, 1.08] |

Both improve in six of seven seasons. Leaving out any one season retains a
positive pooled improvement. Without ECR, keeping older training rows lowers
modern error by 2.79 for QB, 1.54 for RB, 0.72 for WR and 0.84 for TE; each
position's interval is positive. With ECR, all four point estimates remain
positive, but only WR's position-specific interval excludes zero. This supports
retaining the full training history for this model; it does not establish that
every model, era or feature benefits equally from older data.

**The useful small-sample signal is stronger for established players with a
short prior season.** These cohorts are defined before outcomes and include
failures, not just the 13 known misses. All rows below are modern returners.

| Cohort | Rows | Basic + ECR MAE | Profile + ECR MAE | Paired improvement, 95% interval |
| --- | ---: | ---: | ---: | --- |
| Fewer than 10 prior-season box-score games | 2,011 | 21.63 | 20.92 | 0.71 [0.18, 1.28] |
| Short prior season, at least 10 career observed weeks | 1,421 | 22.04 | 21.18 | 0.86 [0.43, 1.32] |
| Fewer than 10 career observed weeks, no known left truncation | 590 | 20.87 | 20.48 | 0.39 [-0.48, 1.29] |

The profile advantage for players with genuinely little NFL evidence remains
uncertain. The older all-history career-under-ten cohort likewise has essentially
no generic gain and a slightly worse point estimate after adding profiles to
the market model. Rookie results also remain uncertain: modern generic rookie
MAE moves from 23.29 to 23.05, while adding profiles to the rookie market model
moves 21.40 to 21.47. Those findings do not validate college information as a
reliable replacement for sparse NFL evidence.

**The individual role-change failures remain visible.** These are descriptive
cases after fitting, not the sample used to train or select the models. Values
are season league points.

| Forecast | Saved original V2 | Basic + ECR | Profile + ECR | Realized |
| --- | ---: | ---: | ---: | ---: |
| Lamar Jackson, 2019 | 130.09 | 173.09 | 152.30 | 428.68 |
| Darren Waller, 2019 | 20.33 | 41.89 | 42.99 | 221.00 |
| Jordan Love, 2023 | 15.94 | 194.56 | 187.29 | 319.06 |
| Dak Prescott, 2025 | 146.04 | 210.93 | 232.43 | 323.78 |

Dak's longer career helps this forecast, while adding profiles worsens the
market-conditioned Lamar and Love forecasts. A complete historical profile
does not automatically turn sparse backup production into the correct expected
workload for a newly assigned job. Generic profile accuracy and role-transition
accuracy are distinct requirements.

The linear comparator also exposes an extrapolation failure. In the five-year
ridge fit, Tim Tebow's 2021 TE row receives approximately 41,055 season points;
his historical QB production is an unusual input for that TE regression. The
all-history ridge predicts zero on this row; the primary profile booster predicts
79.77, against zero realized. Kendall Hinton also exposes position-change
extrapolation. These outputs are retained and disclosed, not removed to improve
the comparison. The primary booster's predictions range from zero to 332.11
without ECR and zero to 345.05 with ECR. Its all-history advantage is therefore
not the result of that ridge outlier. Neither linear profile model is suitable
for promotion without further work on these failures.

**No tested model establishes an acquisition edge over ECR.** The original V2
comparison pool is preserved separately from the expanded pool with rookies and
market-only players. The original modern baseline reconciles exactly to 87
calls, 24 hits, zero of 13 small-prior-season misses recovered, and two of 29 RB
misses recovered. Every model now directly forecasts all 13 named cases; all
still recover zero. Their predicted points and selection ranks are saved in
`original_small_sample_misses.parquet`. Antonio Brown 2021 ranks 61st under
profile + ECR, but the declared top-60 threshold is unchanged.

Original V2 pool, modern years:

| Ranking | Model-only hits / calls | Market-only hits | Net value / season |
| --- | ---: | ---: | ---: |
| Saved original V2 | 24 / 87 | 36 | -12.64 |
| Generic basic booster | 21 / 101 | 51 | -18.24 |
| Generic profile booster | 26 / 93 | 46 | -15.39 |
| Basic + ECR booster | 15 / 55 | 27 | -5.19 |
| Profile + ECR booster | 15 / 57 | 25 | -7.98 |

Profile + ECR is worse than basic + ECR on this modern selection metric despite
its lower average point error. Over the broader 2011–2025 original pool, their
net values are -10.41 and -7.81 respectively: an improvement that still trails
ECR. The generic profile model recovers one of 38 small-prior-season misses in
that longer window; the market-conditioned profile model recovers zero of 38.

In the separate all-candidate modern pool, basic + ECR returns -5.86 per season
and profile + ECR -6.02. The expanded pool has 12 small-prior-season returner
misses because adding candidates changes the realized top-60 boundary; this does
not replace or revise the original 13-case definition. The acquisition target
remains positive positional VOR weighted by actual games / 17, not raw points,
an executable draft price or roster-constrained profit. No late ADP is backdated.

**The research direction is to retain full history while improving how profiles
inform future opportunity.** This pass supports using the broader historical
training population and career context for established players. It supplies
weaker evidence for genuinely sparse careers and no solution for the named
market misses. More historical data is useful here, but the mapping from a
player's past roles to his expected future workload still needs development.
The profile bundle's individual sources and the position-transition failures
need separate tests before interpreting the results as a validated role model.

All eras have previously informed research. The comparisons are retrospective,
source coverage varies, the first folds have little training history, and
historical records may have been revised later. Season-block bootstrap intervals
do not eliminate dependence from repeated players and overlapping histories,
and no interval corrects for the multiple comparisons. A prospective frozen
evaluation is still required before production promotion.

Reproduce into a new output directory:

```sh
.venv/bin/python research/profile_history.py \
  --history historical_backfill_20260923_r2 \
  --profiles player_profiles_v1_20260923_r2 \
  --college college_nfl_v1_20260922_r5 \
  --output data/research/profile_history_reproduction
.venv/bin/pytest research/test_profile_history.py research/test_qb_role_transition.py \
  research/test_preseason_role_workload.py research/test_research_audits.py \
  tests/test_player_profiles.py tests/test_college_pathways.py
.venv/bin/ruff check research/profile_history.py research/test_profile_history.py
```

Validation: 54 targeted tests pass and Ruff is clean. Tests cover old-history
retention, future-data isolation, missing profiles and rookies, source-quality
gaps, unknown early feeds, duplicate rejection, dated context, train-only
transforms, all-earlier-year training, the five-year control, and absence handling.
Integration checks verify cohort preservation, source/version invariants, the
original market reconciliation and unchanged protected artifacts. The final
[validation record](../data/research/profile_history_20260923_r1/validation.json)
also checks every frozen implementation snapshot. Concurrent workspace edits
changed `src/patron/data/releases.py` after the run; the saved implementation
records the version actually used, and the experiment's input and output hashes
still match.
