# Individual statistics, combinations, and scoring consistency

September 23, 2026. Final research artifact: `individual_stats_20260923_r4`.

The [searchable results and 2026 consistency table](../data/research/individual_stats_20260923_r4/index.html)
contains every inventory entry, individual result, combination comparison, and
839 current draft candidates. Downloads are beside the report. This is a local
research artifact, not a new dashboard release or default forecast.

## The answer about averages and variance

We already have prior-season PPG, weekly standard deviation (`volatility`), a
25th-percentile weekly floor, recency-weighted production, and weekly distribution
and trend descriptors. Player details already display PPG, games, floor and standard
deviation. Career profiles add full-career production and a weighted three-season
scoring rate. These have not disappeared.

The new report adds explicit concentration statistics for the previous season,
previous three seasons, and all available prior career weeks:

| Statistic | Calculation | What it describes |
| --- | --- | --- |
| Mean | Sum of observed weekly points / observations | Average production |
| Median | Middle weekly score | A typical observed week, less affected by outliers |
| Standard deviation and variance | Population SD and its square | Spread of observed scores; at least two observations |
| Coefficient of variation | SD / positive mean | Spread relative to scoring level |
| Best-two-week share | Largest two positive scores / all positive scores | Concentration in two weeks |
| Mean without best two | Remove highest two scores, average the rest | Production outside the two biggest weeks; at least three observations |
| Observation count | Weeks with a scoring record or positive offensive snaps | Evidence behind the estimate |
| Between-season variation | SD of earlier season mean scoring rates | Changes across seasons, including role and age changes |

All actual zero and negative scores remain in means and spread. The concentration
denominator alone uses positive points to avoid nonsensical shares above 100% when
negative scoring reduces the net total. No observation is different from an
observed zero. These are observed-week summaries, not guaranteed active games,
medical risk estimates, or calibrated next-season intervals. Legacy `volatility`
uses sample SD; the explicitly named new distributions use population SD.

For example, ten scores of **40, 40, 4, 4, 4, 4, 4, 4, 4, 4** yield an 11.2 mean,
a 4.0 median, a 71.4% best-two share, and a 4.0 mean without the best two. That
describes exactly the concern about a flattering average driven by two games.

A real example in this gold snapshot is Michael Wilson's 2025 history: 17 observed
weeks, 13.09 mean, 8.40 median, 31.8% of positive points in the best two weeks, and
10.13 mean after removing those weeks. This describes that captured season; it is
not a conclusion that he should be discounted in the draft.

**Predictive result:** the prior-season median, volatility, best-two share and
trimmed mean do not pass the preregistered incremental-information screen for any
position, either above production/usage or above production/usage plus ECR.
Best-two-share MAE gains above the own-data baseline are QB +0.01, RB −0.02,
WR +0.03, TE −0.02 season points. All four 95% intervals include zero. Their
market-informed gains are also approximately zero. Displaying them as evidence is
reasonable; giving them an automatic forecast penalty is not supported here.

## Data and historical scope

The study starts with published gold `canonical_20260923_r5`, pinned to enriched
`canonical_20260923_r4`. It derives a new immutable, research-only gold release,
`canonical_targets_20260923_r3`, without changing `data/current.json`.

The 2003–2008 target feed had more receptions than targets in 20,570 fantasy-position
weekly records. Invalid receiving-target history is made unknown, including its
annual ratios, weekly descriptors, lagged/cross-year summaries, cumulative
opportunity totals and team vacancy derivatives. Team vacancy is masked by the
previous *team* season even for a rookie with no personal prior season. One isolated
2002 offensive-tackle target inconsistency is also masked. Independent play-level
red-zone usage is retained; it does not use the defective target counter.

Every row and outcome is retained. The completed universe is **17,396 player-seasons**:
12,677 returners, 4,115 rookies and 604 market-only candidates. The first 698 rows
train the initial models; **16,698 forecasts cover 2005–2025 in 84 position/year
folds**. Every fold uses **all earlier completed years**. Career features retain
all available earlier NFL weeks back to 2001. Recent-three and prior-season
summaries are additional views, not restrictions on training or career storage.
Earlier careers can be left-truncated; the profile carries that indicator.

Individual tests require three earlier seasons with at least ten finite values
and training variation, so their earliest evaluated year is 2007 and newer feeds
start later. “Full history” does not invent unavailable early tracking, college,
contract or market observations. Market incremental tests begin in 2014 after
three covered training years, use all earlier training rows, and score the paired
ECR-observed test candidates. Population and fixed-era breakdowns accompany the
full-period results. Those breakdowns never shorten training history.

The 2026 consistency table uses completed observations through 2025. No 2026
outcome enters training, scoring or those descriptive summaries.

## What was evaluated individually

The registry contains **1,438 entries**, covering all 97 configured metric-catalog
entries, numeric gold and legacy enriched columns, original profile predictors,
new consistency measures, deterministic reconstructions and named retired formulas.
**1,339 have admissible, varying research inputs**; evaluation still depends on
earlier coverage in each position. The remainder have explicit dispositions for
outcomes, metadata, constants, missing gold evidence, saved fitted/proxy outputs,
or unreconstructed formulas. None of those exclusions means “proven useless.”

There are **22,178 primary comparisons** and 1,061,379 stored fold/population rows.
Nineteen comparisons pass the full screen, all own-data additions. They are
correlated representations, not nineteen independent sources of advantage.

For each eligible statistic the fixed ridge screen tests addition to Basic and
Basic+ECR, then removal from its coherent family combination. Original profile
members also receive removal from the full profile. Baseline members receive a
removal test instead of a fictitious add-one test. A value and its missingness
indicator are always added or removed together.

Basic has 28 predictors: previous observed production/usage/efficiency and exposure,
age, experience, draft-pick information, population and era. The corrected profile
recipe has 218 predictors; adding the six defined ECR transforms gives 224. Six
numeric gold fields enter through the historical profile recipe's prefix rules
beyond the archived 212-feature profile: `career_games`, `career_opportunities`,
`career_pass_attempts`, `college_major_conference`, `injury_data_available`, and
`injury_report_weeks`. This is a current-gold reconstruction, not a claim that the
old raw input matrix or exact feature count was preserved.

The old dated-news ledger is not in gold. Its starter/competition/backup features
remain unknown here, rather than importing an undeclared source. Contract and
combine-derived fields without valid coverage remain unknown too. Fitted V2,
Adaptive and other saved model outputs are registered but are not repackaged as
new raw statistics. Retired WOPR/weighted opportunity are diagnostic reconstructions,
not reinstated production features. ECR aliases are explicitly market inputs;
review-cohort membership is metadata.

The registry supplies definitions, formula/source references, availability years,
aliases and input dispositions. The pinned source snapshot is the executable
definition for complex legacy projection formulas. Some missing-input semantics
are deliberately corrected: unknown age is not a neutral multiplier, partial
career totals are not complete exposure, and missing target history does not
give college production its maximum exposure weight. Legacy projection components
whose position priors touch invalid targets are withheld until their entire
three-season input window is clear. Original legacy three-season formulas retain
that definition even though every new model trains on all earlier seasons.

## Useful information versus familiar information

The clearest individual gains above Basic are predominantly **draft capital
represented appropriately for rookies**, not a large collection of independent
new football discoveries. Representative full-period gains in season-point MAE:

| Added statistic | Position | Gain | 95% season-bootstrap interval |
| --- | --- | ---: | ---: |
| Rookie draft-capital score | QB | +2.16 | +1.67 to +2.69 |
| Rookie draft round | RB | +2.02 | +1.66 to +2.39 |
| Rookie draft-capital score | WR | +1.93 | +1.59 to +2.28 |
| Full-career league points | QB | +1.16 | +0.73 to +1.63 |
| Combine speed score | RB | +1.03 | +0.75 to +1.34 |

These meet the screen in their tested contexts. Draft round, draft pick and
capital transforms largely represent the same information; count them as a family
of alternative representations. Basic already includes draft information, so the
rookie-specific gains also reflect improved functional form/interactions. Combine
coverage and population matter: the result does not establish a universal speed
effect for every veteran or an advantage over market price.

Weighted three-season PPG remains useful context, with smaller conditional gains
above Basic: QB +0.70, RB +0.18, WR +0.33, TE +0.26. These do not meet the fixed
one-point practical threshold. Adding raw age again cannot establish a new age
signal when Basic already contains the same age. The heuristic `age_factor` adds
approximately zero here; that is not a finding that age itself is irrelevant.

**No individual addition passes all gates above Basic+ECR.** For example, prior-year
peak weekly opportunities adds about 1.03 QB points with a positive unadjusted
interval, but does not survive the exhaustive multiple-comparison gate. This
distinguishes a plausible hypothesis from established information the market misses.

## Do combinations hide problems?

Yes, there are reasons to inspect combinations; the evidence does not identify a
single universally harmful stat to delete. No individual removal clears the full
preregistered practical/multiple-testing screen. Several small, contextual removal
gains remain in the complete results rather than being presented as discoveries.

The declared combination controls show model-specific problems:

- Removing raw cumulative career volumes from the **market-informed ridge profile**
  improves QB MAE by 0.89 points [0.34, 1.43] and TE by 0.38 [0.03, 0.73].
  These are unadjusted combination intervals and below the one-point threshold.
- Deduplicating training columns improves some ridge results. Duplicate columns
  change ridge's effective penalty even when they add no independent information.
  Alias-bearing individual comparisons are labeled accordingly.
- Those changes do not yield the same benefits in the fixed histogram booster.
  A linear removal result cannot authorize deleting the feature from every model.
- The large ridge profile still produces unacceptable extrapolations: Michael
  Robinson at RB in 2006 receives 5,568 points versus 37.3 actual; Kendall Hinton
  at QB in 2021 receives 1,301 versus 38.54 actual. The booster forecasts 81.4 and
  26.7 respectively. Cleaning inputs does not solve model extrapolation or sparse
  support for unusual career/position histories. These cases remain in all metrics.

For the fixed booster, full profile versus Basic improves own-data MAE by QB
1.41 [0.69, 2.16], RB 0.66 [0.01, 1.39], WR 1.00 [0.57, 1.44], and TE 0.57
[0.15, 0.99] over 2005–2025. Above Basic+ECR in the covered 2014–2025 universe,
the gains are QB −0.15 [−1.24, 0.97], RB −0.19 [−0.83, 0.43], WR +0.48
[−0.02, 0.91], TE +1.04 [0.06, 2.24]. The TE combination is worth further work;
its unadjusted interval is not a multiplicity-cleared individual-stat result or
proof of draft profit. Full combinations can exploit interactions that the linear
individual screen cannot identify.

The [starter-recognition study](qb_role_transition_2026-09-23.md) remains a separate
task: recognizing an opening job and forecasting production in that job are
different targets. Its prior improvement in job classification did not establish
better full-season workload. It needs a gold contract for its dated evidence and
a rebuilt, target-matched evaluation before being included in this evidence registry
as an approved forecast input. It is not declared ineffective by this point-total audit.

## Interpretation, reproducibility and publication

Positive gain means the tested *change* lowers error: addition helps, or removal
helps, depending on the action column. Effects are conditional on the named
baseline and model. Correlated substitutes can hide individual usefulness. Exact
aliases can change regularization without adding information. Do not sum isolated
gains to predict a combined model's improvement.

The [protocol](../research/individual_stat_protocol.md) fixes ridge alpha 100,
training-only transformations, unchanged booster settings, 10,000 season-bootstrap
resamples, sign-flip tests, Benjamini–Hochberg adjustment across the exhaustive
primary screen, and the one-season-point practical threshold. A candidate also
needs five evaluated seasons, positive interval and leave-one-season-out means,
q≤0.05, and nonnegative squared-error improvement. Bootstrap intervals are not
simultaneous intervals; sign-flip exchangeability, dependent tests, changing eras
and limited seasons constrain interpretation. Subgroup/era tables are descriptive.

This is an exhaustive **linear conditional screen** plus fixed nonlinear
combination checks. It is not exhaustive nonlinear interaction search, a calibrated
injury model, a four-week evaluation, or a draft/waiver utility experiment. No
optimized feature subset is selected on these results and presented as an honest
historical winner. Historical reconstruction uses current captured past records;
complete original publication vintages are not available for every source.

Validation: 54 targeted tests pass. Exact block additions/removals match explicit
sklearn refits on missing/correlated synthetic inputs; eight real fold/recipe checks
match within 1e−7 points. Temporal filtering, missing/zero semantics, full-career
retention, transitive target masking, market-alias separation and existing gold/profile
contracts are covered. All 13 protected live/frozen artifacts remain unchanged.

The final artifact includes feature matrices, registry CSV/JSON/Parquet, every
individual fold/result, population/era results, combination predictions/results,
current descriptive consistency tables, code/protocol snapshots and a hash manifest.
Earlier incomplete research versions are marked superseded; use r4. Benchmarks
were reused only after equality checks on every consumed feature and target.

The existing dashboard still shows its historical floor/volatility fields. New
concentration statistics and this audit are viewable in the linked standalone
report. They are **not yet wired into the dashboard or used to change draft rankings**.

To reproduce into a new, unused directory with the pinned gold release available:

```sh
.venv/bin/python research/individual_stat_data.py data/research/individual_stats_reproduction --gold-version canonical_targets_20260923_r3
.venv/bin/python research/individual_stat_audit.py data/research/individual_stats_reproduction
.venv/bin/python research/individual_stat_report.py data/research/individual_stats_reproduction
```

Use the pinned source snapshot and recorded dependency versions when checking exact
reproduction. The finished report's feature matrix was also independently rebuilt
from corrected gold and checked value-for-value, including its model specifications.
