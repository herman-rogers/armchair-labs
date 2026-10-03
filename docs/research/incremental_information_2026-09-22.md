# Research focus: predict what the strong baseline misses

> Dated research record. Findings and roster advice apply to the stated data and cutoff; follow-up ideas are not an active implementation commitment.

The next research question should be **whether an added signal improves a strong,
decision-appropriate baseline on unseen outcomes**. Predicting fantasy points from
more inputs is insufficient when usage, draft capital or consensus already captures
their useful content.

The [reproducible audit](../../research/incremental_information.py) reanalyzes the accepted
RB role, player outlook and college forecasts. Its
[saved results](../../data/research/baseline_incremental_20260922_r1/report.json) include
paired season differences, selection overlap, forecast-adjustment diagnostics, all
existing market comparisons, and the original college identity sensitivity results.
It does not fit another model, grade 2026, or change a published forecast.

The first proposed experiment has now been run:
[RB residual-correction results, 2026-09-23](rb_residual_correction_2026-09-23.md).
It does not establish reliable incremental value over usage. Neither this audit
nor that experiment has a dashboard view yet; their research verification is complete.

## What the evidence supports

The evidence supports **limited demonstrated incremental value**. It does not prove
that the underlying data contains no new information: a weak model, noisy outcomes,
limited coverage, or inadequate sample size can also obscure a useful signal.
Prediction correlation alone would not resolve that distinction.

### RB roles: usage captures most of the apparent improvement

2019–2025, three selections per decision window, next four calendar weeks:

| Comparator | Role gain per selection | Season-bootstrap 95% interval | Selections changed |
| --- | ---: | ---: | ---: |
| Trailing points | +3.930 | +1.687 to +6.339 | 118 / 231 |
| Recent snaps | +1.116 | -0.687 to +2.751 | 76 / 231 |
| Usage ridge | +0.652 | -1.331 to +2.533 | 43 / 231 |

The usage model itself gains 3.277 points over trailing points. Consequently,
**83.4% of the role model's apparent gain over trailing points is already achieved
by usage**. This is arithmetic about these selection policies, not an estimate of
the percentage of features that are redundant or a causal attribution.

The role model retains 188/231 usage selections (81.4%). Its remaining gain is
positive in four of seven seasons. There is insufficient evidence that its
disagreements are consistently useful.

For the small-prior-sample cohort, the role model retains 199/207 usage selections
(96.1%). The eight replacements yield -0.200 points per selection, interval
[-0.421, -0.021], with no positive season. This particular role extension has not
earned a larger role for sparse-history players. These are repeated selections,
not counts of unique players or independent trials.

### Outlook: RB/WR adjustments do not improve typical error

Week 2, next-four-week point totals, 2019–2025. Positive improvement means lower
error. MAEs pool player rows; improvement and its interval give each season equal
weight, so their arithmetic need not match exactly.

| Position | Forecasts | Usage MAE | Outlook MAE | Equal-season MAE improvement | 95% interval |
| --- | ---: | ---: | ---: | ---: | ---: |
| QB | 309 | 18.225 | 17.846 | +0.385 | +0.069 to +0.643 |
| RB | 770 | 12.959 | 12.992 | -0.034 | -0.085 to +0.017 |
| WR | 1,187 | 12.648 | 12.726 | -0.078 | -0.174 to +0.017 |
| TE | 695 | 8.488 | 8.434 | +0.055 | +0.021 to +0.086 |

These comparisons cover 2,961 of 4,846 Week 2 candidate rows. The other 1,885
lack a complete finite forecast/outcome comparison and are excluded consistently
from both baselines. These results do not establish performance for unscored players.

Dropping any one season leaves RB/WR's average MAE improvement negative. Their
result is not explained by a single bad season. QB/TE have small positive MAE
signals, so an assertion that *all* additions are useless would overstate the data.
Those position-wide findings do not override the original subgroup/range gates.

The audit also asks whether adjustments address baseline errors. Let `e` be actual
points minus baseline points, and `d` be challenger points minus baseline points:

```text
baseline MSE − challenger MSE = mean(2 × e × d) − mean(d²)
```

The first term measures alignment with baseline errors; the second measures the
squared size of the adjustment. Both use equal-season weighting in the report.
This identity is diagnostic, not an independent test or causal feature decomposition.

For Week 2 WRs, alignment is 3.482 squared points and adjustment cost is 3.284,
leaving only +0.198 MSE improvement, interval [-3.471, +4.351]. None of the four
positions has a positive lower endpoint for its MSE-improvement interval. The
small QB/TE MAE signals do not establish an improvement in large misses.

### College: distinguish rookie points from participation and later development

2018+ entry classes, completed rookie-year outcomes through 2025:

| Position | Players | Draft MAE | College + draft MAE | Equal-season MAE improvement | 95% interval |
| --- | ---: | ---: | ---: | ---: | ---: |
| QB | 74 | 45.632 | 48.160 | -5.152 | -10.047 to -0.803 |
| RB | 217 | 31.722 | 31.858 | -0.307 | -1.583 to +0.875 |
| WR | 329 | 24.116 | 23.360 | +0.682 | -0.085 to +1.464 |
| TE | 150 | 18.142 | 17.269 | +0.931 | -0.162 to +1.900 |

The WR/TE rookie-point gains remain uncertain. Draft capital already summarizes
professional evaluations; that is a reason to demand this comparison, not proof
that it fully captures every useful college signal.

There are narrower exploratory findings: WR/TE rookie-games MAE improves by
0.198/0.210 games on an equal-season basis, with positive differences in all eight
seasons. TE first-three-year points improve by 3.364 MAE points across six entry
classes (91 players), interval [+0.325, +5.656], while equal-season MSE worsens.
Keep these as target-specific research leads. Games are not an injury probability,
and three-year points are not immediate lineup value. Overlapping three-year
outcomes also weaken an interpretation of entry classes as independent blocks.

### Consensus remains a required decision benchmark

[FantasyPros' current methodology](https://support.fantasypros.com/hc/en-us/articles/115001219327-What-is-ECR-Expert-Consensus-Rankings-and-how-do-you-calculate-it)
describes combining experts' rank points and selecting NFL experts using accuracy
and recency criteria. ECR is an aggregate forecasting product. Its current
description does not establish identical methodology in every historical season.

ECR supplies ranks, not league-specific expected points. Do not subtract a rank
from actual points or treat rank correlation as evidence of value. Use matched
selection/ranking comparisons, or learn a rank-to-outcome mapping using earlier
seasons only when a point forecast is necessary.

The accepted modern returner/top-60 audit gives standalone model net values of
-12.64, -11.81 and -11.37 against ECR. An 80% ECR/20% next-generation blend gives
+1.56, interval [-0.55, +3.12]. These units are summed positive VOR weighted by
games/17; they cannot be compared numerically with the point-error tables above.
All 288 existing variants remain in the saved audit. Picking the best blend after
viewing those variants is not independent evidence of an edge.

## What to test next

Use a task-specific baseline contract before adding a feature family:

| Task | Required comparator | What the addition must predict |
| --- | --- | --- |
| Preseason acquisitions | Cutoff-matched ECR and ADP; test a market-conditioned candidate | Errors in market-relative selections at an available price |
| In-season four-week points | Recent usage; include available market context in a separately aligned comparison | Production changes beyond the role level already visible in usage |
| Rookie preseason points | Draft capital; also market once available at the cutoff | Rookie outcomes beyond the draft/market evaluation |
| Rookie in-season points | Draft capital + recent usage | Remaining college information after observed NFL usage |

**First experiment: a restrained correction to the usage baseline.** Reuse the
existing RB role variables—snap change, its observation count and prior sample—
to forecast the usage model's remaining error. The hypothesis is that measured
role changes anticipate sustained work before a recent-usage average catches up.
The present role model refits the full point target; it does not directly test this
residual-learning design. Try this before acquiring additional context feeds.

Use the existing decision weeks, candidate eligibility and four-week target so
that changing the sample cannot manufacture a win. Retain both original cohorts
and report them separately. Sample-dependent shrinkage toward zero correction
is an explicit challenger, with its strength chosen only within earlier seasons.
The usage model's zero correction is the fallback.

**Second experiment: college as a diminishing prior for sparse NFL samples.**
Start from draft capital plus recent NFL usage, then test whether a college-derived
prior corrects its errors. Any weighting by observed NFL sample must be learned
inside historical training folds. Comparing this integrated model only with
recent points or a position mean would repeat the weak-baseline problem. Treat
the WR/TE participation findings as a separate target; do not multiply an
absence-inclusive points forecast by another participation discount.

Both experiments should use the same evaluation discipline:

1. Generate earlier-season **out-of-fold baseline predictions**. Define residual
   training targets from those predictions, not a baseline fitted on the same
   outcomes. Enforce full outcome maturation before each later decision date.
2. Fit correction features, imputation, penalties and shrinkage on earlier data
   only. Score the unchanged baseline and baseline-plus-correction on identical
   held-out players and windows. A baseline-only recalibration control separates
   simple bias correction from information supplied by additional features.
3. Declare the primary target, cohort, baseline, feature family and minimum useful
   improvement before evaluating an untouched period. Record every attempted
   variant. Use historical results here as development evidence; 2019–2025 is
   already inspected and cannot become a fresh confirmation set by renaming it.
4. Report paired seasonal gains, MAE and squared-error diagnostics, changed
   selections, coverage, and uncertainty. A positive point estimate alone does not
   justify replacing the baseline. Confirm practical benefit in a prospectively
   frozen evaluation before changing defaults.
5. Freeze forecast inputs, consensus, executable price and availability together
   for a future decision-value study. Until these snapshots exist, describe
   selection results as historical proxies rather than obtainable waiver gains.

Timestamped injury/roster/teammate events may eventually supply useful information,
but they need the same residual test and source-availability controls. Broadening
data collection is lower priority than establishing that corrections work.

## Reproduction and limits

```sh
.venv/bin/python research/incremental_information.py \
  --history data/research/historical_v2_20260922_r5 \
  --outlook data/research/player_outlook_v1_20260922_r5 \
  --college data/research/college_nfl_v1_20260922_r5 \
  --version YOUR_NEW_INCREMENTAL_VERSION
.venv/bin/pytest research/test_incremental_information.py research/test_research_audits.py
```

The runner requires matching accepted history and artifact hashes, uses a common
finite sample across each point-model baseline ladder, checks observation keys
and recorded training chronology, and refuses to overwrite an output version.
It reports excluded nonfinite observations and the 683 pending college target
rows in the 2018+ window. A player can contribute multiple target rows.

The new intervals use 10,000 seeded resamples of equal-weight season differences;
small endpoint differences from older outlook/college reports reflect their
different bootstrap draws. Season blocking handles overlapping weeks within a
season, but does not eliminate dependence from repeated players across years or
overlapping college outcome horizons. No interval adjusts for the many historical
experiments. The audit verifies consistency with accepted artifacts, not original
publication-time availability of every historical source. It does not establish
statistical equivalence, feature irrelevance, or a market-beating edge.

Validation: 20 targeted audit tests pass; Ruff checks and formatting pass. The
saved audit reconciles 70 role/outlook metrics and all 12 college sample/MAE
comparisons with the original reports; all 36 recorded input hashes match.
