# Deep feature and ensemble results

Completed September 24, 2026 (America/New_York). The expanded search produced a
stronger model library, particularly when combined with our existing forecasts.
More elaborate ensemble selection usually did **not** beat a simple blend of the
three models that performed best on earlier seasons. These are retrospective
results, not a production promotion or proof of an absolute optimum.

## What completed

- 12,480 annual pipeline fits: 11,232 fixed-library fits and 1,248 adaptive fits.
- 152 new candidate configurations, plus persistence and compatible saved models.
  The preseason library contains 184 candidates; combined in-season libraries
  contain 164–200. These are correlated forecasts, not independent experiments.
- 2,929 candidate inputs; 69,584 examples representing 17,396 candidate
  player-seasons and 4,770 players. NFL observations begin in 2001.
- QB/RB/WR/TE; preseason, next week, next four weeks and remaining season.
  In-season forecasts originate after week 2. Primary evaluation: 2019–2025.
- LightGBM, XGBoost, CatBoost, histogram boosting, random/extra trees, SVR,
  kernels, neural networks, PLS, hurdle and auxiliary-target models; learned
  feature selection, interactions and tree representations.
- Equal blends, inverse-error weights, greedy ensemble selection, convex weights,
  linear/nonlinear stacking, cohort weights and calibrated rank combinations.
  Four chronological policies optimize MSE, MAE, NDCG@24 or top-K capture.

Every forecast uses earlier seasons for fitting and selection. Ensemble weights
learn from earlier out-of-year predictions; the choice of combination method uses
another three earlier evaluation years. A larger library is not fitted against
the year whose results are being reported.

## Comparison with our published predictions

The following reports the **MSE-selected policy using both new and saved models**,
not the hindsight-best candidate. Positive reductions mean lower squared error.
Top-K capture measures how much of the best achievable actual points the selected
top players deliver (QB10/RB20/WR30/TE10); changes are percentage points.

| Position | Next-four MSE reduction | Remaining MSE reduction | Next-four capture change | Remaining capture change |
|---|---:|---:|---:|---:|
| QB | 2.70% | 15.37% | +2.17 | +1.30 |
| RB | 7.30% | 17.91% | +5.03 | +6.29 |
| WR | 13.29% | 23.13% | +1.83 | +1.91 |
| TE | 9.28% | 25.53% | +5.90 | +9.55 |

This policy improves average MSE in all eight tasks. Remaining-season MSE improves
in all seven test years for RB/WR/TE, and six of seven for QB. The seven-year
descriptive bootstrap interval for QB next-four MSE includes no improvement;
the other seven intervals favor improvement. These intervals are unadjusted for
the many research comparisons and do not establish prospective superiority.

**MSE improvement does not mean every prediction metric improves.** This policy
improves MAE in only three of eight tasks: remaining-season RB/WR/TE. Its MAE rises
by 0.024–1.106 points in the other five. NDCG@24 improves in seven of eight tasks;
WR next-four changes by −0.0023. The separately evaluated MAE policy improves MAE
and MSE in seven of eight tasks, with QB next-four the exception. Selecting the
objective matters; a top-K-selected policy does not universally win at top-K.

The benchmark is the actual published release
`nextgen_qb_variations_20260924_r2`: approved QB next-four policy, reference forecasts
elsewhere. Both compared horizons match 5,938 of 6,006 published candidate-season
rows (98.9%); the lab retains its preseason candidate universe. Identity, origin,
horizon end and canonical outcomes agree on every compared row. No equivalent
published preseason or next-week forecast was available for this comparison.

Nine historical published labels omitted points recorded under another weekly
position; four are in 2019–2025. Comparisons re-score the unchanged saved forecasts
against canonical totals and preserve the original labels. Unknown 2026 outcomes
remain unknown. Production files were not corrected or overwritten.

## What added value

**Keep useful existing models.** Adding saved models improves the MSE policy in
all eight matched tasks versus the new library alone, with additional MSE reductions
of 1.30–8.42%. Existing profile ridge and profile boosting models are frequent
ensemble members. Linear models are therefore not universally useless: their
value depends on the inputs, task and complementary errors.

**The simple ensemble remains a strong control.** Within the primary expanded
library, the adaptive MSE selector beats the earlier-selected top-three average
in only 4/16 tasks. The combined-library top-three blend itself beats published
MSE in all eight tasks, with reductions of 5.97–27.29%, though it also has MAE
tradeoffs. More elaborate selection is not a justified universal replacement.

**Broader inputs help, with diminishing gains.** Holding the first LightGBM
recipes' structural settings fixed at 120 trees:

| Input expansion | Tasks with lower MSE | Median task MSE reduction |
|---|---:|---:|
| Summary history → all own-data inputs | 15/16 | 3.12% |
| Raw history → all own-data inputs | 14/16 | 2.36% |
| Basic plus raw → all own-data inputs | 11/16 | 0.78% |
| All own-data → market-augmented | 11/16 | 1.27% |

These are task medians, not pooled reductions or significance claims. The named
`inventory` and `all` views resolve to identical columns and are not counted as
independent input expansions. Frequent discovered inputs include recent usage and
fantasy production, target share, opportunities, receptions, variability and gaps
in observed history. Discovery recurrence does not prove that an individual input
or interaction causes an improvement; representation recipes also vary settings.

**Tree count has a practical stopping point.** Moving from 40 to 120 trees usually
helps. Moving from 120 to 360 worsens MSE in 317/384 LightGBM recipe/task comparisons,
101/128 XGBoost comparisons, and 63/64 histogram-booster comparisons. CatBoost
improves in 49/64, with a median 1.19% reduction. These correlated comparisons are
diagnostics, not independent trials.

The adaptive search allowed 4,096 trees with 80-round patience, selected a count on
the latest earlier season, then refitted all earlier history. Selected counts:

| Engine | Minimum | Median | Maximum |
|---|---:|---:|---:|
| LightGBM | 46 | 141 | 1,335 |
| XGBoost | 36 | 148 | 787 |
| CatBoost | 52 | 320.5 | 1,706 |

No selected count approached the ceiling, including its final 80 rounds. These
recipes do not appear limited by that ceiling; this does not certify an optimum
over every learning rate, architecture or feature set.

**Library size also has diminishing returns.** Increasing the defined 64-member
budget to the full primary library improves convex-blend MSE in 12/16 tasks, by
0.25–2.34%; the other four worsen by 0.27–1.12%. These are fixed nested subsets,
not an exhaustive search over all subsets. Median prior-error correlation between
candidates is about 0.93. For MSE-selected methods with explicit weighted-average
metadata, median effective membership is four in the primary library and three
in the combined library. This describes weight concentration, not the number of
independent information sources.

## Reliability and boundaries

Boosted trees supply the hindsight-best new fixed candidate in all sixteen tasks;
that descriptive result must not be relabeled a validated model-selection policy.
The search does not establish that neural networks or kernels cannot work. There
are 33 neural optimization-limit warnings among the modern fixed fits, so complete
neural convergence is not claimed.

Raw forecasts and failure diagnostics are retained. For example, the main MSE
policy predicts −84.53 season points for Tim Tebow's 2021 tight-end season, an
implausible extrapolation. The combined MSE policy's minimum is −8.27 next-four
points for that player. Empirical coverage of nominal 80% intervals varies roughly
75–85% across policies/tasks; intervals use earlier errors and are unavailable in
the first policy year. These issues need attention before any serving integration.

The evidence reuses historical years and retrospective provider/identity vintages.
Some inputs are missing or derived from the same observations. We lack a complete
dated live-news/starting-job ledger and tracking coordinates. In-season inventory
context remains preseason-dated, while raw observation history updates. This
campaign does not test every weekly origin, every optimizer setting or every
possible feature. It makes no new 2026 forecast and promotes no model.

All 56 focused tests pass. Completed run manifests verify sources, dependencies,
input hashes and checkpoint integrity. Both final dashboards pass all 24 task-view
browser checks, comparison filters and sorting, with no JavaScript errors. The
published catalog checksum is unchanged.

## Artifacts

- [Primary-library dashboard](runs/deep_report_001/index.html)
- [Combined-library dashboard and published comparisons](runs/deep_report_001/published_library/index.html)
- [Full diagnostics](runs/deep_diagnostics_001/report.md)
- [Experiment design and reproduction](deep/README.md)
- [Fixed fits manifest](runs/deep_search_003/manifest.json)
- [Adaptive fits manifest](runs/deep_capacity_002/manifest.json)
- [Ensemble analysis manifest](runs/deep_report_001/manifest.json)
- [Delivery verification](runs/deep_delivery_001/manifest.json)
