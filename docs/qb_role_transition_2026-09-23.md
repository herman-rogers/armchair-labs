# QB job probabilities and conditional workload

**The experiment improves overall QB forecasts, but does not establish an
incremental role-model advantage over the market.** Most of the improvement
comes from adding preseason ECR. Dated starter evidence improves job recognition;
the conditional workload model still fails badly on Lamar Jackson's 2019 case.
Nothing from this experiment is promoted to production.

The [design](../research/qb_role_transition_design.md) was saved before fitting.
The [runner](../research/qb_role_transition.py),
[evidence ledger](../research/qb_role_evidence_20260923.csv),
[tests](../research/test_qb_role_transition.py), and
[complete results](../data/research/qb_role_transition_20260923_r3/report.json)
are research artifacts. The final artifact directory contains the exact code,
design, evidence, inputs' hashes, predictions, and review cohort used. The r1–r3
iterations have identical numerical predictions; later revisions add ADP timing
audit/provenance and formatting, without changing evidence or model choices.

**The cohort is selected using preseason information.** Returning QBs qualify
for evidence review when their saved positional ECR is at most 32 and they had
fewer than ten prior games with at least 50% offensive snaps. This includes
promotions, injury returners, unresolved competitions, displaced starters and
utility players classified as QBs. It is not a pure promotion cohort.

There are 67 review player-seasons: 24 in the initial 2014–2018 training period
and 43 across the expanding 2019–2025 test folds. Every qualifying case remains
in evaluation. The broader evaluation includes all 556 eligible held-out QB
returner forecasts. Of 918 feature rows, 917 have eligible prior snap/identity
coverage. Rookies remain outside this model.

The manual ledger has 47 records. Two announcements are after their forecast
cutoffs and are excluded; two records corroborate the same Lamar case. There
are 44 review cases with admissible evidence and 23 explicitly unknown. In the
held-out review cohort, the breakdown is **16 starters, eight competitions and
19 unknowns**. Evidence collection is incomplete and uneven across years; it is
not an exhaustive historical depth-chart database. Unknown cases are listed
in the report and remain in every applicable comparison.

The ledger records publication, event when established, and capture dates,
source URL, job classification, competitors and a short interpretation.
Classification uses the latest admissible evidence. Same-date conflicts resolve
to unknown. Probable/expected choices stay in the competition category; that
conservative rule loses information about which competitor is favored.
Competitor names are descriptive metadata, not model inputs.

Lamar's starting designation and Flacco's subsequent departure provide a dated
carry-forward inference, while Love's team report explicitly describes his new
responsibility. These cases have evidence before their respective cutoffs.
[Baltimore's designation](https://www.baltimoreravens.com/news/joe-flacco-is-disappointed-but-handles-backup-role-like-a-pro),
[Baltimore's departure announcement](https://www.baltimoreravens.com/news/press-release-ravens-player-announcements),
[Green Bay's Love report](https://www.packers.com/news/matt-lafleur-on-jordan-love-this-offseason-he-s-the-guy-in-charge).

These are publication dates on currently accessible pages, captured September
23, 2026. They do not establish that the historical text was identical. The
primary analysis admits the entire cutoff day because the saved forecast has
date precision; a second run excludes all same-day evidence. For example, the
2019 cutoff is July 17 and the 2020 cutoff is August 13. Familiar later starter
announcements cannot be backdated into these forecasts.

**The model separates the opening job from season workload.** The job outcome
is whether the player starts his team's first scheduled regular-season game,
using the frozen schedule's starting-QB IDs. Using each team's first game
handles postponed openers. These future IDs are labels only, never predictors.
Opening starter status is distinct from subsequent job retention and availability.

Four fixed logistic specifications predict opening-start probability: prior
usage/role summaries and shrunk efficiency; those inputs plus dated evidence;
those inputs plus ECR; and all inputs together. ECR enters as log rank and inverse
rank. Evidence enters as starter, competition and backup indicators; unknown is
the reference state. Imputation, missingness indicators, scaling and coefficients
are fitted using earlier seasons only. Logistic C is 1, without class weighting.

Two fixed ridge-10 experts, one trained on opening starters and one on
nonstarters, separately forecast full-season points, passing attempts, carries
and offensive appearances. All four probability models share these same experts,
so their mixture comparisons isolate the effect of the probability estimates.
Four corresponding direct regressions receive the same respective inputs and
test whether the scenario architecture adds value.

Efficiency inputs use the prior three seasons' base passing points per attempt
and rushing points per carry. Fixed shrinkage of 200 attempts and 60 carries
pulls small samples toward exposure-weighted QB training priors. Overlapping
three-year histories contribute repeatedly to those training priors. Passing
and rushing scoring use the frozen league rules; bonuses and receiving points
remain in the final league-points target, not these efficiency summaries.
Efficiency has not been independently validated by a separate ablation here.

Within each job, the point regression learns conditional season scoring using
past workload and efficiency together. The final mixture is
`P(start) × E(points | start) + P(nonstart) × E(points | nonstart)`.
It does not multiply separate marginal workload and efficiency predictions, and
it does not apply a second games or availability multiplier. Offensive appearances
are a separate diagnostic, not an injury forecast.

**Point accuracy improves primarily with ECR.** The table reports mean absolute
error in season league points, giving each test season equal weight. Lower is
better; all comparisons within a column use identical player rows.

| Model | All QBs, 556 | Review cohort, 43 |
| --- | ---: | ---: |
| Saved repaired V2 | 57.62 | 114.65 |
| Previous role-summary pilot | 60.78 | 120.28 |
| Direct usage + efficiency | 59.70 | 117.15 |
| Direct + dated evidence | 57.61 | 104.09 |
| Direct + ECR | 50.28 | **93.81** |
| Direct + ECR + evidence | 50.17 | 94.02 |
| Mixture, usage gate | 58.62 | 121.62 |
| Mixture, evidence gate | 56.81 | 110.84 |
| Mixture, ECR gate | 49.46 | 99.83 |
| Mixture, ECR + evidence gate | **49.40** | 97.57 |

The full mixture improves all-QB MAE over V2 by 8.22 points, with a season-bootstrap
95% interval of [6.02, 10.31]. But the matched incremental comparison is much
smaller: adding evidence to the market mixture improves MAE by only **0.06**,
interval **[-0.28, 0.44]**. In the review cohort that improvement is 2.26,
interval [-2.55, 7.07]. The direct market model has lower review-cohort error than
either market mixture. Its improvement over V2 there is 20.83, interval [3.13,
36.00], without dated news or a scenario decomposition.

The strict earlier-day sensitivity weakens the evidence result further:
all-QB market-mixture MAE is 49.46 without news and 49.56 with news. Review-cohort
MAE is 99.83 versus 99.19. Same-day evidence is therefore material to interpreting
the already small observed advantage.

**Recognizing the opening job improves more clearly than forecasting workload.**
Brier scores below pool player rows; lower is better. These are descriptive
calibration results, not independent confirmation.

| Probability inputs | All-QB Brier | Review-cohort Brier |
| --- | ---: | ---: |
| Prior usage + efficiency | 0.1050 | 0.4256 |
| Plus dated evidence | 0.0868 | 0.2505 |
| Plus ECR | 0.0423 | 0.1021 |
| Plus ECR + evidence | 0.0420 | 0.0876 |

Forty of 43 review QBs actually start the opener. The full gate assigns this
group only 72.8% on average, so it still underestimates opening jobs in this
selected group. On the 16 source-identified starters, adding evidence to ECR
improves Brier from 0.1028 to 0.0299. It worsens Brier on unknown review cases
from 0.0588 to 0.0842. Incomplete source coverage changes the learned meaning
of the unknown reference group and limits interpretation of the overall result.

For season passing attempts, all-QB MAE is 90.48 for the market mixture and 90.55
with evidence. Review-cohort errors are 165.92 and 162.66; the direct market
regression is better at 153.98. Thus better job recognition has not established
better conditional workload estimation. Carries, offensive appearances, log
loss and reliability-bin counts are retained in the complete report.

**Lamar remains an explicit failure.** No model parameters were changed after
seeing these results.

| Lamar 2019 forecast | Opening-start probability | Attempts | Season points |
| --- | ---: | ---: | ---: |
| Saved V2 | — | — | 130.09 |
| Mixture, usage gate | 30.0% | 126.55 | 105.88 |
| Mixture, evidence gate | 91.8% | 36.30 | 111.02 |
| Mixture, ECR gate | 66.8% | 72.77 | 108.94 |
| Mixture, ECR + evidence gate | 88.7% | 40.74 | 110.77 |
| Direct ECR + evidence | — | 143.18 | 176.44 |
| Realized | Started opener | 401 | 428.68 |

The starter expert predicts only 24.28 attempts for Lamar, while the nonstarter
expert predicts 170.30. Increasing the correct job probability therefore makes
his workload estimate worse. The point experts predict 111.71 and 103.39
respectively, so changing their mixture cannot yield a credible full-time
starter projection. The 2019 training fold has 149 opening starters, only 33 with
fewer than ten prior high-snap games. The fitted regressions still map this
unusual low-passing-volume profile poorly, even after selecting a starter role.
This is a model failure, not evidence that his intended job was uncertain.

The opposite error also matters. Foles 2019, Winston 2022 and Lance 2022 all
start their openers but finish with fewer than eight high-snap games. The full
mixture predicts 163.93, 177.76 and 193.72 season points; realized totals are
37.74, 43.92 and 12.46. This is an outcome-only diagnostic, without assigning
causes or using those later absences as preseason facts. Opening-job accuracy
does not validate expected season participation or job retention.

**The acquisition test still shows no market edge.** Only covered QB forecasts
are replaced in the original all-player pool. Every other player retains V2.
The positional replacement rule, ECR top-60 comparator and realized value target
are unchanged. Even the prior-pilot rows below replace QBs only, so their ranking
results need not match the earlier multi-position experiment.

| Ranking | Model-only hits / calls | Market-only hits | Net value / season |
| --- | ---: | ---: | ---: |
| Saved V2 | 24 / 87 | 36 | -12.64 |
| Direct ECR | 27 / 81 | 35 | -8.70 |
| Direct ECR + evidence | 27 / 81 | 37 | -9.92 |
| Mixture ECR | 27 / 89 | 39 | -11.65 |
| Mixture ECR + evidence | 27 / 89 | 39 | -11.65 |

Every tested model still recovers **zero of the 13 small-prior-sample market
misses** and two of 29 RB misses. The full mixture recovers ten of 32 QB misses,
versus seven for V2, but its overall acquisition value remains below ECR.
Net value is the repaired audit's positive positional VOR weighted by actual
games / 17; it is not season points or roster-constrained profit. No RB/TE role
model was changed or evaluated here.

The ADP audit finds zero certified admissible windows. Every dated FFC window
ends after its forecast cutoff; the MFL aggregates do not certify an end by
that cutoff. ADP is omitted rather than backdated. The report records every
window and its exclusion reason.

**The next workload design needs to make a starting job change the conditional
opportunity estimate itself.** A reasonable next test is a hierarchical estimate
of attempts and carries per game in the relevant role, combined with a separate
model of time in that role, preserving their dependence. New starters need
shrinkage toward comparable starter workloads; low historical season volume
must not mechanically imply backup-level workload inside the starter scenario.
Before attributing incremental value to news, complete the fixed evidence cohort
and distinguish favored competitors from unresolved choices. These are proposed
tests, not fixes established by this run.

The present conclusions remain limited by reused test years, a manual partial
ledger, historical sources without contemporaneous text snapshots, small role
subgroups, and fixed unvalidated architecture/shrinkage choices. Bootstrap
intervals resample only seven season clusters and do not adjust for multiple
comparisons. A prospective shadow period is still required before promotion.

Reproduce into a new directory:

```sh
.venv/bin/python research/qb_role_transition.py \
  --history data/research/historical_v2_20260922_r5 \
  --pilot data/research/preseason_role_workload_20260922_r1 \
  --output data/research/qb_role_transition_reproduction
.venv/bin/pytest -q research/test_qb_role_transition.py \
  research/test_preseason_role_workload.py research/test_research_audits.py \
  tests/test_outlook.py
.venv/bin/ruff check research/qb_role_transition.py research/test_qb_role_transition.py
```

Validation: 37 targeted tests pass. They cover publication/event/capture dates,
same-day sensitivity, conflicts and unknowns, delayed team openers, ADP timing,
future-data perturbation, efficiency shrinkage, expanding-fold separation and
probability-weighted scoring without a second participation multiplier. Ruff
passes. Saved V2's market results reconcile exactly, protected production hashes
are unchanged, and the final predictions match both earlier numerical runs.
