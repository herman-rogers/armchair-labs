# Protocol 2: close the feasible research queue

Frozen before follow-up efficacy results. This is a bounded extension of the literature
review, not a claim to exhaust every possible model or acquire unavailable proprietary
data. Each previously raised path must end with an executed test, or an evidenced data
boundary plus the narrower test that is feasible. No changes outside this lab.

Use the same corrected gold, source pins, 2004+ training cohort, 2007–2025 expanding
folds and fixed boosting parameters as Protocol 1. Reuse verified out-of-fold predictions
from `literature_002`; never replace them with fitted training predictions. Preserve all
variants, missing data, zero outcomes, and unfavorable results. New feature priors and
calibration groups use training/earlier OOF rows only. Historical evaluation is exploratory.

## Predeclared paths

1. **Population calibration and drift.** For all eleven targets and both raw-rate and
   market controls, compare Protocol 1 global scaled intervals with population-specific,
   predicted-volume-quartile, population×volume, partially pooled population, signed
   asymmetric residual, three-year recent, and annual adaptive-radius calibrators.
   Five earlier seasons, three distinct seasons minimum; global minimum 100 rows,
   group minimum 60 rows. Sparse groups fall back to population then global. Quartile
   boundaries use earlier predictions only. Partial pooling gives the group `n/(n+100)`
   mass and the global distribution the rest. Adaptive alpha updates once per completed
   season at rate .05 and clips to [.02,.40]; no within-test-season feedback.
2. **Conditional quantiles.** For each position's league points, fit fixed .05/.10/.50/
   .90/.95 quantile boosters using the market-control inputs. Compare uncalibrated bounds,
   earlier-OOF CQR expansion and population CQR expansion to the same residual baseline.
   Sort quantiles to prevent crossing; report crossing frequency before sorting. Expansion
   uses nonnegative outside-interval residuals, so cannot shrink or reverse bounds.
3. **Raw-rate ablation and stronger controls.** On the original eleven tasks compare
   booster+ECR with booster+ECR+raw rates. For league points also fit the archived 224-feature
   profile/market design with the same learner and compare proposed additions against it.
   Its old gold vintage is audited against current labels/identities before use. Repaired
   market inputs come from current gold; raw archived ECR values cannot reintroduce the
   earlier cutoff defect.
4. **Strength/context.** Compare one-year vs three-year pooled rates, role-stratified
   priors (low/high earlier workload, minimum 40 historical rates), and team-relative
   efficiency features. Group priors use only earlier training rates. Team-relative rates
   are observational comparisons; they do not identify causal player talent or control
   for linemen/coaching. Audit and test NGS rates with their own published exposure counts;
   distinguish numerical reconciliation from verification of every eligible tracking play.
5. **Receiving execution.** Add compatible historical catch fraction, YAC per reception,
   air yards per target, and same-provider NGS metrics separately from participation.
   Retain unknowns. True all-player route counts/openness remain a separate data audit.
6. **QB role and duration.** Recover actual starting-QB labels from the pinned raw schedule.
   Compare direct totals with opening-start mixtures, an evidence-feature ablation using
   the dated local ledger, and duration-aware features (prior starts, late starts,
   designed/scramble carries). Unknown news is not evidence of a backup role. Use strict
   publication/event dates before the player's cutoff. Also test weekly starting-role
   transition probabilities on non-overlapping four-week blocks, training only on earlier
   seasons; no current-year future outcomes. Neither model is a medical forecast.
7. **Joint opportunity/execution distributions.** For WR/TE receiving yards fit appearance,
   positive-target workload and conditional yards/target using earlier seasons. Compare
   product of means, independent residual simulation and paired residual simulation
   against direct receiving-yard prediction. Paired residuals come only from earlier
   positive-opportunity OOF cases. Zero-opportunity probability remains explicit. Report
   MSE, MAE, CRPS and 80/90% interval scores; this is an approximate joint model.
8. **Team volume/allocation.** Fit team attempts/carries and compare prior-season and
   expanding regressions. Reconcile independent preseason QB attempts and RB carries to
   forecast team budgets, retaining a residual allocation for other/unmatched players.
   Estimate the residual fraction from earlier *cutoff-assigned* cohorts, not the held-out
   team's realized roster. Compare partial shrinkage and full reconciliation. Report
   traded-player/missing-team boundaries and how often budgets bind. No oracle team totals.
9. **College development.** Run pre-draft-style versus post-draft feature ablations on
   the existing rookie candidate cohort, and add multiple prior college seasons, trends,
   production shares and transfer indicators. The rookie cohort is already NFL-selected:
   neither version establishes NFL entry probability. Audit full college-season rows,
   graduation/eligibility and identity negatives before any full-exit-cohort claim.
10. **Availability ascertainment.** Compare scoring appearances with offensive-snap
    appearances by position/year; audit injury record coverage and timestamps. Predict
    next-season offensive-snap appearances on years with source coverage using history
    alone versus prior injury-report descriptors. Explicitly label participation, not
    injury risk. Do not turn an unmatched roster/player identity into an injury zero.
11. **Decision utility.** Audit real historical prices, rosters and feasible alternatives.
    If executable-price history is absent, run only a transparent fixed-slot ranking
    policy sensitivity (mean vs median vs lower bound), using matched position pools and
    realized points. This is a counterfactual selection diagnostic without costs, not
    a draft/trade/waiver backtest or evidence of profit.
12. **Tracking and broader attribution.** Inventory actual trajectory/all-route/non-skill
    coverage. Do not label summary NGS tables as trajectories or box-score residuals as
    causal attribution. Execute available descriptive context comparisons; explicitly
    close unsupported trajectory and defensive/line-scoring paths as data-limited.

## Scoring and stopping

Point means: MSE primary, MAE secondary. Probabilities: Brier and log loss. Distributions:
proper interval score at 80/90%, width, coverage, weighted interval score where a median
is available, and CRPS for simulations. Compare on identical rows, with equal-season
weighting and paired 95% season-bootstrap intervals. Report population, modern (2017+),
low/high predicted-volume, and COVID-era (2020) diagnostics; these are not independent tests.

For each executed stage apply a one-sided paired-season sign-randomization test and Holm
adjustment across its declared all-population primary comparisons. The test assumes
exchangeable signs/symmetric null season effects; repeated players and era shifts still
limit interpretation. Population/drift tables remain descriptive. No significance-based
selection, automatic combination, or production promotion. Empty/ineligible comparisons
must remain recorded. Validate temporal and write boundaries with focused tests.

Freeze prospective evaluation for the 2027 preseason separately: capture the final
candidate list, cutoffs, data vintages, configuration and predictions before the first
regular-season game; evaluate all 2027 labels only after season completion. Nothing in
the reused 2007–2025 history is described as an untouched holdout. A future capture is a
specified requirement, not a scheduled service or a claim that the experiment has run.
