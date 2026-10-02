# Initial research readout — September 23, 2026

**Keep this as a component research lab. The best next candidate is a separately evaluated
uncertainty wrapper; these results do not justify replacing NextGen's point models.**
Pooling, passing-play participation and QB mixtures did not show a clear incremental
MSE gain against their corresponding market-aware controls. That is evidence about
these particular implementations, not a rejection of the underlying literature.

The completed run is [literature_002](runs/literature_002/report.md), with a
[searchable browser table](runs/literature_002/index.html),
[all paired results](runs/literature_002/summaries.json),
[interval diagnostics](runs/literature_002/intervals.json), and
[provenance manifest](runs/literature_002/manifest.json).

## What was actually tested

Eleven position/outcome tasks across 2007–2025 produced **209 expanding-window folds,
312,346 model predictions, and 40,413 distinct player-season-target cases**. Those cases
represent 15,212 distinct player-seasons. The fixed menu tested explicit rates, learned
exposure-aware rate pooling, a receiving participation block, coarse QB role-duration
mixtures, and chronological prediction intervals. There are 704 point-comparison slices
and 1,408 interval diagnostic slices; these counts are not independent experiments.

Every imported NextGen boosting forecast matched its corresponding saved preseason
forecast from `nextgen_system_20260923_r5` exactly: **40,413 matches, maximum difference
zero**. All ten focused tests pass; lint passes; the local HTML filter was exercised
in a headless browser without JavaScript errors. The successful run's protected dependency
hashes were unchanged. These are preseason experiments, not a test of the newer
rest-of-season rankings being developed elsewhere in the workspace.

MSE is primary because the models fit conditional means with squared loss. MAE is also
reported. Gains below use equal weighting of seasons; positive means lower error.
Historical years have been researched repeatedly. Bootstrap intervals are nominal,
unadjusted for multiplicity, and do not make this a confirmatory study.

## Point-model findings

| Candidate / outcome | MSE gain without ECR | MSE gain with ECR | Interpretation |
| --- | ---: | ---: | --- |
| Pooled rates, WR league points | +0.41% | −0.13% | Small own-data result does not survive the market control |
| Pooled rates, RB league points | −0.21% | +0.71% | Mixed direction; uncertainty includes no gain |
| QB role mixture, pass attempts | +0.49% | +0.56% | No clear improvement over the matched direct model |
| QB role mixture, carries | +2.22% | +1.25% | Interesting direction, still inconclusive |
| QB role mixture, league points | +0.76% | +1.15% | Inconclusive; modern and market-covered slices remain uncertain |
| Participation, WR league points | approximately 0.00% | −0.20% | No useful gain in this block |
| Participation, TE league points | −0.14% | −0.29% | No useful gain in this block |

Pooling and participation compare with the same raw-rate control; role mixtures compare
with direct prediction using identical inputs, including prior role duration. None of
the market-aware pooling, participation or mixture comparisons has an all-history
nominal 95% MSE-gain interval entirely above zero. An uncertain small gain is not a
demonstration of equivalence or proof that no player subgroup benefits.

There are two useful qualifications:

- **Simple rate features merit a cleaner follow-up.** Before pooling, adding explicit
  rates and exposures to the imported own-data booster reduced MSE by 1.47% for QB
  attempts and 0.42% for WR receiving yards, with positive nominal intervals. These are
  modest, exploratory gains. A separate raw-rate ablation against a booster-plus-ECR
  control is still needed; this run's market control already includes the raw-rate block.
- **Market information remains a substantial control.** Adding ECR to the raw-rate
  control reduced MSE by about 9.5–17.9% across the ten volume/points tasks, while RB
  yards/carry changed by only about 0.07%. These are matched forecasting comparisons,
  not evidence of acquisition value or an edge over executable prices.

The receiving result does not improve materially in the covered population. For WR
league points, the participation-covered slice's market-aware MSE gain is −0.73%; for
TE points it is −0.90%. Both remain uncertain. These features measure passing-play
participation and means of weekly target-earning ratios, not verified routes or openness.

The coarse role mixture also does not resolve the motivating QB example by itself.
For Lamar Jackson's 2019 season, its market-aware version forecasts roughly 243 attempts
and 62 carries versus observed totals of 401 and 176. This is an illustrative hindsight
case, not a selection criterion. It supports keeping dated job evidence, time in role,
and conditional rushing workload as separate research questions. A three-state usage
proxy should not be presented as a validated opening-starter model.

There were 32 fold/family pooling fallbacks because fewer than 20 historical rates were
available, all involving receiving rates. There were no numerical optimizer failures.
The run retains these folds; they are not quietly dropped to improve the comparison.

## The uncertainty layer is promising, with a visible limitation

For the market control's season-points forecasts, scaled residual intervals improve
the proper interval score compared with constant-width residual intervals on identical
calibrated rows. Both use only earlier out-of-fold errors, never same-season outcomes.

| Position | Nominal coverage | Observed coverage | Mean interval width, points | Interval-score improvement |
| --- | ---: | ---: | ---: | ---: |
| QB | 80% | 82.4% | 148.2 | 5.4% |
| RB | 80% | 81.2% | 103.5 | 11.7% |
| WR | 80% | 82.1% | 110.0 | 5.9% |
| TE | 80% | 81.3% | 71.7 | 10.4% |

These are equal-season summaries after calibration warmup, not player-specific coverage
guarantees. Widths are substantial. Scaling also worsens the interval score for some
other targets, notably WR and TE target counts, so one scaling rule is not universally
better.

**Aggregate coverage hides differences between player groups.** On the scaled 80%
season-points intervals, returning-player coverage is about 79.8% QB, 79.5% RB, 79.0%
WR and 78.2% TE. Rookie coverage is approximately 87.6%, 84.7%, 89.2% and 89.5%,
respectively. Modern returning-QB coverage is only about 77.4%. The next experiment
should test population- and opportunity-conditioned calibration with earlier-only
fallback rules, rather than treating good aggregate coverage as readiness for serving.

An earlier-residual median adjustment also lowers season-points MAE by approximately
1.22 QB, 1.87 RB, 1.56 WR and 0.95 TE points on the same calibrated rows. That supports
keeping mean and median outputs distinct. It does not justify replacing expected points
with a median in a decision objective that needs an expectation.

## What to pursue next

1. **Independent uncertainty wrapper:** compare population/volume-conditioned intervals,
   conditional quantile models and simple chronological controls; evaluate coverage,
   interval score, width, and drift by task. This is the most concrete next component,
   still requiring validation before use in the application.
2. **Small rate-feature ablation:** test the modest QB-attempt and WR-yard findings against
   a matched market-aware model without those explicit rates, then the strongest relevant
   archived challenger. Do not bundle pooling into that experiment.
3. **Better role evidence:** audit dated opening-job evidence and fit conditional workload
   with explicit duration and designed/scramble usage. Keep the current direct model and
   this coarse mixture as controls. The literature's opportunity hypothesis remains open.
4. **Keep the remaining tracks separate:** team allocation, college development, availability,
   decision utility and tracking each have their own prerequisites in the
   [research queue](research_queue.md). A useful descriptive strength measure need not
   become a fantasy-points feature, and decision utility needs a separate test.

Do not add the current pooling or participation blocks to serving based on this run.
The useful immediate additions are the isolated test harness, explicit controls,
calibration diagnostics, and a more focused next experiment menu.

## Isolation and audit history

All additions from this work are inside `experiments/player_signals_lab/`. The runner
imports the existing library read-only, creates unique run directories, blocks Python
writes outside its run, and has no publishing or model-registration path. Gold data
and imported source modules are pinned by hashes. Native IO is not an OS sandbox;
our native Parquet writer receives only a lab-owned path and input hashes are checked.

`literature_001` remains marked failed: its original broad audit caught concurrent
changes to `data/current.json`, `ranking_routes.py`, `catalog.py` and the data-side
`nextgen.py`, as other work published analysis `nextgen_rankings_20260923_r6`. None of
those modules or that mutable catalog selects this experiment's explicitly pinned
gold or preseason model. The audit was corrected to distinguish experiment dependencies
from monitored workspace activity, and the entire run was repeated as `literature_002`.
All 312,346 prediction records, including intervals, are identical between the two runs.
No statistical recipe or evaluation choice was tuned during this correction.

The successful run has `status: complete`, `protected_inputs_unchanged: true`, no
concurrent monitored changes, and verified hashes for every recorded output. Use that
run for interpretation. Removing the lab directory removes its code, reports, tests,
temporary files and results without a registry edit or change to the original models.
