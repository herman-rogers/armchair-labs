# Consolidating our Next-gen evidence

> Dated research record. Findings and roster advice apply to the stated data and cutoff; follow-up ideas are not an active implementation commitment.

**The best-supported foundation is draft capital for rookies and production
history for returners.** Participation and dated availability add useful context,
but most improvements shrink substantially after accounting for ECR. WR
participation history is the clearest current candidate for additional information
beyond a market-informed baseline. This is a shortlist for the next model, not a
new production forecast or a proven acquisition advantage.

The review also found a remaining integrity defect: **2003–2008 target counts are
not usable as recorded**, despite the accepted consolidated release. The final
experiment masks those inputs and their affected derivatives, retaining the full
history. See the [data-quality finding](enriched_target_quality_2026-09-23.md).

## What was actually examined

The final research release is
[`nextgen_signals_20260923_r3`](../../data/research/nextgen_signals_20260923_r3/report.json).
It pins published gold **`canonical_20260923_r5`**, derived from enriched
**`canonical_20260923_r4`**. All new observations come through those verified
tables. No raw-cache fitting, legacy model-prediction inputs, or unfinished 2026
outcomes enter this experiment.

- Coverage inventory: **17,396 completed 2004–2025 player-seasons**, comprising
  12,677 returners, 4,115 rookies and 604 market-only candidates. The canonical
  release separately quarantines 54 candidates admitted only by late evidence.
- New forecasts: **14,608 player-seasons**, 10,995 returners and 3,613 rookies,
  in **152 position/population/year folds spanning 2007–2025**. Every fold trains
  on all available earlier candidate years beginning in 2004. The first three
  years seed training; they are not silently discarded. Market-only candidates
  are inventoried but not assigned an own-stat forecast here.
- Player-history observations begin in **2001**. Completed-career and recent
  three-year profile summaries are rebuilt from the enriched observations at
  each forecast cutoff. They are not copied from a different profile vintage.
- Each stat family is added separately to the same usage baseline, then to the
  same usage-plus-ECR baseline. Calibrated prior points is an additional guardrail.
  Position and rookie/returner populations are fitted separately; missing features
  retain the candidate. All-family fits are declared diagnostics, not selected winners.
- The [protocol](../../research/nextgen_signal_protocol.md) specifies fixed ridge
  regression, earlier-training-only imputation/scaling, chronological evaluation,
  coverage admission, fixed eras, and equal-season errors. No hyperparameter or
  blend search was performed.

Positive improvement below means **lower absolute error in a player's full-season
league points**, not extra points scored by our team. Intervals are 10,000-resample
season-bootstrap 95% intervals, unadjusted for the many comparisons. The saved
report includes all 1,120 position, population, era, subgroup and baseline summaries;
they are overlapping diagnostics, not independent experiments. These historical
years have already informed research.

## Our most useful information

| Observation family | Held-out evidence | Practical interpretation |
| --- | --- | --- |
| Rookie draft capital and drafted/undrafted status | 2007–2025; improves over the earlier-position mean in all 19 seasons at every position | Strongest broad-history rookie foundation; it represents information already public to the market |
| Completed-career plus recent-three-year production/exposure | 2007–2025; improves QB/RB/WR above both prior points and annual usage | Keep a player's history and sample size; a single previous-season average loses useful information |
| QB weekly scoring/usage shape | 2007–2025; +3.24 points lower MAE versus usage, interval [2.26, 4.27] | Past within-season changes help QB forecasting; this is not proof of future starting status |
| WR age/experience | 2007–2025; +1.14 [0.71, 1.58] versus usage, positive mean in all three fixed eras | Useful modest career-stage context; incremental gain after ECR remains uncertain |
| TE age/experience | 2007–2025; +0.54 [0.28, 0.79] versus usage, positive mean in all three eras | More stable than the unqualified TE career-history extension in this model |
| QB/WR participation level, changes and historical participation bands | 2017–2025 held-out support; +4.84 QB and +1.26 WR versus usage | Useful role observations, with only nine evaluable years; not a 22-year tracking finding |
| Dated availability/roster context and prior reports | RB/WR 2007–2025, TE 2008–2025; modest own-model improvements | Keep mechanism-specific evidence and unknown states; confirmed full-season absence is a hard constraint shared by every model |

Rookie draft-only errors and improvements over the position-mean baseline:

| Position | Forecasts | Mean baseline MAE | Draft MAE | Improvement [95% interval] |
| --- | ---: | ---: | ---: | --- |
| QB | 361 | 50.51 | 33.47 | 17.04 [12.98, 21.18] |
| RB | 972 | 40.54 | 27.81 | 12.72 [11.08, 14.45] |
| WR | 1,550 | 33.72 | 21.15 | 12.57 [10.87, 14.29] |
| TE | 730 | 20.56 | 13.76 | 6.81 [5.56, 8.16] |

Increment from the returner history family over annual usage, 2007–2025:

| Position | Forecasts | MAE improvement [95% interval] | Positive years | Early / middle / modern improvement |
| --- | ---: | --- | ---: | --- |
| QB | 1,457 | 3.00 [2.19, 3.86] | 19 / 19 | 3.21 / 3.30 / 2.56 |
| RB | 3,219 | 0.76 [0.37, 1.17] | 15 / 19 | 0.99 / 0.25 / 1.00 |
| WR | 4,100 | 1.56 [0.85, 2.17] | 18 / 19 | 1.27 / 1.69 / 1.71 |
| TE | 2,219 | 0.40 [-0.72, 1.30] | 14 / 19 | 0.87 / 0.48 / -0.07 |

The eras are 2007–2012, 2013–2018 and 2019–2025. QB/RB/WR history also improves
over the calibrated points baseline by 3.05, 0.99 and 1.67 points respectively,
with positive interval lower endpoints. The TE extension fails that guardrail
and worsens mean squared error. Taysom Hill's position-switch history causes a
large extrapolation failure that remains in evaluation. A career profile is
valuable information, but an unrestricted linear mapping of its totals can fail.

These are **family-level** findings: the experiment does not establish that every
member is useful or assign causal importance to individual fields. For example,
it cannot yet separate the benefit of a three-year scoring rate from its exposure
count or distinguish snap trend from historical high-participation scoring.

## What ECR already captures, and our remaining candidates

Adding cutoff-safe ECR to usage produces the largest broad returner improvements
on market-observed candidates. With three earlier market-covered training years,
the comparison spans 2014–2025:

| Position | Forecasts | Usage MAE | Usage + ECR MAE | Improvement [95% interval] |
| --- | ---: | ---: | ---: | --- |
| QB | 718 | 73.01 | 59.21 | 13.80 [11.06, 16.83] |
| RB | 1,496 | 49.60 | 44.73 | 4.87 [4.16, 5.68] |
| WR | 2,015 | 48.38 | 44.47 | 3.91 [2.40, 5.21] |
| TE | 1,025 | 34.82 | 32.50 | 2.32 [1.66, 3.06] |

These errors concern a different, market-covered population from the full-history
table. Do not compare their absolute MAEs across tables. The model learns a mapping
from ECR to league points; this is not a direct test against ECR's unmodified ranking.

The history family's incremental gains after ECR become approximately **-0.01 QB,
+0.02 RB, -0.03 WR and -0.71 TE**, all with intervals spanning zero. That is strong
evidence that useful own-model information need not be information the market misses.

| Candidate beyond usage + ECR | Forecasts / held-out years | Incremental MAE improvement [95% interval] | Assessment |
| --- | --- | --- | --- |
| WR participation family | 1,621 / 2017–2025 | **1.07 [0.35, 1.69]** | Strongest follow-up candidate; MSE also improves, including in 2019–2025, and deleting any one season leaves positive mean MAE gain |
| RB availability/roster context | 1,496 / 2014–2025 | **0.56 [0.14, 0.95]** | Modest candidate; modern-only MAE interval and full-history MSE interval still span zero |
| WR route-proxy family | 1,135 / 2020–2025 | **0.53 [0.03, 0.89]** | Secondary candidate; only six years, proxy definitions, and multiple comparisons weaken the claim |

QB route proxies also show a small positive market-conditioned result. Do not
translate that mechanically into “QB routes predict value”: the family can encode
utility-player roles or feed/missingness distinctions. Its football meaning and
population effects require inspection. It is not a primary consolidation choice.

The union of all families adds **-0.24 QB, +0.30 RB, +0.13 WR and -0.54 TE** over
the market-informed baseline, with every interval spanning zero. Simply combining
all available data is not supported by this test. Nor do these MAE gains establish
profitable draft disagreements, waiver gains, or independent discoveries after
multiple testing. That still requires matched ranking/decision evaluation and
prospective evidence.

## What to retain without promoting

- **College production and shares:** useful profile/identity information, but no
  reliable broad rookie-point improvement over draft capital in this new test.
  Incremental gains are -2.99 QB, +0.01 RB, -0.15 WR and +0.10 TE; only QB's
  worsening excludes zero. Full eligible candidates remain included, including
  missing college histories. This differs from the older college-linked-only
  study's sample and feature design.
- **NFL tracking statistics:** not interchangeable with our project's “Next-gen”
  name. Rushing tracking gives +0.34 RB MAE improvement over usage in 2022–2025,
  but -0.10 after ECR with uncertainty spanning zero. Four test years cannot
  justify a full-history claim. Other tracking gains remain uncertain.
- **Efficiency, red-zone placement, team environment and athletic measurements:**
  scattered position-specific improvements, no broad independent market gain.
  More granular ablations may identify useful subsets; this screen cannot declare
  every underlying statistic useless.
- **Starter recognition:** a useful separate problem, still not a validated
  season-workload engine. Existing QB evidence is summarized below. Historical
  high-snap bands are observations, not a substitute for a dated starter label.
- **Hard absence facts:** required constraints even if their average statistical
  increment is small. Missing evidence must not imply healthy or available.

## Historical coverage is part of each statistic's definition

Counts are across all 17,396 completed candidates, not only players who have that
source. Numeric observations must be finite; observed-state counts require true.

| Input | Observed rows | Available forecast years / limitation |
| --- | ---: | --- |
| Prior PPG and games | 12,677 | 2004–2025 returners |
| Recent-three-year profile scoring rate | 13,049 | 2004–2025; can exist for some candidates without immediate-prior-year production |
| Prior targets per game after quarantine | 9,347 | 2010–2025 in this forecast panel; 2004–2009 target inputs are unknown |
| Offensive snap share | 7,082 | 2014–2025; first family test 2017 |
| Route participation proxy | 5,279 | 2017–2025; first family test 2020 |
| Tracking separation | 1,131 | 2017–2025; partial receiver coverage |
| Tracking rushing yards over expected per attempt | 354 | 2019–2025; first family test 2022 |
| Observed cutoff roster state | 6,729 | Uneven 2004–2025 evidence; inferred prior teams are not observed current states |
| Reviewed available-games cap | 90 | 2007–2025; narrow reviewed coverage |
| Positional / overall ECR | 7,303 / 5,624 | 2011–2025, varying population coverage |
| Quality-gated latest college receiving production | 2,880 | Rookie candidates, 2005–2025; legitimate recorded zeros count as observations |
| Contract APY / complete-team vacated target share | 0 / 0 | Not usable modeling inputs in this release |
| Dated depth-chart rank | 545 | Only 2025 among completed outcomes; no longitudinal validation |

Source coverage is not equivalent to predictive validation. Profiles preserve
more information than these lean families use. College seasons begin in 2004,
NFL weeks in 2001, and individual careers can be left-truncated. Current player
team/status and future career endpoints are not historical features. Reconstructed
transaction event dates still do not guarantee original publication vintages.

## How the existing research fits together

These are historical readouts with their own cohorts and frozen inputs, not
claims that every older model has been revalidated on the new consolidated data.

| Research | What it established | How to use it now |
| --- | --- | --- |
| [V2 review](../reference/v2_metrics_review.md), [fold repairs](../history/metrics_historical_fold_repairs.md), [metric report](../reference/metric_report.md) | Some archived V2/Adaptive ranking metrics were close to ECR; later repairs and matched populations changed comparisons | Retain as versioned benchmarks; near-parity, calibration, top-K recovery and value capture are different questions |
| [Integrity/role audit](data_integrity_and_role_research_2026-09-22.md), [repaired rerun](../history/historical_data_repairs_2026-09-22.md) | Scoring/population/roster/route/contract problems materially constrain interpretation; repaired RB role gain over usage is +0.65 four-week points with interval spanning zero | Repaired evidence takes precedence; role detail is not automatically incremental information |
| [Value capture](value_capture_research_2026-09-22.md) | Modern disagreements generally failed to recover enough market misses; historical acquisition-price limitations remain | Require an actual decision benchmark before calling a forecasting improvement an edge |
| [Incremental audit](incremental_information_2026-09-22.md), [RB residual correction](rb_residual_correction_2026-09-23.md) | Strong baselines absorb most gains; residual correction adds +0.36 four-week points per selection with uncertainty spanning zero and slightly worse MAE | Retain negative evidence; do not recycle the same extension as a proven signal |
| [Preseason role/workload](preseason_role_workload_2026-09-23.md) | 2019–2025 role-summary refits lose to saved V2 on QB/RB/TE point error | Better historical representations do not by themselves solve future workload |
| [QB starter recognition](qb_role_transition_2026-09-23.md) | On 556 QBs in 2019–2025, full mixture improves versus saved V2 by 8.22 points MAE; adding news beyond ECR mixture gains only 0.06 [-0.28, 0.44] | Keep the job-probability concept; validate the conditional workload experts separately. Lamar 2019 remains a major failure despite recognizing his job |
| [Player outlook](../history/player_outlook_2026-09-22.md) | 2019–2025 four-week forecasts have little/no RB/WR increment over recent usage; participation/range gates fail | Usage remains a meaningful short-horizon benchmark; this is not a preseason or 22-year validation |
| [College pathways](../history/college_nfl_pathways_2026-09-22.md), [rookie evidence readout](../history/research_evidence_and_rookie_readout_2026-09-22.md) | Draft capital is stronger; modest older WR/TE point gains uncertain, some participation/later-development signals | Keep college history, quality and identity links; distinguish NFL points from games/development targets |
| [Rookie watch](../history/rookie_watch_2026-09-22.md) | 2018–2025 analogs improve WR four-week error over scoring pace; QB sample small and worse | A useful separate in-season research view, not evidence of preseason market advantage |
| [Player profiles](../history/player_profiles_2026-09-23.md) | Canonical observed career, recent and participation summaries | The new audit now tests selected profile families; the profile UI itself is not a validated forecast model |
| [Absence evidence](../history/historical_availability_2026-09-23.md), [historical backfill](../history/historical_backfill_2026-09-23.md) | Repairs known availability constraints and greatly expands dated evidence | Essential input reliability work; more coverage is not automatic predictive gain |
| [Stat-system review](stat_system_review_2026-09-22.md), [implementation](../history/stat_system_implementation_2026-09-22.md) | Forecast units, uncertainty, publication provenance and decision-system design | Verify actual local artifacts rather than assuming an implementation report means a board was rebuilt |
| [Roster opportunities](roster_opportunities_2026-09-22.md), [intelligence workspace](../history/intelligence_workspace_2026-09-22.md), [UI research](../history/ui_research_2026-09-22.md) | Product presentation and current league applications | Useful delivery context, not independent historical forecasting evidence |

Existing residual-correlation screens are hypothesis generators. In particular,
`analyze_residual_patterns` currently labels the unfiltered completed population
`all_returners`; mixed-population rows can therefore appear under that label.
Do not import its “stable” labels as returner-specific proof. This new audit
explicitly filters position and population before fitting and comparison.

## The existing Next-gen model and the consolidation decision

There **is** an existing Next-gen family in
[`metric_report.yaml`](../../src/engine/config/metric_report.yaml):

- Returner PPG uses historical PPG, previous PPG, age, depth-role and component
  projection inputs. A separate games model uses prior/expected games and dated
  status context. Their product supplies season points.
- Rookie PPG and games each use draft capital and drafted status, then their
  product supplies rookie season points. Rookie and returner outputs are coalesced.
- It is `apply_live: false`, and is not a component of the frozen 2026 Adaptive
  selectors. It does not already incorporate the new player-profile families,
  college study, QB starter-recognition model or RB residual experiment.

Consolidate around explicit observations rather than treating “Next-gen” as one
already-validated aggregate score. The best current design priorities are:

1. **Foundation:** audited points/denominators, completed career plus recent
   exposure, draft capital for rookies, and explicit missingness/coverage.
2. **Position-specific additions:** QB weekly history; WR/TE age; QB/WR
   participation; mechanism-specific availability with hard factual constraints.
3. **First residual candidate:** isolate which WR participation components survive
   above ECR and both strong own-data baselines, then test actual ranking/selection
   value. RB availability is a secondary candidate; do not automatically stack it.
4. **Separate role recognition from workload:** preserve dated QB job evidence and
   opening-job probabilities, but repair/test conditional workload behavior before
   integrating a scenario forecast into Next-gen.
5. **Hold out of promotion:** target-dependent early history until canonical repair,
   undated contracts, unavailable vacated shares, unsupported depth history, broad
   college/tracking claims, and an unrestricted all-feature stack.

The new audit does not select a best production forecast. It identifies which
information has earned further use and where the market-relative hypothesis is
most concrete. A future Next-gen comparison must also rerun the existing V2,
Adaptive and Next-gen baselines on the same corrected candidates and outcomes;
this family screen is not that model-versus-model contest.

## Reproduction, verification and publication

```sh
.venv/bin/python research/nextgen_signal_audit.py \
  --gold canonical_20260923_r5 --version nextgen_signals_NEW_VERSION
.venv/bin/pytest -q research/test_nextgen_signal_audit.py \
  tests/test_data_pipeline.py tests/test_player_profiles.py
```

The runner is create-only. The final release contains source references,
implementation/protocol/test snapshots, feature and coverage Parquets, all
predictions, fold admission records, target-quality diagnostics and the report.
Every dependency and output hash was checked. Final r3 features, coverage and
predictions exactly reproduce the corrected r2 on the subsequently published
gold r5; r1 remains a contaminated diagnostic. All 14,608 scored outcomes match
the pinned gold labels, with no duplicate candidate keys or 2026 grading.
All **30 targeted tests** passed, covering the audit, release contracts and player
profile aggregation; Ruff, whitespace and report-link checks also passed.

This audit and its conclusions are **saved, but not wired into the dashboard**.
The consolidated data catalog and its profile/outlook/college products are a
separate publication. Existing boards and all 13 protected production/frozen
artifacts remain unchanged. The target correction is local to this research
view; it has not silently repaired upstream profiles or other models.

The follow-up [individual-stat audit](individual_stats_2026-09-23.md) now derives
an immutable corrected gold research release, extends the transitive target mask,
rebuilds full-career profiles and tests individual formulas plus combination controls.
It also provides a searchable 2026 scoring-consistency table. Neither research
release replaces the published catalog or live rankings.
