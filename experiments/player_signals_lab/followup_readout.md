# Additional experiments and research-path assessment

**The strongest follow-up evidence favors a separate uncertainty component, a weekly QB
role component, and partial RB team-volume adjustment. It does not support bundling the
new player-strength features into a replacement point model.** Everything remains inside
this disposable lab; no application models, registries, forecasts, or source data were changed.

Open the [interactive comparison report](runs/review_003/index.html) to filter **4,941
comparison rows** and examine coverage by population. The [research-path ledger](research_queue.md)
maps every original recommendation to executed work and remaining boundaries. Machine-readable
comparisons and supplemental diagnostics sit beside the report.

This closes the **bounded, feasible experiment menu** raised by the review. It does not
mean that NFL research is exhausted, that untested variants are disproved, or that data-limited
questions have been answered. The new [protocol](followup_protocol.md) was frozen before its
efficacy results. The [closure protocol](closure_protocol.md) explicitly identifies additional
checks proposed after those results; those checks are exploratory too.

**What ran.** Five completed stages reuse corrected, hash-pinned gold and fixed learners.
Expanding training starts in 2004; evaluation spans 2007–2025 where labels permit. Calibration
starts after three earlier OOF seasons, usually 2010; the signed-CQR extension starts in 2013;
joint receiving distributions start in 2015 because earlier target labels and OOF calibration
need warmup. Unknown targets are never converted to observed positive or zero targets.

The signal stage produced 486,202 forecasts across 256 position/target/year folds, 46,029
player-season-target cases, and 15,212 distinct player-seasons. Its 148 stored all-population
comparisons include 14 identity controls, leaving 134 nonidentity comparisons. The calibration
stage produced 1,226,104 interval records across 26 result parts. These counts include repeated
methods on the same outcomes; they are not independent observations.

**Uncertainty calibration, especially by population.** Expected workload explains more
variation in residual uncertainty than rookie/returner/market-only status alone. For season
points, the following compares predicted-workload-quartile calibration with the earlier global
scaled-residual baseline. Coverage and scores are averaged equally across the 2010–2025 seasons.
Positive gain means a lower proper interval score, which penalizes width and missed outcomes.

| Position | Matched player-seasons | Global 80% coverage | Workload 80% coverage | Interval-score gain | WIS gain |
| --- | ---: | ---: | ---: | ---: | ---: |
| QB | 1,664 | 82.4% | 80.9% | 14.0% | 9.4% |
| RB | 3,633 | 81.2% | 81.8% | 15.0% | 9.1% |
| WR | 5,064 | 82.1% | 79.3% | 15.1% | 9.6% |
| TE | 2,662 | 81.3% | 82.0% | 14.5% | 9.2% |

The interval-score gains survive the calibration stage's Holm correction (adjusted p≈.0332)
and retain positive gain intervals under player-career, franchise, and consecutive-season
block resampling. WIS includes both the 80/90% intervals and median error; its table is a
descriptive complementary score, not an additional adjusted significance claim.

**Overall coverage conceals a real WR problem.** Workload calibration covers only **74.8% of
rookie WR outcomes**, versus 80.3% with population-only calibration. Returner WR coverage is
81.1% under workload calibration, versus 79.0% under the global baseline. Market-only WRs are
also undercovered at 74.8%. Population-only methods improve the overall interval score only
about 0.5–1.9%; they address a different tradeoff. Population×workload, partial pooling,
asymmetric residuals, recent-window calibration, and annual adaptive updates were all run and
retained. Finer groups sometimes worsen coverage or fall back because samples are small.

Conditional quantile models and chronological CQR offer another useful comparison. QB coverage
improves from 74.3% uncalibrated to 81.1% with CQR, while improving interval score about 17.1%
against the global residual baseline. RB/WR/TE quantile intervals overcover at roughly 87–88%.
There were **1,752 player-seasons with crossed raw quantiles**; sorting repaired the order
before evaluation. This repair is part of the tested method, not evidence that the raw models
were already coherent.

The signed-CQR follow-up allowed contraction as well as expansion. On its matched 2013–2025
cohort, population CQR brings QB returners from 78.5% global coverage to 80.4%, with a 13.7%
subgroup interval-score gain. It does little to reduce RB/WR/TE overcoverage. Their lower
quantiles are almost always zero, many actual outcomes are zero, and the empirical correction
often lands exactly at zero. This is a substantive discrete-outcome limitation, not a reason
to hide zero-production players or tune toward exactly 80% on these reused seasons. None of
the signed corrections produced crossed final intervals.

Recommendation: retain workload and population diagnostics together in a **standalone
uncertainty experiment**. Do not declare a universally calibrated wrapper or choose separate
population winners from these retrospective slices. Freeze one candidate and its fallback
rules before a prospective test. [Full calibration results](runs/calibration_001/calibration_summaries.json),
[signed-CQR results](runs/audit_002/signed_calibration_summaries.json).

**Other predictive paths.**

| Path | Executed result | Interpretation |
| --- | --- | --- |
| Team volume | Attempts MSE improves 20.6%, carries 23.2%, against season-length-adjusted prior volume | Useful team-level mean reversion; this comparator is deliberately simple |
| RB team allocation | Half adjustment toward predicted team budgets improves carry MSE **3.41%** and MAE 2.05%; adjusted p=.0072 | Promising component, with the residual allocation estimated only from earlier cutoff-assigned cohorts |
| Full team allocation | QB attempt MSE worsens 8.46%; RB carry MSE worsens 2.64% | Coherent totals alone do not establish better player forecasts |
| Weekly QB role | Brier improves **7.95%**; log loss falls .4248→.3868; adjusted primary p=.0031 | Promising separate in-season role forecast |
| Preseason opening-role mixture | QB attempts MSE +0.37%, carries +0.80%, points +0.46% versus identical-input direct forecasts; MAE worsens | No convincing general improvement; do not replace direct totals |
| Joint receiving distributions | Paired simulation CRPS improves WR 8.50%, TE 6.74% versus direct residual simulation | Better distribution score against this baseline, but 80% intervals cover about 91%; not calibrated enough to promote |
| Joint receiving point means | WR yard MSE +0.32%, TE −4.30%; no adjusted evidence | A better distribution score is not automatically a better mean forecast |
| Multi-season/role pooling, team-relative rates, NGS, receiving execution | No nonidentity comparison in the 134-test signal menu clears Holm p<.05 | Retain as exploratory/descriptive features; no general point-model addition justified |
| College development | Draft information improves rookie-point MSE 36.6–43.2% over pre-draft-style college inputs; development features then worsen MSE 0.1–1.4% | Draft capital remains essential for the post-draft task; no demonstrated extra developmental increment here |
| Prior injury reports | No corrected MSE improvement in predicting next-season offensive-snap appearances | Participation model only; no inference of medical injury risk |
| Recency/age | Five-year training windows yield points-MSE gains QB +0.02%, RB −0.53%, WR −1.42%, TE −2.50%; no adjusted age increment for role/efficiency tasks | Keep older training history for these recipes; this does not mean age has no football relevance |

The half-adjusted RB carry gain survives career, franchise and two-season block sensitivities:
absolute MSE-gain 95% intervals are respectively [34.1,160.6], [49.3,140.2], and [57.4,132.9].
QB weekly-role improvement also survives career and block checks. These checks assess different
dependence structures; none proves full independence or resolves every era shift.

The weekly role event is **starting at least half the team's games in the next non-overlapping
four-week block**, among QBs who had already started earlier that season. The internal target
name contains “majority,” but the actual threshold is ≥50%, including ties. It does not cover
every unseen backup. Team allocation uses only known cutoff teams and retains an other-player
budget. Unknown team assignments are excluded from that matched experiment; trades can make
full-season player totals inconsistent with one preseason team. Trade flags are saved for audit.

The stronger control uses 224 archived profile/market inputs with current corrected market
fields substituted. All 17,396 training/evaluation panel rows matched current labels. Added
context/NGS/execution blocks do not show a reliable increment over that control. The dated QB
ledger supplies only 40 eligible panel rows, so its near-zero aggregate increment is not a
general rejection of news. Lamar Jackson's 2019 forecast is now 318 attempts with duration
features and 298 with the opening mixture, versus 401 observed; the archived 24-attempt failure
does not recur, but one repaired example does not validate the mixture.

**Data findings that change what we can claim.** The NGS raw snapshot retains published
exposures. They agree exactly with gold counts for 98.8% of passing, 99.1% of receiving, and
98.6% of rushing season rows. CPOE and YAC identities reconcile numerically. However, RYOE per
attempt agrees with total RYOE divided by published rush attempts within .02 on only 92.1% of
405 nonmissing pairs. Counts remain exposure proxies until eligible-play definitions and
provider vintages are verified. The [official NGS dictionary](https://nflreadr.nflverse.com/articles/dictionary_nextgen_stats.html)
defines the statistics; aggregate arithmetic alone cannot establish causal talent.

The pinned NFL source inventory has 217 provider tables and **no x/y/frame trajectory table**.
Ten tables contain primary-receiver route and pressure fields. Those are useful observations,
but do not supply routes for every receiver, tracking-based alternatives to the QB's chosen
target, STRAIN, or defensive positioning counterfactuals. The
[participation dictionary](https://nflreadr.nflverse.com/articles/dictionary_participation.html)
limits the route field to the primary receiver. These advanced scouting/attribution methods
remain data-limited, with observational team-context tests executed separately.

College identities include 134,499 unmatched and 368 review records; neither is a verified
NFL-nonentry label. No complete declaration/graduation/exit ledger is present. Raw roster
experience exists, but a spot audit even finds later draft-year data in historical biographies.
Our rookie experiments therefore concern an already NFL-selected population. Competition,
exit selection and biological development are not identified by the tested production trends.

Scoring and offensive-snap appearances disagree for 4.1% of QB, 23.7% of RB, 18.2% of WR and
19.9% of TE audited rows. Source game coverage is complete for the tested snap years, but
unmatched player identities remain excluded. The local injury table includes **2025** reports;
the earlier review's statement about a 2024 endpoint does not describe this capture. Current
[upstream documentation](https://nflreadr.nflverse.com/articles/nflverse_data_schedule.html)
also describes ongoing injury updates. This correction is recorded here without changing the
original historical review or shared datasets.

**Decision utility remains unproven.** Real 2026 draft, transaction and lineup snapshots
exist, but they do not provide decision-time alternatives and completed outcomes across the
2007–2025 forecast history. The historical “transaction replay” artifact is NFL roster evidence,
not executed fantasy trades. We ran fixed-slot mean/median/lower-bound selection diagnostics.
The original global adjustments select identical sets. Conditional quantile policies change
sets, but median-policy gain intervals include zero at every position. Lower quantiles often
tie at zero, making tie-breaking decisive: both player-ID and mean-forecast tie rules are
reported. These results support neither a downside-penalty ranking nor a claim of fantasy
profit. [Supplemental diagnostics](runs/review_003/supplemental_diagnostics.json).

**What to carry forward.** Keep three independently testable candidates: workload/population
uncertainty, weekly QB role retention, and partial RB team-budget adjustment. Keep the fixed
market-aware point model as the control. Combining candidates, learning richer contextual
effects, and fitting a new event-level xFP model would be new hypotheses, not validated
consequences of these results. Imported expectation-model vintages still prevent original-
decision-date claims for retrospective NGS/EPA/xFP values. Do not import an aggregate xFP
feature merely because it appears in an older archive.

The next clean preseason confirmation is **2027**: freeze the candidate list, cutoff, inputs,
method and predictions before the first regular-season game, and score completed outcomes
after the season. This is a documented requirement, not a running scheduled service. The
already-started 2026 season is not an untouched preseason holdout. No further historical
search is being presented as confirmation.

**Reproducibility and checks.** All five valid stage manifests report complete and unchanged
protected inputs. Earlier folds, calibration fallbacks, matched controls, full unfavorable
results, code snapshots and hashes are saved. Nineteen focused tests pass, including chronology,
signed calibration with extreme held-out labels, proper-score arithmetic, joint zero mass,
allocation residuals and write boundaries. Ruff passes. `audit_001` is preserved as failed:
reused interval rows carried an existing baseline-score column that collided with the new
matched join; the corrected run is `audit_002`. The final `review_003` includes explicit
zero-quantile tie sensitivity and separates calibration cohorts in its population chart.
Earlier report revisions are preserved; no trial was overwritten.

All point and interval gain intervals use paired season resampling unless a sensitivity is
explicitly named. Holm families are stage-specific, not one correction for every historical
research decision. Sign randomization assumes symmetric/exchangeable null season effects.
Repeated careers, source revisions, small modern samples and repeated historical model search
remain limits even when adjusted p-values and several bootstrap checks agree.
