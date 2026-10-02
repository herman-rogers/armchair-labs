# Raw-data distribution experiment — initial findings

The completed RB experiment supports **opportunity being more persistent than
efficiency**, but **this decomposition simulator does not beat direct boosted trees**.
The count distribution that works best also depends on the forecast horizon.

These are exploratory results from `runs/raw_distributions_001`: all available
complete historical years, 2002–2025 forecast panels using raw observations from
2001 onward; 2023, 2024 and 2025 test folds; 1,000 draws per forecast. The raw snapshot
contains 4,222 assets. The source audit produces 2,250 numeric candidate inputs
without using any previous prepared features or forecasts. Different fits remove
different all-missing/constant columns using their fitting years only.

The score target is **rushing/receiving PPR**: yards/10 + receptions + 6×TD.
Fumbles, passing, returns, two-point conversions and league bonuses are excluded.
Weekly means Week 3, predicted after Week 2; remaining means Week 3 onward. These
results do not cover every weekly origin, although the notebook and runner support them.

| Horizon | Test player-origins | Prior-rate CRPS | Full raw tree CRPS | Best direct variant CRPS | Decomposed + age CRPS |
|---|---:|---:|---:|---:|---:|
| Next week | 531 | 2.019 | 1.961 | 1.931, without PBP/NGS | 2.032 |
| Remaining season | 531 | 24.003 | 21.113 | 21.113, full raw | 24.866 |
| Preseason full season | 506 | 32.133 | 28.721 | 28.541, without PBP/NGS | 37.800 |

Smaller CRPS is better. The best direct variants improve on the baseline by
**4.4%, 12.0% and 11.2%**, respectively. The weekly difference remains uncertain:
its descriptive player-cluster 95% interval is **−0.204 to +0.020 CRPS**.
For the full raw tree, remaining-season and preseason intervals are
**−4.816 to −1.001** and **−5.184 to −1.597**. These intervals do not account for
shared-team dependence or choosing among recipes after inspecting their results.

The full raw tree's central 80% intervals cover **80.6%, 78.3% and 76.3%** of
weekly, remaining-season and preseason outcomes. The decomposition's weekly
coverage is higher (88.5%) with a worse CRPS: extra coverage alone is not success.

The decomposition does benefit from efficiency shrinkage on weekly/rest-of-season
forecasts: unshrunk CRPS is **2.073 / 26.422**, versus **2.033 / 24.927** after
shrinkage. The age curve adds little. Preseason shrinkage does not help this
implementation. This does not reject every decomposition model; its simple team
assignment, share residuals and availability process still need improvement.

The descriptive preseason correlations are **0.657 for carries/calendar week**
and **0.584 for targets/calendar week**, versus **0.184 for yards/carry** and
**0.074 for rushing TD/carry**. Efficiency correlations condition on positive
denominators in both periods. They are not causal estimates or an aging prior.

For touchdown distributions, the following compares identical squared-error tree
means under two normalized count laws. Negative-binomial dispersion is estimated
only on the separate calibration year.

| Horizon | Poisson negative log score | Negative-binomial negative log score |
|---|---:|---:|
| Next week | **0.373** | 0.389 |
| Remaining season | 1.857 | **1.573** |
| Preseason full season | 2.274 | **1.743** |

Poisson works better here for one week; negative binomial helps on longer horizons.
Poisson-loss boosting does not beat squared-error boosting in this configuration.
This compares **tree training losses and count probability laws separately**;
it does not compare discrete likelihoods with an incompatible continuous density.
The simulator is scored with CRPS because it has no evaluated analytic density.

The closing-line sensitivity gives weekly CRPS **1.961**, essentially the same as
the full raw tree. It is timing-optimistic and does not establish that market
information is useless. Original pre-cutoff betting snapshots are unavailable.

The main limitations are reconstructed provider vintages; incomplete raw injury
timestamps; last-observed team assignments; undrafted/unlinked preseason newcomers;
simple horizon-level availability/efficiency simulation; and only three inspected
test seasons at one in-season origin. College uses exact unique ESPN IDs only.
No Bayesian hierarchy, neural-network benchmark, fixed age peak, or production
ranking change is part of this experiment.

The notebook exposes every raw source's inclusion/exclusion policy, candidate
feature coverage, annual scores, workload cohorts, calibration plots, individual
CDFs, and downloads. The run folder retains the exact computation sources, input
hashes, settings, package versions, forecasts, draws and target audits. Eleven
tests cover scoring identities, all three engines, future-label and raw-cutoff
invariance, unknown versus absent outcomes, and simulation constraints. The notebook
also passes marimo checks and executes with the saved results.
