# Protocol 1: opportunity, execution, and uncertainty

Written before the first efficacy run. This is a fixed exploratory experiment menu, not
a confirmatory study or an automated promotion process. The historical folds have been
used by earlier repository research. All planned comparisons and unfavorable results
remain in the run. Do not describe a nominal positive interval as proven improvement.

**Data and clock.** Corrected gold is pinned by version and SHA-256. Use the existing
preseason candidate population and cutoff dates, retaining rookies, market-only entries,
and zero-production outcomes. Features use only earlier NFL seasons and existing
cutoff-safe market snapshots. Training begins in 2004; first evaluation is 2007, with
at least three earlier seasons. Every fold trains on all earlier completed outcomes.
The last completed label season is 2025; 2026 is not used for efficacy or calibration.
Original provider publication vintages are incompletely known: call this retrospective.
The imported release reader, NextGen metrics module and package initializers are also
pinned by SHA-256. The mutable current catalog is monitored, but is not an input selector.
Unrelated workspace changes are recorded separately from experiment dependency changes.

**Tasks.** QB attempts, carries and league points; RB carries, yards/carry and league
points; WR targets, receiving yards and league points; TE targets and league points.
Efficiency requires a positive valid denominator; other candidates remain included in
count/points tasks. Missing target labels and feature coverage are recorded per fold.
This is not a medical availability model, all-position scouting system, or weekly model.

**Controls.** Import the library's exact prior/population reference and fixed histogram
boosting recipe using all `x_` history columns. Reuse its nonnegative forecast and known
absence constraints. Lab regression recipes use the same fixed hyperparameters.
Add explicit historical rate/exposure features for the raw control. Add ECR, ECR spread,
overall ECR and market presence for the market control. Both have identical cohorts;
also report the predeclared market-covered subset. A market-control result is not a
replication of the wider individual-stat study's full feature screen or a pricing test.

**Pooling.** By position, fit a univariate Gaussian marginal likelihood for each prior
rate with variance `between_player_variance + sampling_variance / opportunities`.
Learn the mean and both variances on training-row historical rates only; no held-out
rate or label fits a hyperparameter. Reliability is `n / (n + sampling_variance /
between_player_variance)`. Compare pooled versus raw rates with the same log exposures,
same remaining features, same learner, same cohorts. No source's NGS statistic is
weighted with an unrelated box-score denominator. Unknown rates stay unknown. Failed
or undersized prior fits revert to no shrinkage and are recorded. This is an approximate
sample-aware forecast feature, not a hierarchical context-adjusted talent estimate.

**Receiving participation.** Use the existing rich weekly summaries whose underlying
weekly participation requires matching captured and expected team dropbacks. Retain
mean participation, mean targets per passing-play participation, and each coverage
fraction. Means of weekly ratios are not annual pooled ratios. Keep partial coverage
visible; no absent measure is made zero. The covered slice requires at least 9/18 grid
weeks in both measures. These are not verified routes or proprietary all-route openness.

**QB role mixture.** Define a meaningful passing-use game as at least 15 attempts.
Stratify a completed season as zero such games, 1–7, or 8+. These are outcome labels,
not preseason predictors. Fit a classifier on earlier labels and a full-season workload
regressor within each earlier role state. Integrate `sum(p(state | x) * E(total | state,x))`.
This preserves a conditional relationship and avoids multiplying marginal expectations.
The matching direct model receives exactly the same predictors, including the earlier
season's meaningful-use game count. Sparse experts use their training mean. This coarse
duration proxy does not validate opening-starter probabilities, injuries or dated news.

**Uncertainty.** Produce historical out-of-fold errors for each model independently.
Before forecasting a season, calibration may use only the last five *earlier* evaluation
seasons, with at least three seasons and 100 rows. Never use another player's outcome
from the current test season. Compare absolute-error intervals and errors scaled by
`sqrt(max(prediction, median_positive_training_outcome, 1))`. Use the finite-sample-corrected
empirical 80% and 90% quantiles. Count bounds intersect zero; yards, efficiency and
fantasy points retain negative lower tails. Known zero availability receives the existing
zero-total constraint. These are chronological residual intervals, not guaranteed
exchangeable conformal coverage or full joint predictive distributions. Report coverage,
width, proper interval score, and population slices. The median adjustment adds only an
earlier OOF residual median; it is an approximate global correction, not conditional
quantile regression. Compare its MAE with the original mean prediction on identical rows.

**Evaluation.** Primary mean-prediction loss is MSE; also report MAE, equal-season paired
gains, percentage gains, positive years, leave-one-year-out minimum gains, and nominal
95% intervals from 4,000 paired season-bootstrap draws. Seasons, not individual rows,
are the resampling unit; players and teams still recur across seasons. Intervals do not
account for every dependency, era shift, multiplicity, or repeated historical exploration.
No family-wide significance or superiority claim follows from a positive cell.

Predeclared slices: all candidates, each candidate population, 2017+, observed ECR,
participation coverage >=9 weeks, prior primary-opportunity exposure 1–49, and QB prior
meaningful passing usage <8 games. Low exposure uses attempts for QB, carries for RB,
targets for WR/TE. These predictor-defined slices do not select on future success.
Tiny slices are descriptive only; retain N and years. No post-outcome cohort tuning.

**Next step decisions.** Favor a small follow-up hypothesis only if direction and useful
effect size survive relevant controls, loss choices, eras and cohorts. Replicate against
the strongest applicable archived challenger before claiming a system-level advance.
Design prospective evaluation on genuinely future cutoffs separately. This lab never
edits serving gates or chooses a live model, and gains do not establish fantasy utility.
