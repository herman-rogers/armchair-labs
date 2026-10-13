# Opening Drive v3 weekly research

V3 adds weekly point forecasts and predictive distributions while preserving the
published v2 season/remaining-horizon models. The completed experiment is
`experiments/future_player_lab/runs/opening_drive_v3_20261011_002`; its frozen
panel contains 267,795 player-weeks, of which 252,315 have scheduled games, and
80 numeric pregame features. Known byes are deterministic zero forecasts outside
the weekly model. Missing labels remain missing rather than becoming zero. The final panel masks
source-wide missing receiving-target captures from 2003–08. The earlier `_001`
run is preserved as superseded; all reported results below use corrected `_002`.

## Reproduce and verify

Run these commands from the repository root using the existing locked environment.
For a fresh experiment from the **same frozen input panel**, choose a new output
directory and copy the original panel receipt before training:

```sh
V3_ORIGINAL=experiments/future_player_lab/runs/opening_drive_v3_20261011_002
V3_REPRO=experiments/future_player_lab/runs/opening_drive_v3_reproduction
mkdir -p "$V3_REPRO"
cp "$V3_ORIGINAL/panel.parquet" "$V3_REPRO/panel.parquet"
cp "$V3_ORIGINAL/dataset.json" "$V3_REPRO/dataset.json"
.venv/bin/python -m experiments.future_player_lab.campaigns.weekly_v3.run --output-dir "$V3_REPRO"
mkdir -p "$V3_REPRO/lineups"
cp "$V3_ORIGINAL/lineups/league_snapshot.json" "$V3_REPRO/lineups/league_snapshot.json"
cp "$V3_ORIGINAL/lineups/league_inputs.json" "$V3_REPRO/lineups/league_inputs.json"
.venv/bin/python -m experiments.future_player_lab.campaigns.weekly_v3.lineups "$V3_REPRO"
.venv/bin/python -m experiments.future_player_lab.campaigns.weekly_v3.audit "$V3_REPRO"
.venv/bin/python -m experiments.future_player_lab.campaigns.weekly_v3.report "$V3_REPRO" --output-dir data/outputs/opening_drive_v3_reproduction
```

The runner verifies the panel checksum against `dataset.json`. Fold reuse requires
identical panel, dataset, feature allowlist, candidate metadata, and training-source
hashes. A changed contract requires a fresh directory. Keep redirected logs **outside**
the run directory so mutable logs are not included in the artifact receipt.
`source_snapshot/` retains the training source. `manifest.json` binds the environment,
training source, and artifacts. Rebuilding from the current local source tables is
also supported, but is a new data vintage, not necessarily an identical replay:

```sh
.venv/bin/python -m experiments.future_player_lab.campaigns.weekly_v3.data --output data/research/opening_drive_v3_rebuild/panel
.venv/bin/python -m pytest experiments/future_player_lab/tests/test_weekly_v3_modeling.py experiments/future_player_lab/tests/test_weekly_v3_data.py
.venv/bin/python -m experiments.future_player_lab.campaigns.weekly_v3.replay --registry data/campaigns --campaign opening_drive_v3
```

The final command verifies and replays an already published v3 bundle. Within the
training run, `replay.json` independently records zero difference in every saved
2026 mean, standard deviation, and quantile for all four positions. The completed
run produced 720 fold prediction artifacts and 68 serialized estimator/calibration
bundles. Models use four CPU threads; the existing GPU workload was left running.

## Locked experimental design

Each forecast-season mean model trains only on earlier seasons. 2015–17 supply
initial out-of-time residuals; 2018–20 select one mean method per position by
equal-year RMSE, then one distribution by equal-year quantile-integrated CRPS.
Those choices remain fixed for 2021–25 evaluation and final 2026 fitting.
2026 weeks 1–4 are diagnostic outcomes; week 5 has no training/evaluation labels.

The nine mean candidates are v2 allocation, past-only recent persistence, direct
LightGBM using full history, a five-year window or four-year decay half-life,
LightGBM residual correction, direct models excluding variability or v2 features,
and a CPU XGBoost challenger. V2 and its residual-correction challenger are excluded
from primary selection: exact v2 ensemble forecasts exist only from 2021 onward;
earlier rows contain a dated persistence proxy. The recent control instead shrinks
the last four observed weekly scores toward the player's historical mean.

Six spread estimators hold the chosen weekly mean fixed: position-pooled residuals,
five predicted-mean bins, player-specific residual variance shrunk toward those
bins with a 24-week prior, conditional residual scale with and without personal
variance features, and 13 boosted residual quantiles. Calibration uses only
earlier out-of-time errors, with a five-season window. Conditional scale fitting
reserves the latest prior season for the standardized residual shape. V2 uncertainty
controls switch to earlier exact-v2 residuals when available, starting in 2022;
their 2021 calibration uses proxy residuals and is identified in the outputs.

## Results and interpretation

The main paired cohort contains 58,514 scheduled player-weeks with exact v2
forecasts in 2021–25. These are pooled-row metrics; equal-year method tables are
also saved, so numbers can differ slightly with aggregation convention.

| Position | Locked mean / spread | v2 RMSE | v3 RMSE | Nominal 80% coverage | Interval width |
| --- | --- | ---: | ---: | ---: | ---: |
| QB | Direct, full / quantiles | 6.788 | 5.614 | 73.6% | 9.020 |
| RB | Direct, no v2 / mean bins | 5.473 | 4.790 | 81.2% | 8.037 |
| WR | Direct, full / mean bins | 5.337 | 4.829 | 78.3% | 8.452 |
| TE | Direct, no variability / mean bins | 4.119 | 3.805 | 80.5% | 6.285 |

Sharing historical information across players works well here. Predicted scoring
level provides useful information about spread: mean-bin distributions outperform
position-pooled residuals. Personal residual-variance shrinkage does not beat the
mean-bin alternative in any position on the held-out distribution scores. Removing explicit
personal variance features barely changes conditional-scale distribution scores.
These findings support pooled, situation-specific uncertainty over a separate model
per player; they do not establish that player individuality never matters.

Full-history and decay point models are close, while five-year windows generally
lose some accuracy. Point improvement over v2 also includes the benefit of updating
features weekly: historical v2 remains fixed at preseason or after week 2. This is
not an architecture-only comparison. `mean_scores.csv`, `distribution_scores.csv`,
`cohort_scores.csv`, and `spread_calibration.csv` retain the full comparison and
show whether wider predicted weeks really have wider centered errors.

Do not equate squared forecast error with irreducible player variance:
conditional MSE equals conditional variance plus squared conditional bias.
The experiment records calibration bias separately and centers residual
distributions, but conditional-scale models still estimate forecast-error
dispersion, which can contain remaining model bias and estimation error. Variance
is not an additive bonus to expected points. Player scoring variance and residual
forecast variance are different quantities.

Calibration is an unresolved limitation: QB and WR intervals under-cover, and
some high-projection cohorts under-cover substantially even when overall coverage
looks acceptable. Raw quantile crossing occurs frequently (roughly 63–67% of rows
across positions), and sorting enforces final monotonicity. That repair does not
prove calibration. All distributions are recentered on the held-fixed point mean;
this can shift/smooth scoring-zero atoms. Probabilities of scores above zero are
not probabilities of playing. Negative lower tails are possible. The inverse CDF
is linear between 13 quantiles with flat 1%/99% tails; saved standard deviations
integrate that distribution exactly, while reported quantile-integrated CRPS is
a numerical approximation. These are research-preview ranges, not guaranteed
80% intervals. There is no post-hoc reselection using evaluation outcomes.

The years 2021–25 were used in previous project research, so this is retrospective
development evidence rather than an untouched external test. Source-vintage and
cutoff reconstruction limits are documented in `dataset.json` and `audit.json`.
K/DST mean transfer, covariance, synthetic lineups, and current-league aggregation
have their own receipts under `lineups/`; player-level results alone cannot
establish full-lineup calibration.

## Research context and remaining methods

[NGBoost](https://proceedings.mlr.press/v119/duan20a.html) learns conditional
distribution parameters through boosting; it motivates distributional prediction
but its implementation was **not** run here. [Quantile Regression
Forests](https://jmlr.org/papers/v7/meinshausen06a.html) demonstrate conditional
quantile estimation with trees; this experiment uses LightGBM quantile losses,
not that forest estimator. [Gaussian Process
Boosting](https://jmlr.org/papers/v23/20-322.html) combines boosting with Gaussian
processes/grouped random effects; its implementation was **not** run. The simple
player shrinkage candidate is not a fitted GPBoost or hierarchical Bayesian model.

Unexamined extensions include an explicit availability/active-points mixture,
conformal recalibration, a systematic pooling-prior sensitivity study, alternative
parametric families, publication-time injury/news and odds feeds, and a future
prospective holdout. Quantile calibration and high-usage cohort reliability are
the most direct next questions exposed by this completed bounded experiment.


## Weekly refresh without retraining

The normal campaign refresh driver also checks v3 even when v2 already contains
its current origin. Run it after an accepted current-season snapshot is captured:

```sh
.venv/bin/python -m experiments.future_player_lab.campaigns.rescore --due
.venv/bin/python -m experiments.future_player_lab.campaigns.weekly_v3.replay --campaign opening_drive_v3
```

To request only v3, add `--campaign opening_drive_v3`; `--no-capture` reuses an
existing supplement capture. The normal driver refreshes v2 before v3. If requesting
v3 alone while v2 lacks the matching origin, baseline-dependent models use the
explicit persistence fallback; receipts record the exact-v2 fraction and this policy.
The refresh uses the accepted snapshot's completed
week and predicts the following week, applies sealed mean/distribution weights,
and retains original sealed lineup calibration without consuming new K/DST
errors. There is no model fit. The temporary snapshot/availability overrides are
restored even if feature construction fails. These weights support 2026 only;
a new season requires a new training run.

Each new forecast references its own `forecast_matrix_2026_<origin>` artifact and
dataset receipt, allowing offline replay without live sources. Existing origins
are never overwritten; existing weights and forecast files remain byte-identical.
Replacement seals an appended revision and archives the prior bundle. Duplicate
origins are a no-op. Future/mismatched origins, observed forecast outcomes, duplicate
identities, source/environment drift, and missing features fail before publication.
The refresh tests exercise archival, byte preservation, no-op behavior, source
restoration, bad inputs, and the v3-due/v2-current path.
