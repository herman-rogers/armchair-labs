# Running back GPU research

This independent research campaign learns from the full eligible numeric inventory
of the pinned canonical raw snapshot, never the prepared feature matrix or a prior
feature registry. It does not change the application's serving policy.
The [October 3 findings](../../../docs/research/rb_gpu_campaign_2026-10-03.md)
summarize the completed comparisons and their limitations.

To reuse the saved models on current raw observations, see the
[saved-model forecasting workflow](../forecasting/README.md). Its dated RB preview
uses the original Week 2 horizon and requires no retraining.

Restore and install using the repository's shared-data and GPU setup guides, then:

```sh
uv sync --extra shared-data --extra research --extra research-gpu
.venv/bin/python -m patron.data.shared fetch --profile notebooks
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 POLARS_MAX_THREADS=4 .venv/bin/python \
  -m experiments.future_player_lab.gpu_lab.run --run-id rb_gpu_20261003_001 --parallel-cpu
.venv/bin/python -m experiments.future_player_lab.gpu_lab.finish \
  --run-id rb_gpu_20261003_001 --wait
.venv/bin/python -m experiments.future_player_lab.gpu_lab.audit \
  --run-id rb_gpu_20261003_001
```

`finish` waits for completed training if needed, then runs chronological model
selection and uncertainty scoring, source/usage diagnostics, final model refits,
and exportable figures. Each step is also available as a module (`analyze`,
`diagnostics`, `refit`, `figures`). `audit` checks the configured fit inventory,
artifact hashes, training chronology, finite forecasts, and verified CUDA execution.
These checks establish artifact integrity, not prospective predictive validity.

A separate projection-size sensitivity study uses the same XGBoost anchor at 16
and 128 components for PCA, source-balanced PCA, SVD, and random projection. It
reuses verified main-run raw/48-component forecasts as controls and waits until
main postprocessing is finished before using the GPU:

```sh
.venv/bin/python -m experiments.future_player_lab.gpu_lab.dimension_study \
  --main-run-id rb_gpu_20261003_001 --run-id rb_gpu_dimensions_20261003_001 --wait
```

Its eight additional recipes produce 1,024 target/fold fits. Comparisons are
descriptive development evidence; they do not change the frozen 52-recipe policies.

Two supplementary checks address baseline strength and a discovered raw-field
coverage gap:

```sh
.venv/bin/python -m experiments.future_player_lab.gpu_lab.calibrated_baselines \
  --run-id rb_gpu_calibrated_baselines_20261003_001 --threads 2
.venv/bin/python -m experiments.future_player_lab.gpu_lab.calibrated_baselines \
  --run-id rb_gpu_calibrated_baselines_20261003_001 \
  --integrate rb_gpu_20261003_001 --wait --threads 2
.venv/bin/python -m experiments.future_player_lab.gpu_lab.draft_capital_study \
  --run-id rb_gpu_draft_capital_20261003_001 --main-run rb_gpu_20261003_001 \
  --dimensions-run rb_gpu_dimensions_20261003_001
```

The affine baseline learns an intercept and nonnegative slope from earlier
production and persistence forecasts. Expanding-history and recent-three-season
fits are compared using earlier validation years. This adjusts the point forecast;
the main analysis's `persistence_calibrated` instead calibrates a residual
distribution around the original persistence forecast. The affine check was
specified after viewing the completed preseason preview and remains a separate
development sensitivity analysis.

The frozen main input has 4,810 fields. It explicitly excluded combine draft
fields and omitted roster draft pick when mixed source schemas promoted that
column to text. The draft study appends known-as-of pick, round, and elapsed draft
years while preserving every original row and value. It compares fixed raw
LightGBM and XGBoost anchors, records source/date checks, and treats missing draft
information as unknown. Its 256 evaluation fits and 32 final refits are separate
from the main search. CPU fitting starts immediately; GPU fitting waits for the
dimension study. Neither supplement revises the frozen main selection policies.

The profile-only and play-level companion benchmarks are separate tasks:

```sh
.venv/bin/python -m experiments.future_player_lab.gpu_lab.profile_build \
  --run-id rb_gpu_profiles_new --years 2021 2022 2023 2024 2025 --threads 4
.venv/bin/python -m experiments.future_player_lab.gpu_lab.profile_predict \
  --run-id rb_profile_predict_new \
  --profile-root experiments/future_player_lab/runs/rb_gpu_profiles_new
.venv/bin/python -m experiments.future_player_lab.gpu_lab.profile_integrate \
  --run-id rb_gpu_20261003_001 \
  --profile-root experiments/future_player_lab/runs/rb_gpu_profiles_new --threads 2 --wait
.venv/bin/python -m experiments.future_player_lab.gpu_lab.profile_players \
  --run-id rb_gpu_20261003_001 \
  --profile-root experiments/future_player_lab/runs/rb_gpu_profiles_new
.venv/bin/python -m experiments.future_player_lab.gpu_lab.coherence \
  --run-id rb_gpu_20261003_001
.venv/bin/python -m experiments.future_player_lab.gpu_lab.play_context \
  --run-id rb_gpu_play_context_20261003_001 --device cuda
```

`profile_build` creates all four geometries and verifies their source artifacts;
`profile_predict` measures profile-only production forecasts, and
`profile_integrate` joins chronological profiles to the main campaign's forecast
errors and interval coverage. `profile_players` builds a searchable named-player
history browser from verified full-data descriptions. `coherence` checks expected
count relationships and agreement between independent direct-point and component
forecasts; it does not correct them. The play-level command restores no data itself:
it reads the separately pinned Big Data Bowl
2020 release, joins source play IDs directly, and forecasts rushing yards at
handoff. Its play-level results do not imply season-level forecasting skill.

Main training run IDs are create-only. Resume interrupted main training with its
original arguments plus `--resume`, after its earlier process has exited. Configuration, training code,
packages, input contracts, and completed artifact checksums must match. A failed
model is recorded, never silently replaced with CPU or another estimator. Inspect
`manifest.json`, `failures.json`, and `progress.json` alongside the report. The runner
uses one GPU fit at a time and bounded CPU threads. `--parallel-cpu` adds one
independent CPU worker with isolated checkpoints and verifies matching inputs
before assembling the combined result.

`final_models/` retains checksum-verified preprocessing and model members, along
with the four policies selected for next use from earlier out-of-year forecasts.
For a new feature matrix that follows the same raw input and forecast-time
contract:

```python
from experiments.future_player_lab.gpu_lab.refit import load_policy, predict_policy

bundle = load_policy(
    "experiments/future_player_lab/runs/rb_gpu_20261003_001/final_models",
    horizon="remaining",
    target="rushing_yards",
    policy="ensemble_diverse",
)
prediction = predict_policy(bundle, x, feature_names, exposure)
```

Here `x` is the untransformed raw-derived matrix, `feature_names` identifies its
columns, and `exposure` is the forecast's calendar-week count for each row. The
loader restores each member's transformations and aligns columns by name. Use
`points_from_stats` to sum component means with the recorded PPR coefficients.
These models train through 2025; replay checks establish serialization integrity,
not new 2026 forecast accuracy. Recorded historical scores use the earlier-year
models, not these final refits.

The default search freezes 52 recipes before fitting: four engines crossed with
eight unsupervised representations, histogram and extra-tree controls, eight seeded
hyperparameter/history variants, eight source ablations, and two opportunity ×
efficiency challengers. Families include XGBoost CUDA, CatBoost GPU, NVIDIA cuML
random forests, and CPU LightGBM, histogram boosting, and extra trees. Hyperparameter
ranges include tree count, depth, regularization, column fraction, leaf support,
learning rate, and history length. CPU/GPU benchmarking is separate from accuracy
comparison; differing algorithms are not treated as hardware speedups.
The main cuML anchor uses 64 trees and four streams; its seeded search includes
32/64/128 trees. This follows the representative throughput benchmark, not
evaluation outcomes. GPU results can depend on floating-point scheduling.

The original numeric observations expand into prior-season, current-season,
recent-four-week, prior-two-season, career, recent-eight-observation, and
last-observation histories, plus college, combine, team, and calendar context.
Eligibility excludes identifiers, future outcomes, future closing lines, and sources
without a usable identity/time contract. Current-season participation known to be
published after postseason is withheld. Older published participation remains
eligible. Missing recorded labels remain unknown for each stat separately; a known
completed-period absence is zero. Native missingness indicators distinguish absent
observations from observed zero.

Each representation fits only earlier seasons: median imputation, variance checks,
correlation filtering, ordinary or source-balanced PCA, SVD, sparse random
projection, feature-cluster representatives, or all raw inputs plus PCA.
Correlation and clustering use bounded,
seeded approximations documented in each fitted summary. Learned components do not
establish the importance or causal effect of a single original stat.

The default horizons are preseason full season and remaining season after Week 2.
All available 2002–2025 examples may supply history. Candidate forecasts for
2018–2020 initialize earlier-year validation; 2021–2025 are the later evaluation
years. Each year selects its best recipe and equal-weight top-three ensemble using
only the preceding three seasons' mean normalized MSE.
Different-family equal blends and simplex-constrained blends also compete; the
latter learn weights from earlier out-of-year predictions with fixed weak L2
regularization. Seven RB stats and direct
rushing/receiving PPR points are separately forecast. A second PPR estimate sums
component-stat means. Opportunity × efficiency models test an approximation to
the conditional mean; they are not assumed to be mathematically equivalent to a
direct model. Receptions and touchdown efficiency are constrained to [0, 1].
For carries, targets, and direct PPR, the factorized recipes use the same direct
specification as their raw/correlation anchors. Top-three selection counts recipes
and may therefore include duplicate specifications for those targets; the
different-family policies require distinct engines. This is disclosed without
changing the frozen policies after evaluation.

Point metrics are RMSE, MAE, R², and Spearman association. Persistence comparisons
use identical player-origin rows. Season-block bootstrap intervals reflect between
season variation; five seasons provide limited uncertainty resolution. Empirical
CRPS, 80% coverage, and width use nearest-production residuals from earlier
out-of-year predictions. Those are marginal distributions; summing stat means
does not manufacture a joint points distribution. No normalized likelihood is
claimed for the empirical residual CDF.
Selection and calibration reuse earlier years, so conditional coverage may be
optimistic. Held-out coverage is measured and is not a guarantee.

Full-data player profiles are descriptive and may use all years. Separate fixed-k
profiles fit only earlier seasons for conditional forecast-error analysis. Cluster
numbers are local to their fit and do not imply stable semantic identities across
years. Small groups and groups dominated by era or source coverage require caution.
Held-out source permutation and native feature importance are diagnostics, not
same-fold feature selection. Source ablations and controlled engine/representation
comparisons help distinguish redundancy from useful information.

The canonical snapshot covers a broad but finite inventory. Newer auxiliary FTN
and Big Data Bowl frame-level releases are inventoried separately, not silently
joined without validated identities and publication times. Historical provider
revisions, imperfect newcomer coverage, and last-observed team assignment limit
backtest realism. PPR here excludes fumbles, passing, league bonuses, returns, and
conversions. Same-date ECR is unavailable in this raw contract. The repeatedly
examined 2025 season is development evidence, not a fresh untouched test.

## Research context

- The [nflverse NGS dictionary](https://nflreadr.nflverse.com/articles/dictionary_nextgen_stats.html)
  distinguishes rushing path efficiency, stacked-box rate, yards over expectation,
  receiving separation, air-yard share, and YAC above expectation. These describe
  different dimensions; they should not be collapsed into a single claim of talent.
- [NFL expected rushing yards](https://www.nfl.com/news/next-gen-stats-intro-to-expected-rushing-yards)
  uses tracking context at handoff. Our season forecasts address a different task.
- [nflWAR](https://arxiv.org/abs/1802.00998) motivates separating individual production
  from surrounding context and quantifying uncertainty. This campaign's predictive
  associations are not causal player-attribution estimates.
- [Route identification](https://arxiv.org/abs/1908.02423) and
  [unsupervised coverage identification](https://arxiv.org/abs/1906.11373) demonstrate
  learned football roles using tracking data. Applying role discovery to our
  player-history matrices is a research hypothesis, not a replication of those
  trajectory-based methods.
- [Grinsztajn, Oyallon and Varoquaux](https://arxiv.org/abs/2207.08815) investigate
  feature orientation and irregular target functions in tabular modeling. This
  motivates keeping raw-feature controls when testing projections; their results
  do not establish which representation will win on this NFL dataset.

These sources inform hypotheses. The saved experiments determine whether those
hypotheses help this RB population and forecast horizon.
