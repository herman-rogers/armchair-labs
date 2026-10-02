# Research notebooks

Planned local GPU benchmarking, broader model searches, and GCS/Parquet storage with
a local cache are described in the
[compute and shared data plan](../../../docs/research_compute_and_shared_data_plan_2026-10-02.md).
Marimo and notebooks remain local on each machine; deployment and authentication
setup are deferred.

Start one marimo server with a home page listing all notebooks:

```sh
bash experiments/future_player_lab/notebooks/launch.sh
```

Open `http://127.0.0.1:2718` and choose `raw_distributions.py`, `rb_stats.py`, or
`boosted_trees.py`.
New marimo notebooks saved in this directory appear in the same browser.
Listing notebooks does not execute them or train models.

The new **raw-data distribution experiment**, `raw_distributions.py`, starts from
the full pinned raw snapshot rather than any prepared feature matrix. It inventories
every asset, exposes original rows and schemas, and derives historical numeric
inputs with explicit source exclusions and forecast cutoffs. Compare direct boosted
trees with team volume → competing player shares → efficiency simulations, no-shrinkage,
learned shrinkage and age variants, and Poisson/negative-binomial touchdown forecasts.
There are no Bayesian hierarchical models. CRPS and count negative log scores lead
the evaluation; interval calibration, width, player CDFs and point errors accompany them.

The notebook displays the completed local RB experiment across 2023–2025 tests and
preseason, next-week and rest-of-season horizons. Opening it does not retrain anything.
Its form supports RB/WR/TE, LightGBM/XGBoost/histogram boosting, multiple test years
and forecast origins. All source years are available; unfinished 2026 targets are
excluded. The initial weekly/rest-of-season run forecasts after Week 2. The optional
closing-line experiment is labelled as timing-optimistic because original betting
publication timestamps are unavailable.

The points target is rushing/receiving PPR (yards/10 + receptions + 6×TD), excluding
fumbles, passing and league-specific bonuses. This experiment does not reuse the
production scoring policy or publish predictions. Raw historical injury reports
are filtered by their own timestamps; it does not apply the other notebook's
reviewed injury-archive overrides. See [the readout](raw_distribution_readout.md)
for findings and limitations.

Reproduce it with a **new** output directory (existing runs cannot be overwritten):

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m \
  experiments.future_player_lab.notebooks.run_raw_distributions \
  --out experiments/future_player_lab/runs/raw_distributions_002 --market
```

The runner saves source code, configuration, source hashes, package versions,
simulation draws, individual proper scores, target audits and a Markdown readout.
Use `--origins 2 6 10 14` to examine additional weekly origins; those are full
candidate populations at each cutoff. Run validation with:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest \
  experiments/future_player_lab/tests/test_raw_distributions.py
experiments/future_player_lab/.notebook-venv/bin/marimo check \
  experiments/future_player_lab/notebooks/raw_distributions.py
```

The separate, simpler **RB stat prediction** notebook, `rb_stats.py`, has
only model/settings selection, RB stat predictions, and train/validation/test
diagnostics. It uses all 2,929 prepared features, including dated market inputs.
Choose histogram boosting, LightGBM, XGBoost, CatBoost, random forest or extra trees;
model constructors and `.fit()` are editable notebook cells. Nothing loads or
trains until **Train RB models** is submitted. It does not reuse cached forecasts.
Results and fitted estimators stay in the notebook session; predictions and settings
can be downloaded. In Python, `results["rushing_yards"]["fitted"]` contains the
final estimator, active column indices and optional fitted imputer.
The data/target helpers are shared; the training implementation is in the new notebook.

The RB notebook now supplements those inputs with eight dated availability features
from the pinned `injury_archive_20260925_r1` release. A reviewed, confirmed absence
covering the entire forecast period sets **both model and baseline** to zero.
Partial/uncertain absences are features, not proportional discounts. Training labels
remain unchanged, and raw/adjusted predictions and metrics are shown separately.
The archive coverage panel is available before training. New rules and inputs take
effect on the next submitted training run; opening or updating the notebook does
not retrain it. See the [injury archive audit](../../../docs/injury_archive_2026-09-25.md)
for coverage gaps, timestamp recovery, chronology rules, and validation.

The RB notebook also evaluates **predictive distributions with CRPS**, alongside
RMSE/MAE. Choose `crps` as the validation metric to select tree count by distribution
quality. The default adds three chronological calibration fits per stat: for a
2025 test, these predict 2021–2023 using only seasons before each calibration year.
2024 remains validation, and 2025 remains test. Each tree-count candidate gets its
own historical residual distribution. The final refit reuses the selected earlier
calibration; it never estimates uncertainty from test errors or individual-tree spread.

Distributions translate residuals from the nearest historical predicted production
rates (100 neighbors by default), scale rates to the requested period, and censor
count forecasts at zero. Yardage may be negative. These are empirical marginal
forecasts, not a parametric likelihood or a joint distribution over all seven stats.
Residual calibration can shift the distribution mean away from the original point
prediction. Injury overrides create a point mass at zero, identically for the model
and baseline. The baseline receives its own out-of-season residual calibration.

The **Distribution quality** section includes raw/adjusted CRPS, improvement over
baseline, 50%/80%/95% interval coverage and width, a coverage chart, player CDFs,
and downloads of per-player scores, predictive samples and calibration forecasts.
Lower CRPS is better; compare it within a stat/horizon because it has the stat's
units. Coverage alone is insufficient—very wide intervals can achieve high coverage.
Discrete outcomes and zero atoms can produce conservative interval coverage.
Calibration is approximate and may transfer imperfectly between seasons or to the
final refit; inspect coverage rather than assuming it is guaranteed. Training CRPS
is an in-sample diagnostic. Validation selects settings; test data never fit the
uncertainty distribution. Tuning repeatedly against test scores still compromises
that test season, just as it does for point metrics.

`rb_probability.py` computes the exact empirical-distribution CRPS:
`mean(abs(draw - actual)) - mean(abs(draw_i - draw_j))/2`, with denominator `m²`
in the pairwise term. This scores the reported finite distribution itself; it is
not the "fair" Monte Carlo estimator for an assumed underlying distribution.
See the [scoringrules explanation](https://scoringrules.readthedocs.io/en/latest/crps_estimators.html).
No log-likelihood is displayed because no continuous density model is assumed.
The implementation tests compare against an independent pairwise calculation,
point-mass limits, scaling, censoring, and invariance to changed test outcomes.
Notebook wiring tests use fake estimator callbacks and demonstrate that CRPS can
select a different tree count from RMSE. No models were trained for this update.

The RB notebook's **How good are these predictions?** section puts final-test
errors in context: actual mean/median/90th percentile, WAPE (absolute error divided
by absolute production), R², baseline improvement and the fraction of players
within a user-chosen absolute tolerance. It also shows each player's prediction
beside the actual outcome. All-player and positive-production views use existing
predictions and never retrain. Percentages with zero denominators remain undefined;
WAPE is not a percentage of correct predictions. Reporting calculations live in
`rb_accuracy.py` and are tested with fixed numbers, without model fitting.

The original comprehensive notebook is `boosted_trees.py`.
For an optional direct launch of the RB notebook, `launch_rb.sh` still opens it
on port 2719. `MARIMO_NOTEBOOK` can also override the main launcher's directory.

The local marimo home page opens at `http://127.0.0.1:2718`. Change the port with
`MARIMO_PORT=2719`. It listens only on the local machine. Use `--headless` to start
without opening a browser. The editor lets you inspect and change every cell.
If your editor settings leave cells idle, press **Run ▶** once. Use the bottom
toolbar's **Preview** button to hide code while exploring; switch back to edit cells.
The recipe-preset selector loads the original boosters' settings or a new engine.

The launcher installs marimo 0.25.0 and Plotly 7.1.0 into the lab's ignored
`.notebook-venv`, then references the repository's existing scientific packages.
The completed campaign's isolated `.packages` supplies LightGBM/XGBoost/CatBoost.
It does not change the application's dependencies or lockfile. On a fresh checkout,
first follow the [campaign setup](../deep/README.md); its local data and completed
`runs/` artifacts are required and are not committed.

## Contents

1. Original `profile_boost` and `enriched_boost`, the actual published policy,
   individual new models, and both new-only and combined-library ensembles.
   Position/horizon/season/cohort/model filters; matched metrics and annual curves.
2. Predicted-versus-actual, residuals, cohort errors and individual large misses.
3. Candidate counts, outcome distributions, feature missingness and input exploration.
4. Saved train/later-season tree-capacity curves, with both axes in the same units;
   ensemble library-size curves and exact recipe parameters.
5. Annual ensemble method selections and inspectable weights/stacking parameters.
6. Engine explanations, parameter semantics, and a submitted tree-training form.
7. A custom ensemble form with editable membership and equal/convex/greedy weights.
8. Reproduction guidance and links to official model documentation.

## Training and saved results

The new-model form defaults to **RB / all relevant stats**, with position-specific
inputs from the expanded own-data inventory. Set position and horizon at the top,
then use **What to predict** in section 6:

- QB: pass attempts, completions, passing yards/TDs/interceptions, carries and rushing yards/TDs.
- RB: carries, rushing yards/TDs, targets, receptions and receiving yards/TDs.
- WR/TE: targets, receptions, receiving yards/TDs and secondary rushing production.

Choose one stat or **All relevant stats (separate models)**, then **Train tree
experiment**. Each target has its own fitted model and validation-selected tree
count. The resulting player table contains predicted period totals and downloads
as CSV. The stat selector below that table chooses which learning/capacity curves
to inspect. RMSE/MAE use each stat's units; do not combine errors across units.
These are held-out historical forecasts, not new live 2026 projections.

Targets are summed directly from verified canonical weekly records after the
forecast origin through the horizon end. No recorded production is zero only for
completed outcomes; recorded unknown stats stay unknown and those examples are
excluded, with counts by season recorded in metadata. Predictors remain dated
before the forecast cutoff. The training target is the stat per calendar week,
converted back to period totals for evaluation. Count predictions are floored at
zero, remain fractional expected values, and are independently fitted: joint
constraints such as receptions <= targets are not yet enforced. Yardage can be
negative. A prior-production-rate baseline uses only weeks before the cutoff
(the prior season for preseason forecasts). Archived ranking/ensemble comparisons
above the sandbox still concern fantasy points, not individual stats.

Position input filtering excludes player passing stats and their derivatives for
RB/WR/TE, and receiving stats for QB. Rushing stays available to all positions.
Team context, historical fantasy production, availability and background remain.
The exact input audit is displayed with every result. These are candidate sets,
not proven optimal sets; `selected` feature discovery learns a smaller subset
using training data only. Original presets keep their original inputs and points
target. Python defaults remain backward compatible; specify `target` and
`feature_scope` explicitly for stat experiments.

To reproduce the initial 29-model stat suite (2025 remaining-season evaluation):

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m \
  experiments.future_player_lab.notebooks.train_position_stats
```

Its report is `runs/notebook_position_stats_001`; choose a new run ID before
repeating the report. Individual model trials reuse their verified caches.

Opening the notebook never trains models. A verified, previously computed default
example appears automatically when available. Submit **Train tree experiment** to
compute new training-size and capacity curves. Submit **Evaluate custom ensemble**
to recompute weights from earlier out-of-year forecasts. Changing ordinary chart
filters does not launch fitting; changing parameters in a form does not run until
submission. Submitted parameters remain the source of truth until the next submit.

The chosen test year's immediately preceding season is validation. Earlier seasons
train the inner model; validation selects tree count; then a final model refits on
all permitted earlier seasons, including validation. Learning curves add older
complete seasons while holding validation/test fixed. This is a history-length
experiment, not random-row cross-validation. Repeatedly examining test results
makes that year development evidence, so freeze a recipe before future evaluation.

Saved campaign capacity curves use **points per exposure**, matching the stored
training loss. New sandbox curves and final comparisons use **totals in the chosen target's units**.
The final-refit score can differ from the test curve because it includes validation
in its final training set. The notebook labels these differences explicitly.

Original-input experiments preserve original training targets and nonnegative
predictions, but evaluate against corrected canonical labels. Their earlier training
years start in 2004, including years before saved published forecasts begin. Two
replay tests reproduce the actual original boosters on 2025 TE remaining-season
players before adding the sandbox's validation selection.

Tree results live in immutable `runs/notebook_tree_<hash>/` folders. The key binds
settings, prepared data, original release, computation sources and package versions.
Each contains capacity and learning curves, final forecasts, selected inputs,
chronology and a checksum manifest. Identical requests reuse verified results.
CSV downloads are available for forecasts and tables. Custom-blend output exposes
its full configuration, training years and weights plus a forecast download.

Example from Python (with the notebook environment):

```python
from experiments.future_player_lab.notebooks.workbench import DEFAULT_TRIAL, run_trial

trial = run_trial(
    {
        **DEFAULT_TRIAL,
        "position": "RB",
        "horizon": "remaining",
        "engine": "lgb",
        "view": "market",
        "representation": "interactions",
        "max_trees": 360,
        "learning_rate": 0.03,
        "year": 2025,
    }
)
print(trial["path"], trial["metadata"]["selected_trees"])
```

For the original structural settings, use `hist`, `original_profile` or
`original_enriched`, `identity`, 120 rounds, learning rate .05, 15 leaves, depth 0,
leaf support 30, L2 10, feature fraction 1, all history, no recency weighting and
seed 20260923. The sandbox still adds a validation split; its result is a new
experiment, not a replacement for the archived original forecast.

To add model recipes or inputs to the full research campaign, make the change in
`deep/models.py` or `deep/data.py` and create new run IDs. The notebook does not
change completed campaign artifacts, serving eligibility or canonical rankings.

## Validation

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest \
  experiments/future_player_lab/tests/test_notebook.py
experiments/future_player_lab/.notebook-venv/bin/marimo check \
  experiments/future_player_lab/notebooks/boosted_trees.py
```

The tests exercise all four editable engines, future-label invariance, blend
chronology, historical score reproduction, original-booster replay, loss-unit
consistency and detection of changed artifacts.
