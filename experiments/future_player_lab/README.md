# Future player lab

The [interactive marimo boosted-tree notebook](notebooks/README.md) compares the
original boosters, new models and ensembles, with data diagnostics, learning and
overfitting curves, and editable training controls.

Independent experiments for learning future NFL player production from dated histories.
The lab imports no forecasting, feature selection, registry, or promotion code from the
older systems. Its inputs are checksum-verified gold observation tables. Experiment
artifacts are written under this directory's ignored `runs/` tree.

This is a configurable research platform, not a claim that a finite search exhausts
all models or that missing historical information can be recovered by an algorithm.

Start with the [completed deep feature and ensemble results](deep_readout.md), the
[combined-library dashboard](runs/deep_report_001/published_library/index.html),
or the [initial results](initial_readout.md), the
[breadth dashboard](runs/exploration_report_001/index.html), and the
[17-year preseason dashboard](runs/history_report_001/index.html).

The [deep feature and ensemble campaign](deep/README.md) adds three more boosting
engines, 152 new model configurations, the broad verified statistical inventory,
learned combinations and matched comparisons against published forecasts.
Use each run's manifest to distinguish completed work from an ongoing search.

## What is implemented

- QB, RB, WR and TE. Separate positional models and identical evaluation populations.
- Preseason full season; next calendar week; next four calendar weeks; remaining season.
  In-season origins are configurable, including every completed week.
- Five outputs: league points, workload, yards, touchdowns and observed record weeks.
  Models share learned representations across outputs where their observed labels align.
  Unknown labels have separate training masks; they are never imputed to zero.
- Raw NFL production and weekly usage histories, measured college history, and dated
  preseason context. Recent observations, variability, slopes, observation counts,
  career summaries and raw observation sequences compete as input representations.
- Ridge, histogram boosting, extra trees, RBF SVR, approximate kernel regression,
  multilayer neural networks, supervised PLS representations, learned interaction
  bases, neural representations feeding trees, and participation/production models.
- Joint seeded search over representations, model settings, training history length
  and recency weighting. Sparse and binary features remain eligible. No global
  correlation screen, top-80 gate, or registry usefulness decision is inherited.
- Chronological inner validation, then later outer evaluation. Separate policies
  select for point accuracy, ranking quality, and an ensemble of the three best
  earlier candidates. Persistence also competes as a fallback.
- Saved forecasts, annual losses, rookie/returning/market-only cohort diagnostics,
  model choices, input evidence, generated interaction pairs, optimizer diagnostics,
  prior-error intervals and season-cluster comparisons.
- Parallel task execution and resumable per-model/per-year checkpoints. Input,
  implementation, dependency and checkpoint changes invalidate reuse.
- A standalone filterable HTML report and machine-readable Parquet/JSON artifacts.

## Run

From the repository root, using the existing Python environment:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest \
  experiments/future_player_lab/tests -q

# End-to-end plumbing check; not evidence of predictive superiority.
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python \
  -m experiments.future_player_lab.run --run-id smoke_new --smoke

# Initial breadth pass: all positions, horizons and model families, 20 recipes.
# Evaluate 2023–2025, choose on two earlier seasons; in-season origin is week 8.
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python \
  -m experiments.future_player_lab.run --config experiments/future_player_lab/exploration.json \
  --run-id exploration_new

# Main protocol: 2019–2025, three earlier validation seasons, origins 4/8/12.
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python \
  -m experiments.future_player_lab.run --run-id main_new

# Longer preseason test: 2009–2025, eight model families, 16 recipes.
# Complements the all-family/all-horizon breadth pass with wider historical evaluation.
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python \
  -m experiments.future_player_lab.run --config experiments/future_player_lab/history.json \
  --run-id history_new

# Larger, substantially more expensive search: 80 recipes, 2009–2025,
# every in-season origin. This profile is an available experiment, not a completed result.
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python \
  -m experiments.future_player_lab.run --config experiments/future_player_lab/extended.json \
  --run-id extended_new

# Resume with exactly the original code, configuration, environment and input artifacts.
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python \
  -m experiments.future_player_lab.run --config experiments/future_player_lab/exploration.json \
  --run-id exploration_new --resume

# Exact matched-row comparisons against persistence and the fixed boosting control.
.venv/bin/python -m experiments.future_player_lab.tools.compare \
  --source exploration_new --run-id comparison_new

# Regenerate a report in its own versioned directory, preserving the fitted run.
.venv/bin/python -m experiments.future_player_lab.tools.render \
  --source exploration_new --run-id report_new

# Refit the selected pipelines on completed earlier seasons and forecast unlabeled 2026.
# The export identifies itself as a preseason reconstruction issued now.
.venv/bin/python -m experiments.future_player_lab.tools.forecast \
  --source exploration_new --run-id forecast_new --year 2026
```

Run IDs are create-only. Never run two processes against the same run ID. Resume is
for an interrupted run after its previous process has exited. Each completed task
verifies its saved artifact hashes before reuse. A failed candidate is recorded;
it does not silently become another estimator. `complete_with_failures` is distinct
from `complete`, and the report must be interpreted alongside `manifest.json`.

Open `runs/<run-id>/index.html`. Its filters compare like-for-like tasks; the initial
view displays adaptive policies and persistence. Uncheck that filter to inspect
individual candidates, including the fixed histogram-boosting control.

The dashboard scores each model's available predictions and calculates Δ MSE
on identical rows. For workload,
unknown prior target counts can leave persistence predictions missing even where
another model produces a prediction. The separate matched comparison report
also compares against the fixed boosting anchor. Points predictions
are complete. The comparison run preserves the source run and records its hashes.

## Extend the search

Copy a configuration under this directory and give the run a new ID. Increase
`trials_per_family`, change the deterministic `seed`, extend `evaluation_years`,
choose `origins`, or change `history_windows` and `sequence_length`. Model recipes
are saved before training in `candidates.json`; the selected recipe may change each
year using only earlier forecast errors. A new hypothesis or search space requires
a new run, even when old checkpoints would be faster.

`models.py` owns model and representation extensions. `data.py` owns input and label
contracts. `evaluate.py` owns scoring and chronological selection. `run.py` owns
execution/provenance. `report.py` owns the portable report. The protocol explains
the distinction between configurable search choices and necessary validity checks.

## Known boundaries

Historical observations start in 2001; the verified preseason population starts in
2004. In-season evaluation retains that preseason population; it does not yet add
players first appearing later in the season. Preseason roster/context fields stay
at their recorded preseason values while observation history updates. There are
no tracking coordinates, unrecorded news, or inferred medical states. College
identity links and NFL source revisions are retrospective. These are data limits,
not restrictions that changing model families removes.

The lab currently evaluates through completed 2025 outcomes. Its inference export
can produce a 2026 preseason reconstruction without using any 2026 outcome labels.
The export records the actual issuance time; it is not a live in-season forecast
or a forecast that was issued before the season. The lab does not publish
rankings, modify existing experiments, or claim a prospective 2026 validation.
Nothing imports this lab from application code; removing its directory removes it.

Read [protocol.md](protocol.md) before interpreting any apparent improvement.
