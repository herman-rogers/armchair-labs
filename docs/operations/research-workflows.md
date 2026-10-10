# Research workflows

## Raw running back GPU campaign

The [GPU setup guide](gpu_research_setup.md) installs optional NVIDIA cuML,
XGBoost CUDA and CatBoost GPU backends and documents measured hardware behavior.
The [RB campaign guide](../../experiments/future_player_lab/gpu_lab/README.md)
describes raw-input chronological searches, unsupervised representations,
full-data profiles, conditional profile forecasts, and the separate handoff study.
The [October 3 findings](../research/rb_gpu_campaign_2026-10-03.md) summarize measured
accuracy against stronger baselines, representation tradeoffs, and limitations.
These experiments retain fitted models and verified checkpoints under ignored
`experiments/future_player_lab/runs/`; they do not publish application forecasts.

## Local notebooks

```bash
bash experiments/future_player_lab/notebooks/launch.sh
```

Follow the [notebook guide](../../experiments/future_player_lab/notebooks/README.md)
for model-specific inputs, launches, archive handling, and notebook controls. Research
runs need generated datasets that are not included in a fresh clone. Proposed shared
storage and GPU work are tracked in the [compute plan](../plans/research_compute_and_shared_data_plan_2026-10-02.md).

Experiments and protocols remain beside their code in `experiments/` and `research/`.
The [research index](../research/README.md) collects dated findings. Running a model
or retaining a dataset does not change the published serving policy by itself.

## Retired metric and board experiments

The V1/V2/Adaptive boards, the rolling metric report and feature discovery were
removed in October 2026 together with the hand-coded feature matrix. Their findings
remain in [docs/reference](../reference/README.md) and [docs/history](../history/README.md);
the code is in git history only. Observation tables now come from the DuckDB
recipes in `src/engine/data/enrich/sql/`.
