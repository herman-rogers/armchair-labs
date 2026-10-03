# Research workflows

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

## Archived metric and board experiments

These commands reproduce the earlier V2 system; they do not publish NextGen:

```bash
just metric-report
just board
just feature-discovery
```

The metric report writes `data/outputs/metric_report.json`, retained player/fold
predictions in Parquet, and a Markdown summary. It includes per-position fitted
ranker weights for the archived V2 board. A stored model must match the fit
configuration, season, and depth-chart snapshot, so rebuild the metric report before
rebuilding that board. `just metric-report --reanalyze` re-scores retained folds.
See the [archived metric report reference](../reference/metric_report.md).

Feature discovery consumes retained metric folds and richer source-aware weekly
features, then evaluates challengers with nested walk-forward validation. It writes
`feature_discovery_report.{json,md}` and `feature_discovery_predictions.parquet`.
Use `--force-rich` to rebuild its rich panel and descriptors. This workflow is
report-only: it does not change production metric configuration or the frozen
2026 forecast.
