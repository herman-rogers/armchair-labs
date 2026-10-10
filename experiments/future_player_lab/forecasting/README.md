# Saved-model forecasts

Reuse completed research models without refitting them or changing application
rankings. The first registered model, `rb_ppr_v1`, forecasts RB remaining-season
rushing/receiving PPR after Week 2. WR and QB models require their own trained
artifacts and registry entries; this command never applies RB weights to them.

[2026 Week 2 preview](../runs/rb_ppr_2026_w2_preview_002/index.html) ·
[Rankings CSV](../runs/rb_ppr_2026_w2_preview_002/rankings.csv) ·
[Expert comparison](../runs/rb_ppr_2026_w2_experts_001/index.html)

The preview is a **reconstruction using observations through Week 2**, generated
later. It is not an originally issued Week 2 forecast or an October 9 injury-adjusted
ranking. Its 170 candidates include prior-year participants and exactly linked
drafted entrants, including players without a current-season appearance. Teams
follow the last box-score observation, as in training. No 2026 outcomes select or
fit a model. Week 4 adaptation and validation are deferred.

## Layout and contracts

| Component | Responsibility |
| --- | --- |
| `registry/*.json` | Model position, horizon, supported cutoffs, scoring, policy, and pinned artifact hash |
| `capture.py` | Save optional raw provider sources once, with checksums and acquisition provenance |
| `data.py` | Reproduce training aggregations using only observations at/before the cutoff |
| `run.py` | Load saved policies, score, rank, and write an immutable forecast run |
| `report.py` | Searchable standalone HTML report and links to downloadable data |
| `replay.py` | Verify run hashes and reproduce all three saved-policy predictions offline |
| `compare.py` | Add dated expert ranks and agreement metrics without rescoring or changing the forecast |

The model registry points at the original `final_models` directory. Its existing
loader verifies model and preprocessing artifacts. Feature alignment is by name,
not caller column order. Missing **values** remain unknown; missing feature
definitions fail. Unsupported origins fail instead of silently applying a Week 2
model at Week 4. No library search or estimator fit occurs in the forecast path.

The feature builder checks all 4,810 input fields against the original sealed 2021
and 2025 matrices before producing a new forecast. Model scoring also checks that
reversing input-column order leaves predictions unchanged. Dependency versions
must match the saved-model contract. Historical caches verify both the loader and
source snapshot hashes.

Replay verifies the archived implementation copies and the current inference
implementation. Changes to report rendering, capture orchestration, or the replay
wrapper itself do not invalidate an otherwise identical reproduced prediction.

## Run

Use the [GPU research environment](../../../docs/operations/gpu_research_setup.md).
The original model library and canonical raw snapshot must be restored locally;
generated runs are ignored by Git. All paths below are relative to the repository.
Limit numerical-library threads when invoking these commands:

```sh
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 POLARS_MAX_THREADS=4
```

Capture supplemental raw sources into a **new directory**. The explicit NGS
snapshot supplies the retained 2026 records; the provider loader returned an empty
2026 NGS table during this run. That source choice is recorded rather than silently
replacing it with another dataset.

```sh
.venv/bin/python -m experiments.future_player_lab.forecasting.capture \
  --season 2026 \
  --ngs-snapshot data/raw/snapshots/tracking_nflverse_ngs_20261002_r2/manifest.json \
  --output experiments/future_player_lab/runs/forecast_sources_NEW
```

For an offline rerun of the completed preview, use the retained
`forecast_sources_20261009_003` capture instead. Choose a new forecast run ID:

```sh
.venv/bin/python -m experiments.future_player_lab.forecasting.run \
  --model rb_ppr_v1 \
  --snapshot data/raw/snapshots/canonical_20260923_r4/manifest.json \
  --supplements experiments/future_player_lab/runs/forecast_sources_20261009_003 \
  --season 2026 --origin 2 \
  --run-id rb_ppr_2026_w2_NEW
```

The reusable historical observation cache is created automatically on first use.
Use `--history-cache PATH` for a separate cache when the historical source contract
changes. A run directory is create-only: a failed or completed run is never
overwritten. Completion is established by `manifest.json`, not directory existence.

```sh
.venv/bin/python -m experiments.future_player_lab.forecasting.replay \
  experiments/future_player_lab/runs/rb_ppr_2026_w2_preview_002
```

Each run retains the input feature rows, feature coverage, CSV/JSON/Parquet rankings,
HTML report, model registry entry, source-code copies, package versions, source
references and hashes, and a manifest covering all output files. Replay requires
the original saved models but does not contact providers or rebuild features.

## What the ranking means

The primary score is the original equal-weight top-three policy, predicting PPR
for Weeks 3–18. It uses 16 remaining **calendar weeks** for exposure, matching the
training contract. The report also shows the original single-model and
different-family policies. Their disagreement is sensitivity, not a confidence
interval. PPR per scheduled game divides the total by remaining team games; it is
not a conditional forecast assuming the player is healthy or active in every game.

Inputs come from raw box scores, offensive snaps, play-by-play, NGS, weekly rosters,
and official team statistics, plus the original historical observations. Current
injuries without `date_modified` and timestamp-grain depth charts are withheld
because they do not satisfy the original feature contracts. Current FTN
participation is withheld under the existing publication-delay rule. The source
audit lists exact exclusions, and nulls retain the trained missing-value behavior.

The historical research RMSE is prior development evidence, not a measurement of
this 2026 ranking's accuracy. Source captures can include retrospective revisions,
and missing or unlinked newcomers remain a coverage limitation. Nothing here
updates the application's production forecasts.

## Compare with experts

Build a separate comparison report from the completed forecast. The original run,
predictions, and manifest stay unchanged; no model loads or refits are needed.

```sh
.venv/bin/python -m experiments.future_player_lab.forecasting.compare \
  --forecast experiments/future_player_lab/runs/rb_ppr_2026_w2_preview_002 \
  --experts data/research/rb_wr_ecr_20261002/ros-ppr-rb.json \
  --output experiments/future_player_lab/runs/rb_ppr_2026_w2_experts_NEW
```

The expert JSON must be a FantasyPros positional ROS PPR export for the same
season. The retained October 2 source includes seven experts. The report prominently
labels its different information cutoff from the model's Week 2 inputs: these
metrics measure agreement, not forecast accuracy or same-cutoff superiority.

Joins require unique normalized player names within position. Suffixes/punctuation
are normalized; ambiguous names remain unmatched. Team changes are flagged.
Unlisted players retain null expert ranks. The table displays original list ranks
and `expert rank - model rank` (positive means the model ranks the player higher).
Correlation, mean gap and the scatter plot rerank only the common player pool,
using average ranks for ties. Top-12/24 overlap uses the original full lists, so
unmatched players are not silently removed from that comparison.

The comparison run retains the expert snapshot, source hashes, enriched rankings,
metrics and a standalone report with player search, list filters and sorting.

## Add a model later

Train and validate the new position/horizon separately, retain its preprocessing
and feature names, and export a compatible completed model library. Register its
position, scoring, allowed origins and manifest hash under a new ID. Then extend
the scoring/report adapter if its target semantics differ (QB fantasy scoring
needs passing/interception components). Do not edit the RB entry to point at
unrelated weights or claim Week 4 validity by changing its allowed origins.

```sh
.venv/bin/python -m pytest experiments/future_player_lab/tests/test_saved_forecasting.py
```
