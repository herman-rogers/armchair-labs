# V2 metric report and rolling backtest

The metric report answers two different questions for each configured v2 field:

1. Does it rank next-season outcomes in a useful direction?
2. Does it retain signal after controlling for the model's historical PPG prior?

Run it with:

```bash
just metric-report
# or: uv run patron metric-report
```

The command writes three gitignored artifacts under `data/outputs/`:

- `metric_report.json`: API and frontend report;
- `metric_report.md`: compact human-readable summary;
- `metric_backtest_predictions.parquet`: every player/forecast fold for deeper analysis.

## Forecast construction

The seasons and metrics are declared in
[`src/patron/config/metric_report.yaml`](../src/patron/config/metric_report.yaml). With
the default configuration, each forecast uses exactly three prior seasons:

| Forecast seasons | Input seasons | Status |
|---|---|---|
| 2004–2025 | Rolling prior three seasons beginning in 2001 | scored against actuals |
| 2026 | 2023–2025 | retained as a pending prospective forecast |

Results are calculated separately for the long-horizon 2004–2025 window, the
route-enriched modern 2019–2025 window, the recent 2020–2025 window, and the
FantasyPros archive's fully covered 2021–2025 market window. Injury metrics begin
with the 2012 forecast, route/participation metrics with 2019, ffopportunity xFP with
2007, and NGS with 2017. Unavailable history is never treated as observed zero
evidence.

Current manual player/team overrides are disabled. Timestamped depth charts are cut
off on August 31 of each forecast year. For 2004–2024, the legacy nflverse feed has no
publication timestamp, so its published Week 1 rank is explicitly treated as an
August 31 proxy. The report labels this approximation rather than presenting it as a
true timestamped preseason observation.

Participation files from 2016–2022 omit position arrays. The usage builder maps GSIS
player IDs to the matching player-stat position and fails the report if a requested
participation season still produces no route opportunities. V2 PPG and games-played
outcomes use participation-observed active games when available; the original
production-row `games` field remains in the data for compatibility.

The forecast population is returning players with at least one game in the source
season. Rookies remain outside v2 until a separate rookie model exists.

## Does v2 beat the naive baseline?

The report's first table is a head-to-head ranking test, configured in the `ranking`
block of `metric_report.yaml`. Each baseline (`historical_ppg_prior`, last-season
`ppg`, last-season points, and dated positional FantasyPros ECR when fully covered)
and candidate (`v2_score`, `proj_ppg`, …) is scored on the same top-K per
position — the starters a ten-team league drafts — with hit rate @K, NDCG @K, and
Spearman inside a draftable pool, per fold and per window. Candidates carry their lift
over the best baseline and the number of folds they won; "beats" requires positive
pooled lift and more folds won than lost. The JSON exposes this as
`ranking_results`. A partial-history baseline cannot win by scoring only its favorable
folds: it competes as "best baseline" only in windows where it covers every candidate
fold. See `docs/v2_metrics_review.md` §0 for why this is the goal.

## Learned weights (walk-forward fitted ranker)

The `fit.models` block of `metric_report.yaml` declares named model specs (ridge on a
target and features, or a product of earlier outputs). For every forecast season each
ridge model is fitted per position on all completed folds strictly before it, with the
ridge strength chosen by nested walk-forward validation (the pending season uses every
completed fold), and every output competes in the ranking table like any other
candidate. The OVERALL rows score one top-K across all positions, each ranker expressed
as per-game VOR against its own positional replacement — the test the live board's
cross-position order must pass. The JSON carries every refit in `fitted_models` and the latest weights
with their spread across refits in `fitted_model_summary`; the Markdown shows the same
as a "Learned weights" table. A coefficient whose spread approaches its magnitude is a
signal to drop the feature, not evidence. See `docs/v2_metrics_review.md` §8.

The report also fits separate report-only challengers for expected opportunity, QB
process, situation-neutral team tendency, and NGS residuals before fitting a smaller
survivor-only `fitted_research_ppg`; its season-points variant multiplies the same
calibrated games model as the production fit. These outputs are evaluated and
serialized but `apply_live: false` prevents an experimental feature set from silently
changing the live board.

## Reading the evidence

- **Spearman** is rank correlation with the next-season outcome.
- **Partial vs prior** is partial Spearman correlation after controlling for
  `historical_ppg_prior`. It helps identify whether a field adds information instead
  of restating recent production.
- **Consistency** is the share of completed forecast seasons whose correlation agrees
  with the pooled direction.
- **Coverage** is the share of eligible historical rows containing the metric.
- **Fold range** is the empirical 2.5th–97.5th percentile range of season-level
  correlations. This avoids pretending repeated rows for the same player are
  independent observations.

Assessments are descriptive, not claims of causation:

- `strong`: substantial incremental rank signal with consistent yearly direction;
- `useful`: smaller but repeatable incremental signal;
- `harmful`: repeatable signal in the opposite direction from the metric's configured
  expectation;
- `redundant`: raw association largely disappears after controlling for the prior;
- `weak`: little raw or incremental association;
- `mixed`: magnitude or yearly direction is unstable;
- `insufficient`: too few samples or completed folds.

Direct projection fields also receive MAE, RMSE, bias, and rank-correlation results in
the JSON `model_results` collection. These distinguish ranking quality from calibration.

## Extending the report

Add a metric entry to `metric_report.yaml` with its board column, label, family,
description, applicable outcomes, and optional positions. No analysis or frontend code
change is required. Set `prediction_target` when the metric is on the same scale as an
outcome and should receive direct error metrics. Set `expected_sign` to `negative` for
metrics such as injury burden or depth-chart rank, or `either` when direction is not a
quality judgment.

Changing `forecast_seasons` or `history_seasons` expands the rolling folds. Source data
is cached, so ordinary reruns avoid downloading unchanged nflverse partitions.
