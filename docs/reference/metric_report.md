# V2 metric report and rolling backtest

> Archived system reference. This describes the earlier metric/board system, not the current NextGen serving path.

The metric report answers two different questions for each configured v2 field:

1. Does it rank next-season outcomes in a useful direction?
2. Does it retain signal after controlling for the model's historical PPG prior?

Run it with:

```bash
just metric-report
# or: uv run engine metric-report
```

The command writes three gitignored artifacts under `data/outputs/`:

- `metric_report.json`: API and frontend report;
- `metric_report.md`: compact human-readable summary;
- `metric_backtest_predictions.parquet`: every player/forecast fold for deeper analysis.

Automated feature discovery is intentionally separate from this production-facing
report. After generating the retained folds, run:

```bash
just feature-discovery
```

That experiment fits broad nonlinear models, generic weekly time-series descriptors,
training-only PCA/K-means archetypes, deterministic random convolutions, symbolic
residual formulas, and out-of-fold stacks. Its rich weekly layer adds source-aware
snap share, route participation, targets per route, expected opportunity,
injury/practice and roster status, team volume, red-zone work, and EPA rates. Feed
availability is explicit: for example, route descriptors are null before the source
begins rather than being backfilled as zero. Run `engine feature-discovery
--force-rich` when the underlying weekly data changes. Every transformation and
feature search is refitted inside its expanding historical window. Its artifacts use the
`feature_discovery_*` prefix and cannot be loaded by the live board.

## Forecast construction

The seasons and metrics are declared in
[`src/engine/config/metric_report.yaml`](../../src/engine/config/metric_report.yaml). With
the default configuration, each forecast uses exactly three prior seasons:

| Forecast seasons | Input seasons | Status |
|---|---|---|
| 2004–2025 | Rolling prior three seasons beginning in 2001 | scored against actuals |
| 2026 | 2023–2025 | retained as a pending prospective forecast |

Results are calculated separately for the long-horizon 2004–2025 window, the
route-enriched modern 2019–2025 window, the recent 2020–2025 window, the
FantasyPros archive's fully covered 2021–2025 market window, and the extended
2011–2025 market window. The extension comes from the dated market backfill
(2026-08-31): 2020 is recovered from the archive's own summer combined-offense
pages, and 2011–2019 from timestamped Wayback captures of the same FantasyPros
cheatsheet pages, committed with per-row capture URLs in
`data/static/market_ecr_backfill.csv`. Observed preseason ADP (MFL real-league
drafts 2011+, Fantasy Football Calculator 2008–2010) is archived separately in
`data/static/market_adp_backfill.csv`. Collection policy, derivations, and the
honest gaps (no 2010 ECR — captures too shallow; no 2007 anything) are recorded
in `data/static/market_backfill_manifest.md`. The 2021–2025 window is kept so
new reports stay comparable with previously documented numbers; specs tuned
before the backfill see 2011–2020 as genuinely held-out market folds exactly
once. Injury metrics begin
with the 2012 forecast, route/participation metrics with 2019, ffopportunity xFP with
2007, and NGS with 2017. Unavailable history is never treated as observed zero
evidence.

Current manual player/team overrides are disabled. Market folds cut timestamped depth
charts at the archived ECR snapshot date; folds without a market snapshot use the
configured August 31 boundary. For 2004–2024, the legacy nflverse feed has no
publication timestamp, so its published Week 1 rank is explicitly treated as an
August 31 proxy and is unavailable to an earlier ECR fold. The report labels this
approximation rather than presenting it as a true timestamped preseason observation.

Roster availability is separate from that depth-chart proxy. The canonical
`preseason_*` fields start from the player's prior-season team and apply dated official
club transactions only through the fold's information cutoff for 31 market-era teams;
Dallas and older folds retain a labelled ESPN fallback. The original nflverse Week 1
roster/status fields remain as `week1_proxy_*` sensitivity columns and are not inputs
to cutoff-safe candidates. `data_summary.cutoff_availability` reports transaction
match rate, proxy coverage, and membership agreement. The reconstruction is
conservative: an unmatched retirement or unsigned player can remain provisionally
active, which is disclosed as a report limitation.

Participation files from 2016–2022 omit position arrays. The usage builder maps GSIS
player IDs to the matching player-stat position and fails the report if a requested
participation season still produces no route opportunities. V2 PPG and games-played
outcomes use participation-observed active games when available; the original
production-row `games` field remains in the data for compatibility.

The legacy v2 forecast population is returning players with at least one game in the
source season. The next-generation challenger adds a separate rookie population from
nflverse identity and NFL draft capital; rookies are never mean-imputed into the
returner model. Age, college-conference context, and combine data remain reportable
hypotheses but were removed from the challenger after they ranked rookies worse than
draft capital alone. Public college production is not available through the current
nflverse boundary, so the report names that missing input rather than fabricating it.
`data_summary.population_counts` audits both pools.

## Does v2 beat the naive baseline?

The report's first table is a head-to-head ranking test, configured in the `ranking`
block of `metric_report.yaml`. Each baseline (`historical_ppg_prior`, last-season
`ppg`, last-season points, and dated positional FantasyPros ECR when fully covered)
and candidate (`fitted_ppg`, `proj_ppg`, …) is scored on the same top-K per
position — the starters a ten-team league drafts — with hit rate @K, NDCG @K, and
Spearman inside a draftable pool, per fold and per window. Candidates carry their lift
over the best baseline and the number of folds they won; "beats" requires positive
pooled lift and more folds won than lost. Every candidate is also compared head-to-head with the configured
`ranking.market_baseline` (FantasyPros ECR) on the folds both scored, as the
`market_*` fields. The JSON exposes this as `ranking_results`. A partial-history baseline cannot win by scoring only its favorable
folds: it competes as "best baseline" only in windows where it covers every candidate
fold. See `docs/reference/v2_metrics_review.md` §0 for why this is the goal.

The separate `market_disagreement_results` table is now the primary Moneyball test.
It restricts each model and dated FantasyPros overall-ECR price to identical players
at the same historical information cutoff. A model-top-60 / ECR-outside-60 call is a
hit only when the player actually finishes top 60; merely improving from ECR rank 300
to actual rank 250 no longer counts. The promotion fields are contrarian precision,
the share of ECR's actual top-60 misses recovered, realized VOR of model-only minus
ECR-only swaps, and false-positive cost. Gap bands of 20, 60, and 120 slots expose the
deep bargains separately. Ordinary hit rate, NDCG, and top-60 value remain guardrails.
Overall ECR is labelled as an expert-consensus price proxy, never as observed ADP.
Genuine ESPN ADP panels are archived once per UTC date under
`data/outputs/market_snapshots/` for prospective ADP evaluation; there is no honest
historical backfill for seasons never captured.

## Learned weights (walk-forward fitted ranker)

The `fit.models` block of `metric_report.yaml` declares named model specs (ridge on a
target and features, a product of earlier outputs, a coalesce across mutually exclusive
rookie/returner outputs, or a report-only adaptive selector among earlier outputs).
For every forecast season each
ridge model is fitted per position on all completed folds strictly before it, with the
ridge strength chosen by nested walk-forward validation (the pending season uses every
completed fold), and every output competes in the ranking table like any other
candidate. The OVERALL rows score one top-K across all positions, each ranker expressed
as per-game VOR against its own positional replacement — the test the live board's
cross-position order must pass. The JSON carries every refit in `fitted_models` and the latest weights
with their spread across refits in `fitted_model_summary`; the Markdown shows the same
as a "Learned weights" table. A coefficient whose spread approaches its magnitude is a
signal to drop the feature, not evidence. See `docs/reference/v2_metrics_review.md` §8.

The report also fits separate report-only challengers for market consensus, roster
membership/status, snaps, vacated opportunity, draft/contract commitment, and their
combined additive stack. Binary-rostered and status-independent commitment controls
make the timing and selection effects directly testable. Their season-points variants
multiply fitted PPG and games components. These outputs are evaluated and serialized,
but `apply_live: false` prevents an experimental feature set from silently changing
the live board. Canonical roster inputs are cutoff-safe transaction reconstructions;
historical weekly roster rows are explicitly separate Week 1 proxies.

For market-era folds, the archived ECR snapshot date—not a blanket August 31—is the
input cutoff for transactions and depth charts. Official NFL club transaction logs
replace the ESPN archive for 31 teams in 2020–2026 because the ESPN history contains
date errors around roster cutdown; Dallas is the explicit fallback. Transaction prose
is retained as separate IR, PUP/NFI, suspension, other-reserve, practice-squad, and
off-roster features. The public archive still lacks complete dated soft-injury and
recovery/practice state, so `fitted_market_availability_*` is a separately labelled
same-snapshot ECR-conditioned sensitivity model, not independent injury alpha.

The `fitted_nextgen_*` family is the lean challenger: returner PPG is estimated only
from quality/role inputs, returner games uses one copy of `expected_games` plus cutoff
status, and season points is their product. A distinct rookie PPG × games product is
calibrated from NFL draft capital plus an undrafted indicator, then coalesced with it.
Every member is `apply_live: false`; none is a factor of the frozen 2026 adaptive
selectors.

The hidden-pattern loop adds two anti-overfitting diagnostics. The JSON
`residual_pattern_results` screens only residuals left after the existing fitted
outputs and requires a direction to survive discovery (2007–14), confirmation
(2015–18), and modern (2019–25) eras. `ranking_sensitivity_results` repeats the entire
ranking comparison for both cutoff-rostered and Week-1-proxy-rostered populations.
The `adaptive_select` models use expanding-window top-K results: a forecast-season
choice can inspect only earlier completed folds, needs at least three, and is reported
with its selected source and choice history in `fitted_model_summary`. Adaptive models
remain report-only; the live scorer rejects an adaptive spec marked `apply_live: true`.
The 2026 selector definitions and prediction digest are locked in
`src/engine/config/experimental_freeze_2026.yaml`; a test rejects accidental changes
before that prospective outcome is graded. The complete 611-row snapshot lives at
`data/static/experimental_2026_predictions.csv`, rather than in the replaceable report
output. After the regular season and source data are complete, run:

```bash
uv run engine grade-prospective
```

The command verifies the snapshot digest, refuses to run before January 15, 2027 or
against partial outcomes, and joins only outcome columns onto the frozen rankings. It
reports—but never automatically applies—three primary promotion gates: overall hit
rate above the incumbent, overall hit rate above dated ECR, and NDCG at least equal to
ECR.

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
