# Opening Drive: initial RB, QB, and WR tree campaign

Opening Drive (`opening_drive_v1`) puts six position/horizon packages under one
reusable training, artifact, and API contract. Each package predicts production
components and a separate fantasy-points target. The two horizons are preseason
and remaining regular season after Week 2. This is historical development research;
it has not been promoted into application rankings.

## Protocol

The campaign extends the [RB GPU study](rb_gpu_campaign_2026-10-03.md). Four fixed
recipes preserve the broad raw-derived feature matrix: XGBoost, a shallower and
more regularized XGBoost, XGBoost with a selected provider-metric exclusion mask,
and LightGBM. Three dated draft-history fields are added. A checked 2025 RB slice
matches all 4,810 original feature values at float32 precision, with the same
178 candidates; only the three draft fields are additive.

There are 50 position/horizon/target combinations, 1,600 candidate fold fits, and
100 final saved ensemble members. Training expands from 2002. Candidate forecasts
cover 2018–2025; each 2021–2025 evaluation forecast selects two recipes using only
the preceding three validation seasons. Final 2026 members use 2023–2025 selection
evidence and refit through 2025. Transformations, candidates, and targets have
separate cutoff and completeness checks. No 2026 outcomes enter training.

The primary control is affine-calibrated persistence, fitted on earlier seasons.
All reported paired gains compare the same finite player/year outcomes. This is a
specified control, not a head-to-head test against every legacy model or experts.
Historical seasons have already appeared in earlier research, and provider files
can contain retrospective revisions; these are not pristine prospective results.

QB scoring uses four-point passing touchdowns and minus two for interceptions,
plus rushing production. RB/WR use rushing/receiving PPR. Passing/receiving
exceptions, lost fumbles, and bonuses are excluded as specified in each model's
contract. Cross-position point scales are not directly interchangeable.

## Results

Paired 2021–2025 fantasy-point errors versus affine-calibrated persistence. Positive MSE reduction means lower squared error. These are horizon totals, not weekly errors.

| Position | Horizon | Rows | Model RMSE | Baseline RMSE | MSE reduction | High-prior-usage MSE reduction |
|---|---|---:|---:|---:|---:|---:|
| QB | After Week 2 | 470 | 57.83 | 64.00 | 18.4% | 36.4% |
| QB | Preseason | 458 | 79.13 | 85.88 | 15.1% | -5.0% |
| RB | After Week 2 | 906 | 46.04 | 51.67 | 20.6% | 23.0% |
| RB | Preseason | 865 | 57.37 | 65.54 | 23.4% | 18.1% |
| WR | After Week 2 | 1377 | 41.12 | 47.57 | 25.3% | 19.9% |
| WR | Preseason | 1327 | 50.14 | 57.25 | 23.3% | 9.4% |

48 of the 50 target/horizon combinations improve pooled MSE. The exceptions are WR preseason carries (10.6% worse) and rushing yards (3.9% worse); its receiving forecasts improve. All six point models improve pooled MAE as well as MSE. Ensemble selection is not universally better than the fixed raw anchor or selected single recipe.

RB and WR remaining-season top-24 selections beat affine persistence in each of the five evaluation years. QB remaining-season top-12 selections lose in two years; WR preseason top-24 loses in one. QB preseason top-12 loses in all five. These selection diagnostics and high-prior-usage results make the limitations visible beyond the all-candidate average.

[Full target/year/selection/coherence report](../../experiments/future_player_lab/runs/opening_drive_review_20261009_001/report.md) · [Metrics JSON](../../experiments/future_player_lab/runs/opening_drive_review_20261009_001/report.json) · [Portable target comparison CSV](assets/opening_drive_2026-10-09_metrics.csv)

## What the initial evidence means

The remaining-season results transfer well across positions, including the
high-prior-usage diagnostics. Preseason QB requires more caution: a lower pooled
production error does not establish better selection of starting quarterbacks.
Every tested QB preseason recipe, the single selected policy, and the ensemble
trailed affine persistence in top-12 realized points in every evaluated year.
Every tree variant also had higher point MSE in the high-prior-usage subset.

One concrete failure explains the entire 2021 top-12 deficit: the ensemble selected
Drew Brees instead of Tom Brady, whose recorded outcome was 380.74 points under
this scoring contract. Brees had zero 2021 production. His retirement was publicly
known before the forecast cutoff, but the candidate/history feature contract does
not include that dated offseason event. The Saints date the announcement to
March 14, 2021 in their [retirement account](https://www.neworleanssaints.com/news/drew-brees-knew-time-was-right-to-retire-as-a-new-orleans-saint).

This points to a source-coverage problem as well as model selection. A future
availability model or candidate rule should consume genuinely as-of event evidence
and be evaluated chronologically. Retrospectively removing this player or adding
a named override would invalidate the comparison. The frozen initial forecasts
retain the error.

A separate follow-up tested shrinking the fixed raw XGBoost anchor toward affine
persistence. Mixing weights came from a fixed five-value grid, selected using
only the preceding three validation seasons, either globally or separately for
high-prior-usage players. No new trees were fitted. This question was prompted by
the initial results, so it is explicitly additional development analysis.

Neither variant justified replacing the frozen forecast. Global shrinkage reduced
the high-usage MSE deficit from 5.02% to 1.18%, but pooled MSE rose from 6,261 to
6,299; cohort shrinkage had pooled MSE 6,334 and retained a 3.08% high-usage deficit.
The original six packages remain unchanged. The missing dated availability evidence
is a more concrete research direction than assuming a simple blend solves it.

Independent component models can violate count relationships. The QB remaining
ensemble forecasts completions above attempts on 10 of 470 evaluation rows and
interceptions above attempts on one row; these are low-volume cases, but genuine
inconsistencies. Direct points also differ from the score computed from component
predictions. Both are retained and labeled rather than silently reconciled.

The `xgb_no_provider_draft` ID denotes a fixed selected-metric ablation, not removal
of every provider-derived feature. Nine QB `passing_cpoe` fields survive its
inherited exact-name rule. This behavior is frozen and disclosed; changing the
mask would be another experiment.

## Reuse and review

See the [campaign workflow](../operations/model-campaigns.md) and
[training README](../../experiments/future_player_lab/campaigns/README.md).
The campaign API exposes explicit model IDs, metric evidence, forecast cutoffs,
source receipts, and per-player predictions. Its request path reads verified JSON.
Offline replay loads verified saved weights and reproduces the forecasts using
feature matrices copied into the published bundle.

Research artifacts remain local and outside Git. The independent audit recomputes
selection-year and evaluation-year losses from all candidate checkpoint arrays,
checks row/outcome alignment, reconstructs selections and ensemble arithmetic,
and verifies source, dataset, and artifact checksums. Tests exercise cutoff
isolation, missing outcomes, scoring, invalid selection history, corrupt artifacts,
and replay after original run files are removed.

Promotion should assess each position, horizon, and target separately, with
particular attention to preseason QB selection and dated availability coverage.
Frontend integration and legacy removal remain separate work.

## Completion record

All six packages passed the independent historical audit, including all 1,600 candidate checkpoints and selection warmup years. The local published bundle contains 96 verified artifacts. Standalone replay reproduced all 996 player/horizon forecasts with maximum absolute prediction difference 0.0: QB 84 remaining / 81 preseason, WR 269 / 241, and RB 170 / 151.

The targeted regression suite passed 97 tests; lint and whitespace checks passed. Live HTTP checks returned 200 for campaign discovery, model metadata, evaluation metrics, rankings, and an exact-ID Kenneth Walker forecast. API metadata identifies these as reconstructed cutoffs and research previews.

[Artifact provenance](assets/opening_drive_2026-10-09_provenance.json) pins the
published bundle, all six model manifests, the complete audit, exact forecast
replay, and the separately audited shrinkage diagnostic. Live model cards were
also compared against the independent report; all six matched. Unsupported origins
return 404, and a mismatched campaign-manifest pin returns 409.
