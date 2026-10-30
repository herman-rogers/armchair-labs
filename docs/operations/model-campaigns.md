# Opening Drive campaign

The application serves every sealed bundle in `data/campaigns/`; there is no
allowlist, so publishing a new campaign directory makes it available everywhere.
`/api/chairstats/campaigns` lists them default-first, then newest, and declares
`default_campaign_id`. The default is **Opening Drive v4.1** (`opening_drive_v41`),
set by `DEFAULT_CAMPAIGN` in `src/engine/chairstats/application.py`; if that bundle
is missing, the newest campaign is used. Scoring, freshness and player-forecast
requests without a campaign use the default. Earlier bundles (v4, v3.2 preview, v3
and v2) stay unchanged and selectable. Page pickers show the default plus the newest
few campaigns and link to `/research/models`, a searchable, paged list of every
campaign.

## Version lineage and retained functionality

**Opening Drive versions must build on and improve their declared base version.
V3.2 means v3 plus the injury, participation and workload changes. It must retain
v3's capabilities; it must not silently return to v2 with a new adjustment layer.**
This requirement was explicitly confirmed by the user on 2026-10-14. V3.1 is a
paused research branch, so v3—not v3.1—is the agreed base for v3.2.

Every release must:

- Name its base version and trace the models supplying each served output. Older
  versions may be labeled controls or inherited input features. Their presence as
  an existing feature does not permit replacing the newer forecasting method.
- Preserve the base's point forecasts, weekly prediction/distribution capability,
  variance and interval outputs, supported horizons, component-stat outputs,
  averages, rankings, lineup behavior, API fields and refresh/replay functionality.
  Changes to estimates are expected; silently dropping a capability is not.
- Extend the newer forecasting approach across weekly, remaining-season and
  projected full-season views. Making totals add up is necessary, but using old
  model totals underneath a new weekly adjustment does not satisfy version lineage.
- Compare against the declared base on matched data, cutoffs, scoring and cohorts.
  Evaluate point accuracy, ranking quality and uncertainty separately, with a
  functionality comparison. A version number or improved interval coverage alone
  does not establish better point predictions or rankings. Disclose tradeoffs.
- Keep unsupported extensions as research. Any deliberate rollback, loss of
  functionality or change of base must be explicitly described and agreed with
  the user; it must never be hidden behind the newer version name. Existing
  authorization still applies to changes already explicitly agreed.

**Current implementation gap:** v3.2's weekly injury/workload models build on v3,
but the 2026-10-14 season projection bridge returns toward v2 remaining-season
estimates. That bridge does **not** satisfy the intended v3-based extension. Its
new horizon forecasts also expose points only, removing component-stat outputs
from those current views. Archiving the old outputs does not preserve their live
functionality. The bridge is a documented interim implementation, not the pattern
to copy for future releases. V3's own original season forecasts were inherited
from v2; that historical fact does not fulfill the requested newer-model extension.

The publication gate verifies arithmetic and matching cutoffs. V4 additionally
requires explicit v3.2 lineage, a v3-based anchor, live components, component
coherence and finite mean-preserving distributions. It does **not** establish
accuracy improvement. Passing it must not be described as completing evaluation.
The current
v3.2 preview also withholds K/DST uncertainty and mixed-timing lineup ranges;
these limitations remain visible in the functionality comparison.

## Current horizon implementation

The [v4 candidate](../../experiments/future_player_lab/campaigns/weekly_v4/README.md),
implemented on 2026-10-27, replaces the v2 horizon anchor with a fitted v3-based
extension and restores live components. It is evaluated at origins 1–16 and tracked
prospectively against frozen v3.2 controls. Historical bridge comparisons are mixed,
so it did not become the default. Select it with `?campaign=opening_drive_v4`.

V4.1 (`opening_drive_v41`, added 2026-10-27, default since the same day) is a full retrain in one
research run (`runs/opening_drive_v41_20261027_001`): v3 weekly method plus market
lines, the v3.2 injury/workload layer and the v4 horizon/component method, all
fitted through 2025. `weekly_v41/track.py` runs `weekly_v41/live.py` on the latest
accepted snapshot and publishes all 12 models (`weekly_v41/publisher.py`); each new
origin is added to the bundle and a repeated origin replaces its rows. Pregame
updates (`--pregame`) recompute players in unstarted games whose teams have admitted
official report captures and rebuild weekly, remaining and season together. Against
v3.2 it improves early-week QB weekly RMSE by 4.1% and WR by 0.5% (relevant cohort,
2021–25; RB/TE within noise), and remaining-season QB/TE totals against v4. Release
checks only warn.

```sh
.venv/bin/python -m experiments.future_player_lab.campaigns.weekly_v41.track [--force] [--pregame]
```

As of 2026-10-14, v3.2's current season/remaining forecasts are derived from one
future-week curve that includes its saved weekly prediction. Remaining points sum
the curve; projected full-season points add accepted actual points through the
cutoff. Per-game averages and rankings use these new totals. The original preseason
forecast remains at origin zero for draft comparisons. Expand a Rankings row to
see the future-week estimates. Longer horizons are points-only; the prior bundle
preserves its inherited component-stat forecasts.

Future Opening Drive publication and replacement enforce this shared horizon
contract in `engine.chairstats.horizons`: all positions and horizons must share the
current cutoff, weekly capture and eligible population; totals and averages must
reconcile. A weekly-only model update cannot silently retain old season projections.
The [projection workflow](../../experiments/future_player_lab/campaigns/horizon_projection/README.md)
documents evaluation, injury handling and limitations. Historical remaining-season
accuracy is mixed; coherent views do not imply uniform improvement in accuracy.

Dashboard selection does not change the sealed bundle’s research-preview designation. Its outcome ranges have imperfect historical coverage;
they are not confidence intervals on expected ability or guaranteed 80% intervals.
See the [weekly research protocol and reproduction commands](../../experiments/future_player_lab/campaigns/weekly_v3/README.md)
and the locally generated comparison at
`data/outputs/opening_drive_v3_comparison_2026/index.html`.

TE's `te_season` and `te_remaining` models are native members of v2. Their fitted
weights, evaluation evidence and forecast values are preserved from the TE research
extension. Their original training identity remains in model provenance; application
forecast receipts identify `opening_drive_v2`.

## Archives

```sh
just campaigns list
just campaigns check opening_drive_v2
```

The one-time v2/TE consolidation has been run and its code removed: v2 holds the
TE models, and `opening_drive_v1`, `scout_team_v1` and `opening_drive_v2_te` are
archived outside the application registry at
`data/campaign_revisions/<id>/<manifest_sha256>/`. Refresh no longer runs a
migration step.

Only the campaigns in `APPLICATION_CAMPAIGNS` (`src/engine/chairstats/application.py`:
`opening_drive_v32`, `opening_drive_v4`, `opening_drive_v41`, `opening_drive_v3` and
`opening_drive_v2`) are exposed by application queries,
even if a research command later writes an older campaign into the registry. Retired campaign endpoints return
404. Historical research can still be inspected with `CampaignStore` using its
archive directory; it is not offered as an application model choice.

## Overview lineup projections

The overview uses one full-lineup estimate for its ranks, projected points and
current-week matchup cards: the sum of selected starters, with v3 weekly points
(or v2 remaining-season points per scheduled game) for QB/RB/WR/TE and captured ESPN
weekly projections for K/DST. The original campaign remains an offensive model;
K/DST are added by the league forecast calculation, not by changing its weights.

Missing ESPN K/DST projections fall back to up to four prior recorded weekly
scores. Current-week actual scores never enter the estimate. Scheduled byes count
as zero; missing starter estimates withhold the total. No automatic substitutions
are made on the overview. Each expanded team row shows starter estimates and sources.
Actual records, points/week, all-play and previous-week matchups use completed ESPN
results. Depth and alternate-lineup analysis remain on the Team Strength page.

V3 adds player standard deviations and P10–P90 ranges. Lineup ranges use saved
marginal distributions and a provisional dependence approximation; K/DST scales
transfer errors from prior-score forecasts and have not been validated around ESPN
means. Current forecasts are saved at a specific cutoff. A weekly forecast is never
reused as the weekly prediction for another week. Season charts fall back to the
preserved horizon forecasts outside supported weekly cutoffs. Post-kickoff
reconstructions are excluded from verified pregame accuracy.

## Weekly re-scoring

`just campaign-rescore` (also run by `just start` and the weekly refresh) is
driven by `experiments/future_player_lab/campaigns/configs/*.json`; no campaign
id is hard-coded in `rescore.py`. A config's `rescore` block chooses the method:
`{"method": "rolling_origin"}` (currently v2) adds a `rolling_origin` forecast at
the latest completed week to its QB/RB/WR/TE remaining-season models that carry
saved weights; `{"method": "weekly"}` (currently v3) re-scores weekly models.
`selected_by` lists other ids whose `--campaign` filter also selects it. It reuses saved weights and the existing feature builders;
later cutoffs are labelled as rolling re-scores, not newly validated weekly models.
The previous bundle revision is archived on replacement.

The same refresh entry point also updates v3's four weekly models when a newer
completed week is available. It applies saved weights to freshly constructed
features, preserves earlier forecasts and retains the original uncertainty
calibration. Each added forecast binds its own feature matrix for offline replay.
New seasons require a new training campaign; weekly refresh does not silently fit
new weights or claim that retained uncertainty scales have been recalibrated.

See the [TE research results](../research/opening_drive_v2_te_2026-10-10.md) for the
receiving-focused inputs and PPR target definitions, including occasional rushing.


### Tracker configs

After re-scoring (and on no-op or week-18 refreshes), `rescore` runs every
config's `tracker` block; the 15-minute monitor (`campaigns.monitor`, service
`opening-drive-monitor`) runs the same blocks' `pregame` arm and then their
`backfill` arm. Adding a version only requires a config file:

```json
"tracker": {
  "enabled": true,
  "selected_by": ["opening_drive_v3"],
  "requires": ["weekly_v32.integration:DEFAULT_INTEGRATION"],
  "refresh": {"order": 20, "steps": [
    {"call": "weekly_v41.track:track", "args": ["snapshot"], "kwargs": {"registry": "registry"}}
  ]},
  "pregame": {"order": 20, "steps": [...]},
  "backfill": {"order": 20, "steps": [
    {"call": "weekly_v41.track:backfill", "args": ["snapshot"],
     "kwargs": {"registry": "registry", "limit": "limit"}}
  ]}
}
```

**Automatic backfill.** A campaign published mid-season gets forecasts for the
season's earlier weeks without a manual step. `campaigns/backfill.py` finds origins
`0..through_week-1` missing from its weekly models. For each one it writes a derived
snapshot manifest under `data/research/backfill_snapshots/` that uses the same sealed
assets with `through_week` set to the origin (every loader truncates current-season
box scores there), then calls the campaign's own forecast function, which publishes
it with `retrospective: true`. The monitor fills two origins per pass, newest first.
To fill everything at once:

```sh
.venv/bin/python -m experiments.future_player_lab.campaigns.backfill [--campaign ID] [--limit N]
```

Backfilled weeks are reconstructions. Box-score inputs stop at each cutoff, but the
models were fitted before the season, and injury reports and market lines are the
recorded ones, so they can be slightly better informed than a live forecast was.

- Step values are context names (`snapshot`, `registry`, `allow_network`, `limit`, named
  `inputs`), `module:ATTR` references relative to the campaigns package, or JSON
  literals. The first unkeyed step gives the result; a step with `key` adds a field.
- `order` sets the run order per arm (refresh: v3.1, v4.1, v3.2; pregame: v3.2, v4.1).
- `parent` runs a tracker only after its parent ran and nests the result under
  `result_key` (v4 under v3.2, as `v4`).
- `enabled: false` reports the campaign as paused/not tracking (v3.1).
- `requires` skips a tracker when a path is missing; `inputs` can be overridden with
  `--tracker-input CAMPAIGN.NAME=PATH`, which also forces that tracker.
- Trackers run only against the default registry unless inputs are supplied. Each is
  isolated: a failure is printed and returned as `{"status": "failed"}` and never
  blocks other trackers or the refresh.
## Organization

- `experiments/future_player_lab/campaigns/`: shared feature construction,
  chronological training/evaluation, saved-model inference, and campaign assembly.
- `experiments/future_player_lab/runs/`: immutable local research runs, fitted
  models, feature contracts, predictions, provenance, and evaluation records.
- `data/campaigns/<campaign_id>/`: completed, verified bundles for API access.
- `src/engine/chairstats/`: registry, validation, and publication of local bundles.
- `/api/chairstats`: campaign queries used by every forecast-bearing page.

Generated data and weights are excluded from Git. Keep the research run and its
source snapshots when backing up a campaign; source code alone cannot reproduce
a specific saved estimator without its data and recorded environment.

The existing [model library](model-library.md) inventories methods across the
project. The campaign registry binds executable research and queryable predictions
to an explicit ID. The API reads verified prediction artifacts; model fitting and
deserialization happen in the offline research commands, outside request handling.
This uses the existing FastAPI/filesystem stack rather than requiring another
local model-management service.

The [campaign README](../../experiments/future_player_lab/campaigns/README.md)
contains the reproducible training, scoring, audit, and replay commands. After
preparing the [GPU environment](gpu_research_setup.md), the corresponding shortcuts
are `just campaign-train`, `just campaign-score`, and `just campaign-replay`.
Use a new run ID for a changed experiment. New campaigns receive new IDs;
an explicitly requested correction can retain its campaign ID through the verified
replacement workflow below. `just campaigns list`
and `just campaigns check opening_drive_v2` inspect existing bundles.

## Comparing saved campaigns

The [local comparison report](../../data/outputs/model_campaign_comparisons/index.html)
puts the original RB models and Opening Drive side by side, followed by a separate
comparison against the existing published QB passing policy. It records campaign
identities, matching historical outcomes, current forecast differences, and source
checksums. The [RB comparison data](../../experiments/future_player_lab/runs/rb_campaign_comparison_20261010_001/comparison.json)
also retains per-target and annual results.

On matched 2021–2025 RB observations, Opening Drive reduces preseason PPR RMSE
from 58.78 to 57.37 and after-Week-2 remaining-season RMSE from 46.45 to 46.04.
The gains are modest and do not occur in every year; remaining-season targets and
receptions have slightly higher errors. These are reused development seasons.
Opening Drive refits weights using the original research approach; the original
estimators remain saved under RB Exploratory Research.

Rebuild the browser view from the frozen reports without running models:

```sh
uv run --no-sync python -m experiments.future_player_lab.campaigns.comparison_browser \
  --rb-report experiments/future_player_lab/runs/rb_campaign_comparison_20261010_001/readout.md \
  --qb-report experiments/future_player_lab/runs/opening_drive_qb_comparison_20261010_001/comparison.md \
  --output data/outputs/model_campaign_comparisons
```

## External campaign evaluation

`engine campaigns benchmark` compares a frozen campaign with external forecasts.
It is a separate, read-only evaluation command: it never fits models, chooses a
recipe, changes campaign weights, updates serving, or feeds external scores into
the selection pipeline. Run it after a campaign has saved its evaluation artifacts.
The report pins the campaign's manifest hash, so a same-ID correction has its own
evaluation. External inputs are reusable across campaigns.

The initial adapters cover Opening Drive v1, v2, the TE extension, and Scout Team's
saved RB ensemble. Scout Team QB historical evidence has a different schema and
is explicitly reported as unsupported; it is not silently evaluated as an Opening
Drive model. A new campaign can supply the sealed contract described below.

### Run an evaluation

```sh
uv run --no-sync engine campaigns benchmark opening_drive_v2 \
  --source data/benchmarks/espn_season_20261010_r3/source.json \
  --source data/benchmarks/fantasypros_ecr_20261010/source.json \
  --output data/outputs/external_benchmarks/NEW_RUN
```

For v1 or Scout Team RB, additionally pass
`--calendar data/benchmarks/opening_drive_calendar.json`. This explicitly applies
the shared Opening Drive calendar contract to their calendar-week targets. Do not
reuse that calendar for a campaign with different information cutoffs.

To evaluate every saved campaign and create a linked report overview:

```sh
uv run --no-sync engine campaigns benchmark-all \
  --source data/benchmarks/espn_season_20261010_r3/source.json \
  --source data/benchmarks/fantasypros_ecr_20261010/source.json \
  --calendar data/benchmarks/opening_drive_calendar.json \
  --output data/outputs/external_benchmarks/NEW_REVIEW
```

Open `NEW_REVIEW/index.html` to navigate the campaign reports. The same captured
sources can be reused for later campaigns with compatible historical windows;
new seasons require new source captures. Unsupported models remain visible.

Each output directory is create-only. `index.html` is the browser report;
`report.json` contains target/year/cohort metrics, coverage, source limitations,
and exact input hashes. `paired_rows.json`/CSV retain every matched prediction;
`exclusions.json` and `coverage.json` explain omissions. `manifest.json` hashes
the output files and copied source receipts/code. Generated inputs and reports
are local artifacts, excluded from Git, and should be retained with campaign
backups. The report can be read without model libraries or the original runs.

### Comparison rules

- Match exact GSIS player ID, season, position, target, origin week, horizon and
  final regular-season week. There is no name matching, weekly-to-season scaling,
  later-week summation, or replacement of missing players with zero forecasts.
- Numeric points require identical scoring IDs. The initial scoring contracts
  are `opening_drive_qb_v1` and `opening_drive_skill_ppr_v1`; their exact component
  coefficients are in `engine.evaluation.benchmarks.SCORING`. All unlisted
  components, including fumbles, conversions and bonuses, are excluded. Raw
  count/yard targets match their units directly.
- Archived forecasts must have witnessed availability no later than the campaign
  cutoff. An explicit issue time must also precede the cutoff. The default maximum
  source age is 14 days; `--max-age-days` changes this declared evaluation policy.
  Post-cutoff, stale, incompatible and missing records are reported as exclusions.
- Retrospective captures without certified issue times are scored separately as
  `reconstructed_external_vintage`, never as proof of pre-cutoff accuracy.
  Both clocks and the capture receipt remain in the report.
- ECR ranks support Spearman correlation and top-K realized production, not point
  MAE/RMSE. Standard PPR ranks differ from our component exclusions, so those
  comparisons carry `ranking_scoring_difference`. They are contextual rankings
  diagnostics, not exact scoring matches or reproductions of FantasyPros' contest.
- Report each source on its own paired population; never declare a winner between
  sources evaluated on different players. Coverage includes unmatched players,
  external-only IDs, and top picks absent from either side of the comparison.
- Numeric metrics include MAE, RMSE, bias, MSE reduction, annual results and
  equal-season MAE gains. The forecast-relevant cohort is the union of each side's
  top 2K predictions within each matched year, never selected by actual production.
  Selection uses K=12 QB/TE and K=24 RB/WR on matched players. Pooled rank and
  selection summaries average annual diagnostics, not cross-season ranks.
- MAE-gain intervals resample whole seasons with a fixed seed and require at least
  three seasons. These are exploratory intervals, with no multiple-comparison
  correction or claim that repeated-player dependence has been eliminated.

No matching external source is a reportable gap, not a reason to change our
forecast's horizon. The initial external sources are preseason only: remaining-
season cells therefore report unavailable comparisons. The generalized import
contract supports genuine remaining-season snapshots at the exact origin.

### Capture and import sources

Export the sealed availability clock once:

```sh
uv run --no-sync engine campaigns benchmark-calendar opening_drive_v2 \
  --output data/benchmarks/opening_drive_calendar.json
```

The current local preserved source files are:

- DynastyProcess ECR archive:
  `data/raw/objects/83/83f98047d85cf1f5e19c63020ca348d817d753d1351b4fba729404592a73e132`
- Provider-ID crosswalk:
  `data/raw/objects/cd/cdc998f6572bb0c332de763a97cd8d1e071c3e6633f6975af3cf338a442b7737`

Both are parquet assets recorded in the raw snapshot
`canonical_20260923_r4`. Substitute a reviewed newer archive/crosswalk explicitly;
the importer preserves the supplied bytes and their hashes.

```sh
uv run --no-sync engine campaigns benchmark-ecr \
  --archive PATH_TO_ECR_PARQUET --crosswalk PATH_TO_CROSSWALK_PARQUET \
  --calendar data/benchmarks/opening_drive_calendar.json \
  --season 2021 --season 2022 --season 2023 --season 2024 --season 2025 \
  --output data/benchmarks/NEW_ECR_CAPTURE

uv run --no-sync engine campaigns benchmark-espn \
  --crosswalk PATH_TO_CROSSWALK_PARQUET \
  --season 2021 --season 2022 --season 2023 --season 2024 --season 2025 \
  --output data/benchmarks/NEW_ESPN_CAPTURE
```

ECR chooses one latest positional PPR snapshot before each preseason cutoff;
date-only captures become usable at the end of that day. It never fills an omitted
player from an older snapshot. The [DynastyProcess archive](https://github.com/dynastyprocess/data)
is a ranking source, not a numeric projection dataset.

The ESPN adapter fetches the public historical `/seasons/{season}/players` endpoint
and reads only `statSourceId=1`, `statSplitTypeId=0`, `scoringPeriodId=0` records for
that season. It excludes actual-stat records and league-specific point totals.
It preserves raw JSON and request/capture receipts. Conflicting projection records
and ambiguous IDs are withheld. Missing stat keys remain unknown; points require
all scoring components to be explicitly present. This can leave point coverage
much smaller than individual-stat coverage, particularly for receivers without
explicit rushing fields. The report must not describe that subset as the full
fantasy population. Endpoint vintage and original publication times remain
unknown; this is a reconstructed diagnostic rather than a certified preseason
industry benchmark. No credentials are needed by this adapter.

For another provider, normalize its data to this JSON structure and run
`engine campaigns benchmark-import FILE.json --output NEW_SOURCE_DIRECTORY`:

```json
{
  "schema_version": 1,
  "source_id": "provider_snapshot",
  "name": "Provider and forecast definition",
  "kind": "projection",
  "limitations": ["Explain source coverage and original snapshot evidence"],
  "rows": [{
    "season": 2025, "origin_week": 0, "end_week": 18,
    "horizon": "season", "position": "QB", "player_id": "00-0034857",
    "target": "passing_yards", "prediction": 4000.0, "unit": "yards",
    "scoring": null, "vintage": "archived",
    "issued_at": "2025-09-01T12:00:00Z",
    "available_at": "2025-09-01T12:05:00Z",
    "source_url": "https://provider.example/original-snapshot"
  }]
}
```

This example is a schema illustration, not a real projection. `available_at`
means witnessed historical availability, not the date on which a retrospective
download happens. Use `vintage: reconstructed` and null original timestamps when
that evidence does not exist. Preserve capture time separately. For ranks use
`kind: ranking`, `target: points`, `unit: rank`, a positive ordinal prediction and
an explicit scoring identifier. Select one provider snapshot per comparison
window before importing; duplicate keys are rejected. Units are `count`, `yards`
or `fantasy_points` for the supported production targets.

### New campaign contract

New campaign model metadata can include `benchmark_evaluation` with:

```json
{
  "artifact": "evaluation_predictions",
  "recipe": "already_selected_policy",
  "scoring": "opening_drive_skill_ppr_v1",
  "cutoffs": [{
    "season": 2025, "origin_week": 0, "end_week": 18,
    "as_of": "2025-09-04T12:00:00Z"
  }]
}
```

The named artifact must be a checksummed parquet in the sealed campaign with
`year`, `origin`, `player_id`, `target`, `recipe`, `prediction`, and `actual` columns.
Include every historical cutoff. Only the declared saved recipe is evaluated;
other candidate recipes are ignored. Unknown campaign schemas are reported as
unsupported until an adapter is provided. This metadata documents evaluation
access; it is not a new training or model-selection criterion.

## Research protocol

The [RB tree study](../research/rb_gpu_campaign_2026-10-03.md) is the starting
point: preserve raw inputs, fit separate component targets, retain competitive
simple controls, and select ensemble members on earlier seasons. QB models add
passing attempts, completions, passing yards, touchdowns, and interceptions.
Each position receives its own fitted weights.

Compare calibrated persistence and tree forecasts on identical candidate rows.
Candidate eligibility must use evidence available at the prediction cutoff,
including players who later record zero production. Missing observations remain
unknown; absence of an appearance in an otherwise complete outcome season can
represent zero. Training, preprocessing, and recipe selection must precede each
evaluation season. Report errors by target and year as well as pooled errors;
players with substantial prior usage deserve a separate diagnostic.

The initial search includes conservative and more flexible tree recipes, following
the earlier [QB capacity study](../research/qb_boosting_capacity_2026-09-24.md).
Increasing depth improves capacity but does not establish future accuracy.
[XGBoost's parameter reference](https://xgboost.readthedocs.io/en/latest/parameter.html)
documents depth, minimum child weight, and regularization controls. Separate
selection from evaluation as described in the
[nested-validation guidance](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html),
while preserving chronological order rather than randomly shuffling player-seasons.

Historical evaluation is development evidence. These years have appeared in
earlier project research, and captured data may contain retrospective provider
revisions. Later-year predictions withheld from fitting and selection are useful
walk-forward evidence, but are not a previously untouched prospective experiment.

Independent component predictions need not sum to a separately trained points
prediction. Retain both and label their definitions. Dividing a season total by
scheduled games produces a per-game average, not a matchup-specific weekly model.

## Scope and promotion

The first campaign supports preseason and remaining season after Week 2. A
reconstructed 2026 Week 2 forecast uses observations through Week 2, even if the
source bytes were captured later. Do not treat it as an originally issued forecast
or an injury-adjusted current-week projection.

Campaign creation does not change `data/current.json` or approve any model for
application rankings. Serving a research prediction through its explicit campaign
endpoint makes it inspectable; it does not confer promotion. A later promotion
decision should review target-level gains and regressions, scoring compatibility,
coverage, and behavior under the production data-refresh workflow.

## Querying the current campaign

Use the API server (`uv run engine serve`):

```sh
uv run engine campaigns list
uv run engine campaigns show opening_drive_v2
uv run engine campaigns check opening_drive_v2

curl 'http://localhost:8000/api/chairstats/campaigns'
curl 'http://localhost:8000/api/chairstats/campaigns/opening_drive_v2/models'
curl 'http://localhost:8000/api/chairstats/campaigns/opening_drive_v2/models/te_remaining/metrics'
curl 'http://localhost:8000/api/chairstats/campaigns/opening_drive_v2/models/te_remaining/forecasts?season=2026&origin_week=4'
```

Forecast requests require both `season` and `origin_week`; an unavailable snapshot
returns 404 instead of selecting another season or campaign. Optional `player_id`
filters by exact player identifier. `limit` (1–1000, default 100) and `offset`
paginate the saved row order. The response includes the total matching count,
model and campaign IDs, forecast checksum, and campaign manifest checksum. Rows
contain `player_id`, `position`, `season`, `origin_week`, and a `predictions` mapping
whose keys match the model's target contract. Additional saved diagnostics and
display fields pass through unchanged.

The response's `metadata` mapping preserves the saved forecast receipt without
duplicating its rows: scoring, reconstruction/vintage status, generation time,
training cutoff, source snapshot and supplement receipts, and interpretation
notes remain available to frontend consumers. It is empty for a rows-only JSON
artifact. Receipt identity fields, when present, must match the campaign and
model contract.

Campaign detail and model endpoints return `X-Campaign-Manifest`. Consumers may
send that value on subsequent requests to reject a changed manifest with 409.
Responses identify their `research_preview` disposition. Read errors, corrupted
artifacts, invalid contracts, and escaping paths fail closed with 409. Metadata
reads verify the sealed manifest; metrics and forecast reads additionally verify
the requested JSON bytes. `engine campaigns check` audits all JSON outputs and
saved binary artifacts without deserializing weights.

`engine.chairstats.publish_campaign` assembles a bundle from offline outputs. Its
metadata includes `campaign_id` and `name`; each model includes its ID, position,
horizon, observation cutoff, targets, training cutoff, metrics JSON path, forecast
JSON paths, and optional named artifact paths. Publication copies these files,
binds their SHA-256 checksums in `manifest.json`, seals the manifest with
`manifest.sha256`, and atomically exposes a new directory. Existing campaign IDs
cannot be overwritten by this create-only function. A future promotion workflow
can select an explicitly reviewed ID separately. The API does not fit models,
load pickle/joblib code, fetch live data, or accept arbitrary feature vectors.

For an explicitly requested same-ID update, build and replay a complete candidate
in a separate registry first. `engine.chairstats.replace_campaign` requires the
current manifest hash, verifies both bundles, archives the current bytes outside
the API registry, and atomically exchanges the active directory. A stale hash is
rejected; verification or receipt failures after activation restore the old bundle.
The campaign ID stays stable while its manifest hash changes. Clients holding the
old `X-Campaign-Manifest` receive 409 and must deliberately refresh their snapshot.

```python
from pathlib import Path
from engine.chairstats import replace_campaign

receipt = replace_campaign(
    root=Path("data/campaigns"),
    candidate_root=Path("path/to/verified/staged_campaigns"),
    campaign_id="opening_drive_v2",
    expected_manifest_sha256="<verified current SHA-256>",
    archive_root=Path("data/campaign_revisions"),
)
```

The October 10 QB/rookie correction keeps `opening_drive_v2`. Its separate run is
`experiments/future_player_lab/runs/opening_drive_v2_update_20261010_001`.
The candidate repair preserves all historical rows and existing current features.
The QB experiment evaluates attempts and passing yards with explicit past-only
opportunity/efficiency features and an attempts-times-efficiency alternative;
seven other QB targets retain their weights. Each forecast year's policy uses
only the preceding three validation years, requiring at least 1% lower MAE and
MSE and no worsening of MSE for QBs with at least 300 prior-season attempts.
This remains historical development evidence, not an untouched test.

To replay a trusted local bundle using its copied weights and saved feature matrix:

```sh
uv run --no-sync python -m experiments.future_player_lab.campaigns.replay \
  --campaign opening_drive_v1
```

Replay needs the recorded numerical library versions, but not the original raw
snapshots or research-run directory. It verifies all bundle checksums before
deserializing the saved models, then checks target predictions, component points,
scheduled-game averages, exposure, and stable ranks against the saved receipt.
This initial replay contract binds one forecast feature matrix per model. Raw
feature reconstruction remains a separate audit that requires the source snapshots.

## Historical research import (outside application serving)

The import adapter verifies saved source evidence, normalizes forecast contracts,
and copies artifacts into a new immutable bundle. It does not retrain models or
change the selected application release. Choose a new receipt directory:

```sh
just campaign-import-scout \
  --output experiments/future_player_lab/runs/scout_team_import_NEW
```

Existing campaign IDs cannot be overwritten. Once imported, the shared campaign
API reads the copied JSON and does not depend on the old research directories.
Archive weights and source evidence with the bundle. The Opening Drive replay
runner cannot replay the whole Scout Team campaign: the original RB artifacts use
their original adapter, and the QB estimators were not retained. Availability and
replay limitations are explicit in the campaign/model metadata.

The initial imported members are:

| Model | Preserved artifacts | Saved 2026 forecasts |
| --- | --- | --- |
| `rb_season` | Original fitted weights and eight-target evaluation | None retained |
| `rb_remaining` | Original fitted weights, features, and eight-target evaluation | 170 players after Week 2 |
| `qb_season` | Saved passing-policy forecasts and evidence | 116 players, preseason |
| `qb_remaining_w1` through `qb_remaining_w3` | Saved passing-policy forecasts and evidence | 116 players at each cutoff |
| `qb_remaining_w4` | Exact published yardage forecasts, including availability constraints | 116 players after Week 4 |

Earlier QB cutoffs expose attempts, passing yards, and the policy's rate component.
Week 4 exposes served passing yards; the unconstrained attempts/rate remain in
`saved_research_components`, rather than inventing availability-adjusted components.
QB ranks order passing yards, not fantasy points.

To replay all eight RB targets from the copied bundle alone:

```python
from pathlib import Path
from engine.chairstats import CampaignStore
from experiments.future_player_lab.campaigns.scout_rb import replay_archive

root = Path("path/to/historical/research_registry")
model = CampaignStore(root).model("scout_team_v1", "rb_remaining")
bundle = root / "scout_team_v1"
archive = model["artifacts"]["original_library"]
print(replay_archive(bundle / archive["path"], archive["sha256"],
                     bundle / model["forecasts"][0]["path"]))
```

The import receipt is saved under `runs/scout_team_import_20261010_001` in the
future-player lab. Independent replay evidence is under
`runs/scout_team_verification_20261010_001/rb_archive_replay.json`.

## Opening Drive v3.1 interval challenger

V3.1 keeps the v3 point models and distribution artifacts fixed and evaluates
separate calibrated intervals. It is a research challenger, not another application
campaign choice. See the [experiment and tracking contract](../../experiments/future_player_lab/campaigns/weekly_v31/README.md).

The completed historical report is
`data/outputs/opening_drive_v31_comparison_2026/index.html`; the accumulating
prospective scorecard is
`data/outputs/opening_drive_v31_prospective/scorecard/index.html`.

After the default v3.1 run is created, normal campaign refresh also locks the first
complete paired weekly forecast before 00:00 UTC on the first NFL game date and
scores previous captures from completed games. This also happens when v3 already
has its current origin. Existing captures are never replaced, and late captures
cannot be admitted as prospective. No separate scheduler is installed.

```sh
.venv/bin/python -m experiments.future_player_lab.campaigns.rescore --due
.venv/bin/python -m experiments.future_player_lab.campaigns.weekly_v31.prospective track
.venv/bin/python -m experiments.future_player_lab.campaigns.weekly_v31.prospective replay --capture-dir data/outputs/opening_drive_v31_prospective/forecasts/2026_06
```

Both versions share the same point forecasts and evaluated populations. Player
intervals, artificial lineups and saved actual league offensive lineups have
separate metrics. Missing outcomes remain pending. K/DST is excluded, and later
lineup substitutions do not rewrite the saved roster. Review interval score and
width alongside coverage; the system never promotes the challenger automatically.

## V3.2 live comparison (2026-10-13)

V3.1 research and its automatic tracking are paused; all captures and results are
retained. The v3.2 preview is a separate injury/workload challenger based on v3.
The user selected it as the API/dashboard default on 2026-10-13. V3 remains
available for explicit comparisons, and prospective tracking continues.

The regular campaign refresh now captures injury evidence and checks v3.2 even
when no new v3 origin is due. A local user timer also checks every 15 minutes.
Early-week forecasts lock first; per-game updates require fresh, correctly dated
official weekly reports and must be captured 90–60 minutes before kickoff.
Provider tags alone do not qualify, and missing evidence retains the early
forecast. All comparisons retain immutable inputs and support offline replay.

The new empirical uncertainty distributions were evaluated around v3.2's means.
K/DST uncertainty and mixed-timing lineup ranges are withheld for this preview.
At the initial 2026-10-13 integration, season/remaining-season models were inherited
unchanged. The 2026-10-14 bridge replaced current horizon outputs, but its v2 anchor
and points-only outputs leave the [version-lineage requirement](#version-lineage-and-retained-functionality)
unfulfilled. Prospective player and actual offense-lineup scores remain pending
until their games complete.

See the [v3.2 reproduction and collection contract](../../experiments/future_player_lab/campaigns/weekly_v32/README.md)
and `data/outputs/opening_drive_v32_shadow/index.html` for the integration review.
