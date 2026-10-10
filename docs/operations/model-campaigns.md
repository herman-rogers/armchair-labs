# Model campaigns

**Opening Drive** (`opening_drive_v1`) is Armchair Labs' first grouped RB, QB,
and WR tree-model campaign. A campaign fixes the research protocol, data sources,
feature and target contracts, evaluation evidence, and saved models together.
Later campaigns can coexist under different IDs without replacing earlier results.

For comparison, the original RB GPU fits and their controlled extensions are now
cataloged as **RB Exploratory Research** (`rb_exploratory_research`). The new shared
campaign's display label is **Opening Drive v1** (`opening_drive_v1`). The checked-in
provenance catalog at `src/engine/models/campaigns.json` keeps these identities
separate without renaming or modifying sealed runs, trained weights, or bundles.
Historical manifests retain their original text, including `Opening Drive`.

**Scout Team v1** (`scout_team_v1`) groups the original RB study and the earlier QB
passing/variations work in the same campaign store and API used by Opening Drive.
It is a retrospective grouping of distinct studies, not a claim they shared one
training protocol. **RB Exploratory Research** remains the original RB source-study
identity inside Scout Team; existing reports and manifests retain that name.

Scout Team preserves original RB fitted weights and saved predictions. The old QB
workflow retained predictions, evaluation evidence, and training source, but did
not save its fitted estimators. Its API members therefore expose saved passing
forecasts only. They cannot perform new inference without separately fitting and
validating new artifacts. The QB capacity-search results remain research evidence;
the queryable QB forecasts come from the existing selected passing policy.

The model library lists both campaign bundles and the historical RB source study.
RB comparison artifacts remain at
`experiments/future_player_lab/runs/rb_campaign_comparison_20261010_001`.

This is a production-forecast campaign: it predicts observed volume, yards,
touchdowns, and explicitly scored fantasy points. It does not estimate a player's
context-independent ability.

Every published bundle is served immediately. There is no promotion step: the
rankings, league and profile pages ask `/api/chairstats` for a campaign by id, and
the newest bundle is the request-time default when a page names none.

The **Opening Drive v2 tight end extension** (`opening_drive_v2_te`) supplies
`te_season` and `te_remaining` through the same API, with receiving-focused inputs
and PPR targets including occasional rushing. Its separate bundle preserves the
original v2 experiment. See the [TE workflow](../../experiments/future_player_lab/campaigns/README.md#tight-end-extension-of-v2).

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
and `just campaigns check opening_drive_v1` inspect existing bundles.

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

## Querying saved campaigns

Use the existing API server (`uv run engine serve`). These explicit campaign
routes preserve the server's current-catalog verification boundary. They do not
replace `/api/nextgen` or the current ranking endpoints.

```sh
uv run engine campaigns list
uv run engine campaigns show scout_team_v1
uv run engine campaigns check scout_team_v1
uv run engine campaigns show opening_drive_v1
uv run engine campaigns check opening_drive_v1

curl 'http://localhost:8000/api/chairstats/campaigns'
curl 'http://localhost:8000/api/chairstats/campaigns/scout_team_v1/models'
curl 'http://localhost:8000/api/chairstats/campaigns/scout_team_v1/models/rb_remaining/forecasts?season=2026&origin_week=2&player_id=00-0038134'
curl 'http://localhost:8000/api/chairstats/campaigns/scout_team_v1/models/qb_remaining_w4/forecasts?season=2026&origin_week=4'
curl 'http://localhost:8000/api/chairstats/campaigns/opening_drive_v1/models'
curl 'http://localhost:8000/api/chairstats/campaigns/opening_drive_v1/models/qb_remaining/metrics'
curl 'http://localhost:8000/api/chairstats/campaigns/opening_drive_v1/models/rb_remaining/forecasts?season=2026&origin_week=2&player_id=00-0038134'
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

## Importing the preserved studies

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

root = Path("data/campaigns")
model = CampaignStore(root).model("scout_team_v1", "rb_remaining")
bundle = root / "scout_team_v1"
archive = model["artifacts"]["original_library"]
print(replay_archive(bundle / archive["path"], archive["sha256"],
                     bundle / model["forecasts"][0]["path"]))
```

The import receipt is saved under `runs/scout_team_import_20261010_001` in the
future-player lab. Independent replay evidence is under
`runs/scout_team_verification_20261010_001/rb_archive_replay.json`.
