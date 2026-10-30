# Model library

Use the library to find current application methods, reusable research models and historical experiments. It separates ability research, opportunity, team context, production and fantasy applications. It is an inventory, not an alternative model-serving registry.

The application defaults to **Opening Drive v3.2 preview** (`opening_drive_v32`),
with v4 candidate, v3 and v2 available for comparison. The
[v4 candidate workflow](../../experiments/future_player_lab/campaigns/weekly_v4/README.md)
records its restored components, v3-based horizons, uncertainty evaluation and
isolated retraining experiment. Its mixed historical results and pending prospective
outcomes do not justify default promotion. See the [campaign workflow](model-campaigns.md)
for current serving behavior and the
[version-lineage requirement](model-campaigns.md#version-lineage-and-retained-functionality):
new versions must improve their declared base and retain functionality, without
silently reverting to an older model. The current v2-anchored v3.2 season bridge
is a documented gap against that requirement.

The provenance catalog currently describes v2; it is not the authority for the
application's selected default or the completeness of newer versions. Its `archived_campaigns`
records retain Scout Team, Opening Drive v1 and the original TE extension for
historical reference. Those bundles live outside the application registry under
`data/campaign_revisions/` and are not frontend choices or application API campaigns.

```sh
uv run engine models list
uv run engine models list --layer execution
uv run engine models show rb-gpu
uv run engine models build --open
uv run engine models check
```

`build` creates `data/outputs/model_catalog/index.html` and `catalog.json`. The HTML has search, layer filters and separate sections for methods, published decisions, runs, configurations/saved models, source inventory and claim cleanup. It has no external assets. Rebuild after new research or publication; the file is a dated snapshot, not a live view.

`--data-dir PATH` selects another local data directory for `build` and `check`; `--output PATH` changes the generated destination. No command trains, executes saved estimators, fetches sources, changes serving decisions or rewrites research artifacts.

## Source of truth

- Campaign provenance: `src/engine/models/campaigns.json` describes the v2 campaign
  and retains historical records separately. Actual sealed artifacts under
  `data/campaigns/<campaign_id>` supply the application API; its default is declared
  in `src/engine/chairstats/application.py` and returned by `/api/chairstats/campaigns`.
- Method descriptions: `src/engine/models/families.json`. Use scoped assessments and linked evidence; do not put approval flags or benchmark numbers here.
- Current serving decisions: `data/current.json` → selected analysis registry. The library verifies hashes of the selected manifest and displayed registry/reports. Weekly decisions additionally require the selected analysis identity to match.
- Reusable model contracts: JSON definitions under experiment labs, including the existing saved-model registry. Supported positions, origins, scoring and artifact references stay visible.
- Local runs: direct children of `data/research`, `data/campaigns`, and `experiments/*/runs`. Hidden campaign staging directories are excluded. Missing manifests remain explicitly unindexed; a completed run is not an approved model.
- Source inventory: Python modules in metrics, chairstats, research runners, and the experiment/deep/GPU/saved-forecast directories. `source_groups.json` assigns module ownership and shared infrastructure separately from curated entrypoints. Ownership is not a claim that every file implements a model. Unassigned sources are a review backlog, not a count of unvalidated models.
- Removed or narrowed claims: `src/engine/models/claims.json`. Retain evidence for negative findings instead of deleting the record of what failed.

The catalog embeds all selected release registry entries, including measurement and archived entries. Counts of entries, run directories, recipes and fitted estimators are different quantities. The catalog deliberately does not collapse them into a misleading “number of models.”

## Verification boundaries

Current manifest or displayed artifact hash mismatches suppress the affected current metadata and cause `build` to exit unsuccessfully after writing a diagnostic catalog. `check` also fails for missing family sources and unreadable inventory records. Absent generated data is reported; family descriptions remain usable on a fresh clone.

Run status is self-reported metadata, not a replay audit. A matching saved-model manifest does not verify every weight file or package dependency; its original loader still enforces those contracts. Matching manifests also expose an input-name audit against `retired_features.json`. An empty match list establishes only that those named legacy features are absent, not that all inputs are primitive measurements. This library does not replace the application’s complete release-verification path.

## Maintaining the library

Add a method family when it represents a distinct question or methodological family, not every trained fold. Link implementation and evidence, state its targets and horizons, and explain what it cannot establish. New configurations and run directories appear on the next build automatically. Family assignments are explicit; review unclassified files rather than assuming they are unused.

The [player research direction](../plans/player_research_direction.md) records the separate ability-rating and team-context tracks. The [rename plan](../plans/forecast_naming_plan_2026-10-09.md) remains a separate compatibility migration.
