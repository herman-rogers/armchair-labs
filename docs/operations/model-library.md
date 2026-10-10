# Model library

Use the library to find current application methods, reusable research models and historical experiments. It separates ability research, opportunity, team context, production and fantasy applications. It is an inventory, not an alternative model-serving registry.

Named campaigns have a separate [training and API workflow](model-campaigns.md).
The **Campaign provenance** section distinguishes these campaigns:

- **Scout Team v1** (`scout_team_v1`) groups the original RB models and preserved
  QB forecasts. Its RB records retain fitted estimators; the QB export contains
  saved passing forecasts, without claiming reusable QB estimator weights.
- **Opening Drive v1** (`opening_drive_v1`) contains the newly trained RB/QB/WR
  weights under the shared tree campaign protocol.
- **Opening Drive v2** (`opening_drive_v2`) tests the same methods with dated
  availability features and formal retirement eligibility. Its comparison pairs
  forecasts on the same eligible players and separately measures v1 selection
  with the eligibility rule alone.

**Opening Drive v2 — Tight End** (`opening_drive_v2_te`) extends the v2 protocol
with separately fitted TE preseason and after-Week-2 models, a fixed receiving/usage
feature mask, and the same dated availability policy. The sealed original v2
bundle remains preserved. [TE results](../research/opening_drive_v2_te_2026-10-10.md).

**RB Exploratory Research** (`rb_exploratory_research`) retains its historical
identity as a source study within Scout Team v1. It is not an independent
campaign. Parent labels appear beside the source study and on associated runs and
saved-model contracts. Campaign availability does not change the application's
selected models.

Scout Team's QB provenance identifies the exact preserved passing/variation runs
and the earlier capacity study. The capacity study is evidence, not a source of
saved fitted estimators. A mixed application release can supply a referenced QB
artifact without assigning every other model in that release to Scout Team. The
initial QB export concerns attempts, passing yards, and yards per attempt; it does
not import the separate fantasy bridge or assert that all source methods use GPUs.
The Week 4 QB record preserves the published availability-constrained passing-yard
forecast; unconstrained attempts and efficiency remain separate diagnostics.
Earlier cutoff records retain the saved attempts, yards, and efficiency forecasts.

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

- Campaign provenance: `src/engine/models/campaigns.json` associates descriptive
  labels, original manifest references, configurations, and run paths. It neither
  edits historical artifacts nor makes a campaign queryable. Only actual sealed
  bundles under `data/campaigns` appear in `/api/campaigns`. Scout Team's exported
  original forecasts are queried through its campaign ID once its local bundle is
  available; the historical RB source-study ID remains provenance, not an extra
  API bundle. Original saved-model tools and frozen forecasts remain intact.
- Method descriptions: `src/engine/models/families.json`. Use scoped assessments and linked evidence; do not put approval flags or benchmark numbers here.
- Current serving decisions: `data/current.json` → selected analysis registry. The library verifies hashes of the selected manifest and displayed registry/reports. Weekly decisions additionally require the selected analysis identity to match.
- Reusable model contracts: JSON definitions under experiment labs, including the existing saved-model registry. Supported positions, origins, scoring and artifact references stay visible.
- Local runs: direct children of `data/research`, `data/campaigns`, and `experiments/*/runs`. Hidden campaign staging directories are excluded. Missing manifests remain explicitly unindexed; a completed run is not an approved model.
- Source inventory: Python modules in metrics, board, research runners, and the experiment/deep/GPU/saved-forecast directories. `source_groups.json` assigns module ownership and shared infrastructure separately from curated entrypoints. Ownership is not a claim that every file implements a model. Unassigned sources are a review backlog, not a count of unvalidated models.
- Removed or narrowed claims: `src/engine/models/claims.json`. Retain evidence for negative findings instead of deleting the record of what failed.

The catalog embeds all selected release registry entries, including measurement and archived entries. Counts of entries, run directories, recipes and fitted estimators are different quantities. The catalog deliberately does not collapse them into a misleading “number of models.”

## Verification boundaries

Current manifest or displayed artifact hash mismatches suppress the affected current metadata and cause `build` to exit unsuccessfully after writing a diagnostic catalog. `check` also fails for missing family sources and unreadable inventory records. Absent generated data is reported; family descriptions remain usable on a fresh clone.

Run status is self-reported metadata, not a replay audit. A matching saved-model manifest does not verify every weight file or package dependency; its original loader still enforces those contracts. Matching manifests also expose an input-name audit against `retired_features.json`. An empty match list establishes only that those named legacy features are absent, not that all inputs are primitive measurements. This library does not replace the application’s complete release-verification path.

## Maintaining the library

Add a method family when it represents a distinct question or methodological family, not every trained fold. Link implementation and evidence, state its targets and horizons, and explain what it cannot establish. New configurations and run directories appear on the next build automatically. Family assignments are explicit; review unclassified files rather than assuming they are unused.

The [player research direction](../plans/player_research_direction.md) records the separate ability-rating and team-context tracks. The [rename plan](../plans/forecast_naming_plan_2026-10-09.md) remains a separate compatibility migration.
