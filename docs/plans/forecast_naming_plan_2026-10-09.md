# Rename the forecasting system

Replace our use of “NextGen” throughout the active application, research tooling and maintained documentation. The name obscures the individual methods and overlaps with the NFL data provider’s terminology. This is a naming and compatibility migration, not a model change.

Status: proposed, October 9, 2026. Implementation has not started. The replacement naming below is recommended and remains subject to the user’s naming preference.

## Naming direction

Use **Armchair Labs** for the product and descriptive names for capabilities. Avoid introducing another umbrella model name: weekly references, ML challengers, QB passing policies and RB research are different methods.

| Current usage | Proposed replacement |
|---|---|
| NextGen as the overall system | Armchair Labs |
| NextGen forecasts | Forecasts, with horizon and method shown |
| NextGen weekly reference | Weekly reference |
| NextGen ML challenger | Weekly boosted trees or Weekly ridge |
| NextGen rankings | Forecast rankings |
| NextGen overall or position columns | Forecast overall rank or Forecast position rank |
| NextGen player analysis | Player analysis or Player profiles, as appropriate |
| NextGen research | Research, followed by the study’s actual subject |
| NFL Next Gen Stats, NGS and provider field names | Preserve as external-source terminology |

Keep evidence status separate from method name: “Reference,” “Experimental,” and “Historically validated” describe evidence, not brands. Retain the distinction between next calendar week, next four calendar weeks, scheduled team games and remaining season.

## Scope found in the repository

The rename reaches UI copy, component names, API paths and tags, client functions, query keys, Python modules, research runners, command aliases, export filenames and headers, serialized fields, runtime locks/status files, model product kinds and forecast-log source identifiers. Existing release directories and frozen implementation copies also contain the name.

Concrete entry points:

- UI: `web/src/components/NextGenView.tsx`, `NextGenRankings.tsx`, league views, `web/src/researchModels.ts` and `.nextgen-workspace` styles.
- Clients: `web/src/api/nextgen.ts`, other clients calling `/api/nextgen`, and `web/src/api/queries/index.ts`.
- Backend: `src/engine/api/nextgen_routes.py`, dependent routers, API application middleware, `src/engine/data/nextgen.py` and `src/engine/metrics/nextgen.py`.
- Research and operations: `research/nextgen_system.py`, ranking/signal scripts, `Justfile`, refresh orchestration and maintained runbooks.
- Persistence: `nextgen_weekly_model` / `nextgen_weekly_reference`, `nextgen_analysis` / `nextgen_weekly_points`, `fitted_nextgen_*` fields, and runtime refresh lock/status paths.
- Documentation: current architecture/operations pages, dated research indexes and the new HTML research overview.

The existing `nextgen-refresh` command is already an alias for `data-refresh`; reuse that canonical command rather than inventing another.

## Proposed technical vocabulary

Use purpose-specific module names rather than mechanically replacing every occurrence with `armchair`.

| Surface | Proposed destination |
|---|---|
| `/api/nextgen/...` | `/api/analysis/...`, preserving endpoint suffixes initially |
| `nextgen_routes.py` | `analysis_routes.py` |
| `engine.data.nextgen` | `engine.data.analysis` |
| `engine.metrics.nextgen` | A purpose-specific metrics module chosen after reviewing its exports, provisionally `forecast_baselines` |
| `web/src/api/nextgen.ts` | `web/src/api/analysis.ts` |
| `NextGenView` / `NextGenRankings` | `AnalysisView` / `ForecastRankings` |
| `.nextgen-workspace` | `.analysis-workspace` |
| `nextgen_*` client functions | Capability names such as `getCatalog`, `getRankings`, `getWeeklyForecasts` |
| `X-NextGen-Analysis` | `X-Analysis-Version` |
| New product kinds | `analysis`, `weekly_points` |
| New forecast-log source identifiers | `weekly_model`, `weekly_reference` |
| Ranking commands | `rankings-build`, `rankings-publish` |

Check collisions before adopting exact symbols. Distinguish externally serialized names from private variables; do not silently break consumers by changing both at once.

## Implementation sequence

### 1 Establish a behavior baseline

Inventory tracked source, ignored local analytical products, serialized contracts and external consumers separately. Save the current catalog identity and representative API responses: rankings, weekly forecasts, QB passing, league overview and forecast accuracy. Record current formulas, predictions, evidence decisions and history counts for later parity checks.

Identify any scripts or scheduled jobs using old commands or routes. Existing unrelated working-tree changes must remain intact.

### 2 Rename the product language

Update navigation, titles, labels, explanations, errors, download names, API tags and the HTML research overview. Use descriptive method names so the change also resolves the confusion between “NextGen” and its challengers. Preserve legitimate NFL source references.

Keep published versus experimental status and forecasting horizons visible. Update maintained docs; add a terminology note to dated reports where needed without rewriting historical findings.

### 3 Migrate the active code and API

Rename modules, components, imports, internal symbols, CSS classes, tests and runners. Introduce `/api/analysis` as canonical and keep the old API prefix temporarily as a compatibility alias backed by the same handlers.

Update middleware path classification, authentication, query scopes, catalog guards, export headers, client calls and query invalidation together. A changed prefix must not bypass the existing verification/query boundary. Old and new routes must enforce identical behavior.

Choose a cache namespace transition so old cached data cannot appear under incompatible response shapes. Existing public page routes should continue to work unless a route actually includes the retired name.

### 4 Migrate persistence without losing evidence

New writes use the new identifiers. Readers normalize explicitly supported legacy kinds, fields and forecast sources at the boundary. Add a schema version where needed; reject unknown values rather than loosening verification.

Preserve immutable historical releases, their hashes, original forecast timestamps and saved predictions. Do not rewrite frozen manifests or archived source copies just to remove a word. If a new current artifact needs renamed metadata, create a successor with explicit lineage; verify unchanged numerical outputs and update dependent hashes before publication. Do not retrain merely for a rename.

Specifically test old and new forecast-log source identifiers so historical accuracy records do not disappear. Handle serialized profile fields such as `fitted_nextgen_season_points` through explicit mapping; keep their archived model meaning intact.

Treat refresh locks separately from labels. Never allow old and new commands to acquire different locks for the same refresh. Use one shared lock during transition, then retire the old lock path only after all callers have migrated and no refresh is running. Handle runtime status readers and precomputed caches in the same transition.

### 5 Remove active legacy usage and document the exceptions

Switch all first-party callers to the new names. Inventory remaining references and classify each as external NFL terminology, immutable historical provenance, dated documentation, or temporary compatibility support.

Give compatibility aliases a removal condition: no first-party callers remain, known automation has migrated, old saved artifacts are covered by read adapters, and the parity checks pass. Then remove obsolete routes/import/command aliases together. Keep legacy artifact decoding for as long as those artifacts remain supported.

The final active product should not advertise or depend on the old branding. Historical bytes and genuine provider terminology are intentional exceptions, not unfinished search-and-replace work.

## Verification and completion criteria

- No internal “NextGen” branding remains in active UI, current documentation, errors, API tags or newly generated exports; legitimate NFL attribution remains accurate.
- Frontend type/build checks and affected backend tests pass; primary views, controls and exports pass browser smoke checks.
- Canonical and temporary legacy endpoints return equivalent data and enforce the same authentication, release verification and query scope.
- Published forecasts, rankings, model decisions and formulas are unchanged; compare values independently of intended metadata renames.
- Current and historical artifacts remain readable with valid hashes; the existing catalog can be restored for rollback.
- Forecast accuracy retains eligible historical captures and their original timestamps. Renaming does not create new pregame evidence.
- Refresh uses one lock and publishes atomically. Old and new status readers cannot trigger competing refreshes.
- Source search leaves only reviewed exceptions; new source files and first-party commands use descriptive names.
- The research overview and documentation links resolve after file renames.

## Deliverables

Deliver the work in reviewable stages: product copy and documentation; code/API migration with compatibility; artifact/read-adapter migration and parity checks; removal of obsolete aliases after callers move. Keep this plan’s status updated as each stage ships.

The replacement brand is the only user preference needed before implementing the naming. This plan does not request changes to models, scoring, training data, or serving approvals.
