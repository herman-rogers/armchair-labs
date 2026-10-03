# NextGen implementation and operating instructions

> Historical implementation record. Release IDs, results, commands, and interface descriptions reflect the recorded change; use the [operations guides](../operations/README.md) to run the system today.

The default dashboard now consumes a verified analysis release. Descriptive
measurements, outcome-specific forecasts and league decisions have separate
contracts. Research results are preserved even when they cannot enter analysis.

## Published release

The single authority is `data/current.json`. This implementation publishes:

- Gold: `canonical_nextgen_20260923_r1`.
- Analysis: `nextgen_qb_variations_20260924_r2`, adding the approved
  [QB passing and four-week ranking policies](../research/qb_passing_variations_2026-09-24.md).
  It preserves unrelated ranking forecasts from `nextgen_rankings_20260923_r6`
  and all registry decisions in `revalidation_review_20260924_r1`.
- Profiles, college and outlook: `nextgen_products_20260923_r2_*`.
- Individual-stat evidence: `individual_stats_20260923_r4`, with its original
  corrected gold reference and hashes retained.

The [September 24 registry review](model_revalidation_review_2026-09-24.md)
resolved all 213 pending revalidations. Four dependencies were revalidated and
209 entries retired with explicit reasons. `validity=archived` closes a retired
entry's review without claiming valid inputs; `serving=archive` continues to
exclude it. No forecast or model serving permission changed.

The gold builder now applies the target-count correction centrally. Invalid
receiving-target history, shares, opportunity derivatives and affected career
aggregates remain unknown. Players, older seasons and independent scoring/yardage
outcomes remain present. A partially observed counter is not presented as a complete
total. Existing raw/enriched inputs and old research artifacts are unchanged.

The reused individual-stat evidence is accepted only after verifying equality of
all nine consumed tables against its original corrected gold release. A changed
consumed table requires new research; a new filename or timestamp cannot approve it.

## Everyday dashboard

- **Players:** production, opportunity, efficiency and scoring dispersion across
  the previous season, previous three seasons, captured completed career or current
  partial season. There are 839 current candidates, including players without NFL
  history. Missing values remain visible as unknown, never silently zero-filled.
- **Career profiles:** all captured careers, accepted college links, participation
  history, source gaps and provider tracking summaries. Historical cutoffs exclude
  future observations. Profile forecasts use the same published approval registry.
- **League:** actual standings, FAAB, ownership and recorded injury statuses. This
  path refreshes the league snapshot without reading any V1/V2/Adaptive board.
- **Season baselines:** separately labelled preseason estimates for 13 targets;
  the source of each reference estimate and prior sample size are explicit. CSV
  exports obey the same policy as the screen.
- **Evidence:** chronological target-specific errors, annual comparisons,
  season-bootstrap intervals, MSE guardrails, leave-one-season-out sensitivity,
  probability calibration and predictions outside earlier training outcome ranges.
  The September 24 update makes outcome units and preseason scope explicit, removes
  baseline self-comparisons, and adds passing-yard errors by prior passing workload.
  See the [screen changes and challenger audit](../research/qb_passing_forecasts_review_2026-09-24.md).
- **Research archive:** all individual-stat screens, fixed challengers, prior model
  screens and an inventory of older data/studies with dispositions and reasons.

Historical totals, mean/median, dispersion and efficiency are descriptions, not
claims about talent, medical risk or draft value. Efficiency retains its sample
denominator. Tracking summaries retain provider aggregates and missingness; their
underlying tracking play counts are unavailable. They do not need to improve
fantasy prediction to be useful descriptions.

## Evaluation and model status

The default Intelligence tab now serves **NextGen rankings** for remaining-season
and next-four-week points. The published release covers 841 candidates through
2026 Week 2. Roster/free-agent tables and current player profiles use the same
release. The [ranking report](nextgen_rankings_2026-09-23.md) records the protocol,
results, preserved iterations and refresh commands. These are point forecasts;
positional scarcity, weekly lineup decisions and acquisition prices require
separate validation.

All eight ranking scopes currently serve explicitly labelled reference forecasts.
Advanced profile and enriched challengers did not pass the final publication
gates against the stronger role-aware references. The earlier forecast plumbing
alone did not provide a current-season ranking; this release completes that
delivery without reopening the archived NextGen model.

`research/nextgen_system.py` writes its protocol before fitting. Observations begin
in 2001; cutoff-safe candidate/training rows begin in 2004. The first three candidate
years seed training, so evaluation spans 2007–2025. Every fold trains on all earlier
completed candidate years. The modern 2019–2025 view changes evaluation membership,
not the training window. No 2026 outcome is evaluated.

Targets cover passing/rushing/receiving yards, attempts, carries, receiving targets,
receptions, three opportunity-conditioned efficiencies, scoring appearances, the
probability of a scoring appearance, and league fantasy points. When a fold has no
trustworthy training labels, its candidates and outcomes remain in the saved
forecast table with null estimates and an explicit unfitted status. Coverage
separates observed labels, available predictions and matched evaluation rows. Scoring appearances
are explicitly **not medical availability**. Undefined efficiencies are null labels.

The serving reference is last-season observation, adjusted for nominal season
length for totals, or an earlier position/population mean when history is absent.
Appearance probability uses the earlier population frequency. A declared baseline
is not a claim of superiority or a solved role-transition model.

Ridge and histogram boosting are fixed profile/history-informed challengers.
Preprocessing and missing-value handling use only earlier training rows. All
retrospective comparisons remain exploratory; none automatically promote a winner.
The individual-stat screen retains its own fantasy-point outcome, conditional
addition/removal contexts, aliases and multiplicity correction. A screen pass is a
research candidate, not permission to use that statistic in arbitrary models.

Starting-status, medical-availability, development and waiver/trade claims remain
unapproved. Old four-week, lineup, power-ranking and draft/waiver valuations are
archived. There is no approved advanced role model or acquisition/lineup decision
model in this release. Current-season point rankings have their own evaluated,
explicitly labelled reference policy; they do not grant those decision claims.

## Archive and serving enforcement

`src/engine/data/nextgen.py` is the common eligibility gate. It checks input validity,
allowed use, target, position, population, horizon and open source incidents. The
published analysis binds exact gold, profile and research hashes. Tampered,
incomplete or mixed releases fail closed. Nothing selects a model because its
artifact exists or has the newest modification time.

The immutable `registry.json` distinguishes `quarantined`,
`revalidation_required`, `verified`, `shadow`, `archive`, `baseline`, and scoped
research findings. `data_release` and `study` entries record original paths,
manifest hashes when available, original status and exclusion reasons. Earlier
NextGen construction releases are also archived. Known defects are distinguished
from older or inconclusive work; age alone is not evidence that a result is false.

Archives are logical dispositions, not destructive moves. Original files, manifests,
frozen boards and study results remain at their reproducible paths. Old "accepted"
labels describe the old run's checks; they do not grant current serving approval.
A corrected rerun gets a new immutable version and a new registry decision.
An archived study can still contain independently verified observations used by
a current profile; its predictive claims do not inherit that descriptive approval.

The backend requires `scope=research` or `X-Analysis-Scope: research` for legacy
board, model, league-decision and research routes. Responses carry archive/research
headers. Normal analysis uses `/api/nextgen/*`; player profiles require a verified
analysis release and never read legacy forecasts without research scope. With an
open data incident, dependent measurements/forecasts are withheld; the compound
profile view is conservatively suspended until revalidation. Independent valid
measurements remain accessible through the measurement API.

The frontend consumes the policy catalog. Legacy components mount only after the
user explicitly opens an archive section. Export endpoints and direct API calls
cannot bypass model eligibility. New raw result columns cannot enter default
measurements without a registered definition. Catalog changes invalidate cached
data and policy together.

## Rebuild and publish

Version directories are create-only; use new names for subsequent builds. From
repository root, the installed workflow is:

```sh
.venv/bin/python research/data_pipeline.py gold \
  --version NEW_GOLD --enriched-version canonical_20260923_r4
.venv/bin/python research/data_pipeline.py products \
  --version NEW_GOLD --prefix NEW_PRODUCTS
.venv/bin/python research/nextgen_system.py \
  --version NEW_ANALYSIS --gold NEW_GOLD --profiles NEW_PRODUCTS_profiles \
  --evidence individual_stats_20260923_r4
.venv/bin/python research/nextgen_rankings.py \
  --analysis NEW_ANALYSIS --version NEW_RANKING_RESEARCH
.venv/bin/python research/publish_nextgen_rankings.py \
  --source NEW_RANKING_RESEARCH --version NEW_RANKING_DELIVERY --publish
```

The existing gold release may be reused when inputs have not changed. The evidence
reuse check will reject changed consumed tables. Rebuild the individual-stat
research first when its inputs change. Publication verifies all products before
atomically replacing the one catalog pointer, retaining the previous catalog in
`data/catalog_history`. After NextGen is published, a products-only publication
cannot silently drop the analysis policy. After rankings are published, a later
analysis cannot silently drop the ranking product either. The ranking publisher
selects the exact gold/profile/college/outlook dependencies of its source release.

Changes to approved scopes require a new immutable analysis release with scoped
supporting evidence and rationale. Do not edit a published registry or hand-add
an old model to a frontend selector. Historical rollback may restore a prior
verified catalog; an old pre-NextGen catalog will not reopen default model access.

## Verification

Regression coverage includes chronological feature construction and held-out-label
invariance; null/zero/undefined denominator handling; exact target/position/population/
horizon gating; incident suspension; direct and CSV access; artifact tampering;
publication downgrade rejection; profile forecast exclusion; and league observations
without a legacy board. Existing correction, profile, data and research tests remain
part of the offline suite.

```sh
.venv/bin/pytest tests research/test_individual_stats.py
cd web
npm run build
npm run lint
```

`web/tests/nextgen_smoke.py` verifies the current dashboard, profiles, evidence,
CSV download, archive labels, desktop/mobile layout and direct API rejection. It
uses local artifacts and an explicit league observation fixture so the browser test
does not refresh ESPN. Run local API/frontend servers and then:

```sh
.venv/bin/python web/tests/nextgen_smoke.py \
  --url http://127.0.0.1:5199 --api http://127.0.0.1:8011
```

Completed verification for the published release: **622 offline tests passed**
(35 network tests deselected); TypeScript/Vite build passed; targeted Python lint
and whitespace checks passed. Frontend lint has existing warnings in legacy/shared
components. The installed Node 21 runtime emits Vite's supported-version warning;
the build completed successfully.

Desktop/mobile browser checks passed against the exact published catalog, including
723 receiving-yard baseline rows exported through the guarded CSV endpoint, all
normal views avoiding legacy model requests, and explicit archive/direct-request
checks. The current registry permits 28 descriptive definitions and 13 reference
baselines; 1,585 other entries are excluded from everyday analysis. The final
historical table retains 8,279 unfitted candidate/model rows as null forecasts.
The immutable draft artifact still verifies, and product builds preserved all 13
protected artifacts.
