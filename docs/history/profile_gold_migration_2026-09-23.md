# Player profiles on corrected gold and NextGen

> Historical implementation record. Release IDs, results, commands, and interface descriptions reflect the recorded change; use the [operations guides](../operations/README.md) to run the system today.

Player profiles now combine corrected observed histories, NFL Next Gen Stats
season summaries, and the forecasts permitted by the current NextGen registry.
The current catalog selects the data and products together; a broken or mismatched
dependency produces an error instead of substituting an older model.

## Release

| Dependency | Version |
| --- | --- |
| Corrected gold | `canonical_nextgen_20260923_r1` |
| Profiles | `nextgen_products_20260923_r2_profiles` |
| Analysis and forecast policy | `nextgen_system_20260923_r4` |
| College dependency | `nextgen_products_20260923_r2_college` |
| Outlook dependency | `nextgen_products_20260923_r2_outlook` |

The profile release contains **4,931 players**, **149,590 weekly observations**,
and **14,077 player-season tracking rows**, including seasons with unknown tracking
values. Weekly observations exactly match the already corrected profile release.
Invalid target counts remain null, and no recorded reception count exceeds its
non-null target count. The observation cutoff remains **2026 Week 2**; this update
does not fetch a newer NFL snapshot.

## Profile changes

**Next Gen Stats** shows the provider's regular-season aggregates copied from
gold's `nfl_player_seasons`. QB profiles show completion percentage over expected
(percentage points); RB profiles show rushing yards over expected per attempt;
WR/TE profiles show average separation and yards after catch over expected per
reception. Other recorded position-specific metrics remain visible for players
with position changes. Definitions and units accompany each metric.

Captured passing and receiving coverage spans **2016–2025**. Rushing yards over
expected coverage spans **2018–2025**. Coverage varies by player. Missing values
remain unknown, zeros and negative residuals remain valid, and season ratios are
not averaged into a career score. The table reports available measures, not a
tracking play count: those sample counts are not retained in the gold summaries.
There are no current-season tracking observations in this capture.

Tracking summaries are shown only for completed seasons at the requested cutoff.
A midseason 2024 profile cannot show 2024 full-season tracking; the completed 2024
view can. These are descriptive historical reconstructions with the existing
source-vintage limitations, not original publication-time snapshots.

**Forecasts** now reads the published analysis release and applies the same
validity, position, outcome, horizon, allowed-use and incident rules as NextGen.
The current release permits reference baselines; research challengers remain
archived. Each row shows the forecast season, information cutoff, estimate, units,
and a completed outcome only when the selected full season is complete. Forecasts
dated after the selected NFL week are withheld. Week-zero requests expose no
same-season forecast because they do not specify a calendar-day cutoff.

Default profiles never load saved legacy preseason, college, or four-week model
predictions. Explicit `scope=research` retains archive access. The overview's
current opportunity uses observed partial-season workload, without an old outlook
forecast. Profiles and their forecasts must match the same gold and profile
manifest references. Archived data and models remain preserved for reproducibility.

## Validation

- All **608 offline tests** passed, with network tests deselected.
- Targeted regressions cover tracking missingness, zeros, negative residuals,
  duplicate keys, future-season exclusion, pending outcomes, forecast cutoffs,
  position restrictions, unapproved models, incidents, and dependency mismatches.
- All profile tracking values reconcile exactly to gold. Weekly profile records
  reconcile exactly to the corrected predecessor.
- QB/RB/WR/TE profile forecasts match the NextGen endpoint's permitted baseline
  outputs. Legacy prediction arrays and the old current outlook remain absent.
- Targeted Ruff and mypy checks pass. The frontend production build and lint pass
  with existing Node-version and unrelated lint warnings.
- `web/tests/profile_nextgen_smoke.py` passed against both the candidate and the
  published API: four positions, historical selection, tracking, forecasts, and
  desktop/mobile layout. Published profile observations, tracking, and feature
  tables match the validated candidate exactly. Live API checks reject legacy
  boards and unapproved challengers from normal analysis. Publication uses the
  standard atomic catalog publisher after verification of every dependency.

The schema-2 profile manifest includes `tracking_seasons.parquet` and its hash.
Schema-1 archived profiles remain readable, but do not gain synthetic tracking
data or an automatic legacy-model fallback. Future releases can use the existing
`data_pipeline.py products` command to rebuild the product set, build a matching
`nextgen_system.py` analysis, and publish them with `data_pipeline.py publish
--analysis VERSION`.
