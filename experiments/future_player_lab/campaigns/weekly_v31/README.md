# Opening Drive v3.1

V3.1 adds past-only interval recalibration and a prospective paired forecast log.
It preserves the v3 point models, quantiles, standard deviations, scoring contract,
and published campaign bytes. This is a separately sealed research challenger;
it does not replace the production rankings or relabel adjusted endpoints as v3
distribution quantiles.

## Experiment

Run from the repository root in the locked environment:

```sh
.venv/bin/python -m experiments.future_player_lab.campaigns.weekly_v31.run
```

The default immutable run is
`experiments/future_player_lab/runs/opening_drive_v31_20261013_001`.
The review report is `data/outputs/opening_drive_v31_comparison_2026/index.html`.
For another experiment supply fresh `--output` and `--report` directories.
Existing run directories are never overwritten. No point models are retrained.

The designated player challenger uses position and fixed projected-points groups
(`<5`, `5–10`, `>=10`), with position fallback below 200 calibration rows.
Position-only calibration is a diagnostic control, not a candidate automatically
selected from evaluation results. Each season uses at most five earlier seasons
of saved out-of-time errors. The 2026 player calibrator uses 2021–2025 only and
stays fixed during prospective tracking. 2026 results never enter calibration.

For each 80%/90% interval, the calibration score is
`max(lower - outcome, outcome - upper) / max(width, 1 point)`.
The correction uses order statistic `ceil((n+1) * coverage)`, clipped below at
zero; applying it expands both endpoints by correction times the forecast width
floor. The final 90% interval encloses the 80% interval. Byes remain deterministic
zero. Sparse unsupported groups retain their original intervals and are labeled.

This is a width-normalized chronological adaptation of
[Conformalized Quantile Regression](https://arxiv.org/abs/1905.03222).
Repeated players, drift and reused development years mean no formal coverage
guarantee is asserted. Mean and full-distribution CRPS do not change. Evaluate
coverage, width, tail misses and interval score together; wider alone is not better.

Player evaluation covers 2021–2025, with position/workload/season breakdowns.
Direct lineup interval calibration uses original v3 joint draws and both original
artificial roster-selection designs. Lineup evaluation starts in 2022, because
2021 is the first available lineup calibration year. A 100-row minimum applies;
overlapping lineup rows are not independent samples. The 2026 lineup calibration
uses 2021–2025. K/DST uncertainty is not changed.

## Prospective capture and scoring

```sh
.venv/bin/python -m experiments.future_player_lab.campaigns.weekly_v31.prospective track
.venv/bin/python -m experiments.future_player_lab.campaigns.weekly_v31.prospective replay --capture-dir data/outputs/opening_drive_v31_prospective/forecasts/2026_06
```

`track` captures the first complete paired forecast and updates the scorecard.
`capture` and `score` are also available separately. `--snapshot`, `--registry`,
`--run` and `--log` allow isolated workflows. Models currently support 2026 only.

The existing campaign refresh command now tracks v3.1 once the default run exists:

```sh
.venv/bin/python -m experiments.future_player_lab.campaigns.rescore --due
```

Tracking also runs when v3 already has the current origin. Requesting
`--campaign opening_drive_v31` refreshes its v2/v3 dependencies first. Alternate
registries require an explicit `--v31-run`; use `--v31-log` for an isolated log.
This is integrated into refresh, not a separately installed timer. Run refresh
before the weekly deadline; late missed weeks are never backfilled as prospective.

The deadline is **00:00 UTC on the first NFL game date**, conservatively before
kickoff and consistent with the existing league forecast log. Capture checks:

- accepted snapshot timestamps and matching completed-week cutoff;
- no forecast-week labels or game scores;
- all four positions, a common dated feature matrix and dataset receipt;
- frozen baseline weights, dependencies and source checksums;
- exact replay of published v3 player forecasts;
- the actual wall clock both before work and before publication.

Each week seals inputs, baseline payloads and weights, calibration, source,
player forecasts, artificial lineup membership and available actual league
offense rosters. CSV exports provide inspectable point forecasts and both sets
of interval bounds. The original v3 payloads retain their historical metadata;
the new capture independently establishes pregame timing. Existing weekly
captures are immutable. Repeated identical requests are no-ops; a different
challenger cannot replace a locked origin.

Actual league offense rosters come from the current local league snapshot and
frozen identity mappings. Missing/ambiguous identities and unsupported roster
shapes are disclosed rather than silently dropped. Only the studied eight-slot
offense shape is supported. Applying synthetic-lineup calibration to actual
teams is a provisional transfer, evaluated separately. Later substitutions do
not rewrite the captured roster; this measures the saved forecast for that roster,
not necessarily the official final lineup. K/DST and official ESPN totals are
outside this scorecard's target.

The scorecard uses accepted completed-game labels under the original offensive
scoring contract. Unobserved player scores remain missing; a lineup remains
pending until all captured members have labels. Known byes are zero. Both versions
use identical populations and point forecasts. Metrics are broken down by week,
position, projected workload, lineup design and stacking. No automatic promotion
occurs. Every score revision retains its input panel, snapshot and capture hashes.

Latest scorecard: `data/outputs/opening_drive_v31_prospective/scorecard/index.html`.
Revisions: `data/outputs/opening_drive_v31_prospective/evaluations/`.

## Verification

```sh
.venv/bin/python -m pytest experiments/future_player_lab/tests/test_weekly_v31.py experiments/future_player_lab/tests/test_weekly_v3_rescore.py
```

Tests cover temporal isolation, finite-sample ranks, fallback support, bye handling,
nested intervals, point/quantile preservation, timing failures, source tampering,
exact offline replay, immutable publication, missing outcomes, captured league
membership and refresh integration even when v3 itself is already current.
