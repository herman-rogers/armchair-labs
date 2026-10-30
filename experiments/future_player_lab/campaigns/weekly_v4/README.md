# Opening Drive v4 candidate

V4 builds on v3.2 and remains a selectable **candidate**. V3.2 stays the application
default. The first immutable v4 forecast was captured on 2026-10-27 for Week 8:
729 players, including 623 non-bye players. The separate retraining arm consumes
686 genuinely captured earlier current-season forecast outcomes. No future results
are claimed: all Week 8 outcomes were pending when this implementation shipped.

## What changed

- Weekly point means preserve v3.2's v3-based injury/workload predictions.
- A new LightGBM residual extension predicts later scheduled games from the
  fixed-origin v3 forecast and past-only features. It does not anchor season totals
  to v2. Current-week injury adjustments do not become permanent absences.
- Direct component models restore all original passing, rushing and receiving
  targets across weekly, remaining and projected season views. Points and
  component-derived points are separate estimates, both exposed. Component totals
  reconcile across horizons; original preseason forecasts remain labeled at origin zero.
- Weekly uncertainty candidates preserve means and select empirical-error scales
  using earlier-year CRPS. Coverage, width, interval scores, tails and cohorts are
  evaluated separately. The selector retained v3.2's distributions in this run.
- Longer-horizon distributions use errors of complete fixed-origin totals, retaining
  observed dependence between weeks. They do not add weekly interval widths.
- A separate residual Ridge retraining candidate uses the previous three seasons
  and earlier completed current-season weeks. Fixed transforms, regularization,
  weighting and bounded adjustments are declared before evaluation. It is compared
  with frozen v3.2, calibration-only v4 and a prior-seasons-only residual update.
- Shared refresh now has an hourly persistent systemd timer with retries. Injury
  collection and paired v4 tracking run every 15 minutes. `/api/freshness` and the
  application warning distinguish observation, forecast and injury freshness.

## Evaluated results and limitations

The final horizon run evaluates 53,950 player/origin forecasts across origins 1–16
in 2021–25, plus 769,720 component forecasts. These years were previously used in
development; this is retrospective evidence, not an untouched test.

Exact matched Week 2 comparison against the existing v3.2 bridge:

| Position | V3.2 remaining-point RMSE | V4 RMSE |
| --- | ---: | ---: |
| QB | 60.83 | 60.73 |
| RB | 44.67 | 45.11 |
| WR | 41.10 | 42.43 |
| TE | 29.60 | 30.73 |

V4 improves against fixed-origin v3 persistence across most positions and improves
its aggregate total CRPS. It does **not** establish an improvement over the shipped
v3.2 bridge: the matched bridge comparison regresses for RB/WR/TE, and its newly
calibrated research-control distributions also outperform v4 in aggregate CRPS.
The original bridge did not serve long-horizon distributions. Later origins lack
exact historical shipped-bridge controls; do not relabel the v3 persistence control.

Weekly uncertainty changes tie the retained v3.2 control. Residual retraining has
mixed point results and worse aggregate CRPS; it is not used in published v4 means.
Bye-origin horizon fallback and transfer of early-week total residual calibration
to pregame updates are functional but not separately historically validated.
Inherited K/DST and mixed-timing lineup uncertainty limitations remain explicit.

## Artifacts and commands

Final sealed directories under `experiments/future_player_lab/runs/`:

- `opening_drive_v4_horizons_20261027_003`
- `opening_drive_v4_uncertainty_20261027_002`
- `opening_drive_v4_integration_20261027_001`

Earlier development directories remain preserved; they are not the current fits.
Numerical inference source hashes, package versions, fitted models and frozen inputs
are checked during replay. Do not alter receipts or old forecasts when changing inference.

```sh
# A fresh, isolated reproduction; never overwrite a sealed run.
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 POLARS_MAX_THREADS=4 \
  .venv/bin/python -m experiments.future_player_lab.campaigns.weekly_v4.run \
  --output experiments/future_player_lab/runs/NEW_V4_EXPERIMENT

# Normal refresh, including the v4 candidate after v3.2.
just data-refresh

# Existing sealed v4 inference, without recollecting v3.2 inputs.
.venv/bin/python -m experiments.future_player_lab.campaigns.weekly_v4.prospective track
.venv/bin/python -m experiments.future_player_lab.campaigns.weekly_v4.prospective replay \
  --capture-dir data/outputs/opening_drive_v4_shadow/forecasts/2026_08/early_week

# Complete prospective evaluation and human-readable review.
.venv/bin/python -m experiments.future_player_lab.campaigns.weekly_v4.evaluation
.venv/bin/python -m experiments.future_player_lab.campaigns.weekly_v4.report
```

The review is `data/outputs/opening_drive_v4/index.html`. Immutable paired captures
and separate matched v3.2 control receipts live under
`data/outputs/opening_drive_v4_shadow/{forecasts,controls}`. Controls must be captured
before the same deadline and match the frozen baseline; missed captures are recorded
as unavailable rather than reconstructed. Prospective evaluations retain original
rosters and joint distributions, keep missing labels pending, and deduplicate
player-weeks across early/pregame captures. Weekly components score after that week;
whole-season and remaining-season totals wait until the regular season is complete.

New evaluation/reporting modules append separately checksummed evidence without
modifying the original sealed inference source or first capture. The existing
`prospective score` command is the original weekly scorecard; `evaluation` adds
component/horizon controls and the deduplicated release review.

## Release and promotion

Publication requires all 12 position/horizon models, explicit v3.2 lineage, live
components, matching current-week points/components, reconciled totals and averages,
and finite mean-preserving distributions. The default remains v3.2 even when these
checks pass. Historical regressions require explicit review; prospective promotion
also requires at least four completed weeks and 2,000 unique paired observed
player-weeks, matched accuracy/distribution review, and manual approval.
There is no automatic promotion or retraining-driven default change.

```sh
.venv/bin/python -m pytest experiments/future_player_lab/tests/test_weekly_v4*
systemctl --user status armchair-refresh.timer opening-drive-monitor.timer
```
