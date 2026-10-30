# Opening Drive projections across horizons

**Status: interim bridge; does not meet the intended model-lineage requirement.**
The user clarified on 2026-10-14 that v3.2 must extend v3 and retain its functionality.
This implementation instead returns toward v2 season estimates and makes current
horizon outputs points-only. Its totals reconcile, but that does not make it a
complete v3-based model extension. Do not reuse this architecture as the default
pattern for future versions. See the
[version-lineage contract](../../../../docs/operations/model-campaigns.md#version-lineage-and-retained-functionality).

V3.2 now feeds its weekly point forecast into remaining-season totals, projected
full-season totals, averages and rankings. The shared projection curve and the
publication gate make this a release requirement for future Opening Drive versions.
V2/v3 historical comparison bundles stay unchanged.

## Definition

At a single observation cutoff, the current game uses the saved weekly forecast.
Later games blend the locked early-week signal toward a same-origin v2 remaining
per-game anchor. The signal weight is `2 ** (-calendar_weeks_ahead / half_life)`;
zero half-life keeps the anchor after the current week. A bye has no weekly signal
to extrapolate and retains the anchor for later games. Current-week official
report and teammate injury effects are not extrapolated: affected early signals
use the longer-term anchor. Pregame changes update only the current game.

Remaining points sum that future schedule. Projected full-season points add actual
points through the accepted completed week, using the original scoring contract.
The season average divides by the current known team's full scheduled-game count;
past actual points follow the player across teams. Future transfers and missed
games from long-duration injuries are not separately predicted. The old preseason
forecast remains available at origin zero for draft comparisons; it is not relabeled
as a new v3.2 preseason fit. Current season projections use the current origin.

Each saved row includes the future-week curve, anchor, actual points, denominator
and policy. The API exposes this as `projection`; expand a Rankings row to inspect
the weekly estimates. Long-horizon projections are points-only; old component-stat
forecasts are preserved in the archived prior bundle, not presented as newly modeled
v3.2 components. Long-horizon intervals are withheld because weekly errors are dependent.

## Evaluation and reproduction

```sh
.venv/bin/python -m experiments.future_player_lab.campaigns.horizon_projection.research \
  --output experiments/future_player_lab/runs/NEW_HORIZON_RUN
.venv/bin/python -m pytest experiments/future_player_lab/tests/test_horizon_projection.py
.venv/bin/python -m experiments.future_player_lab.campaigns.weekly_v32.shadow track --no-capture
```

The sealed policy is `opening_drive_v32_horizons_20261014_002`; initial exploratory
run `_001` remains preserved. The shared curve source must match the evaluated
policy before publication. Research evaluates the exact archived v2 ensemble at
the Week 2 cutoff, retaining its full candidate cohort; unmatched weekly rows
retain that control forecast. It compares v3 and v3.2 with identical decay choices.
No later weekly forecast or future feature enters a fixed-origin curve.

Half-lives 0, 2, 4, 8 and 10000 calendar weeks are compared. Earlier seasons alone
select the lowest equal-year MSE for each position. 2021–22 seed evaluation in
2023–25, and all five earlier seasons select the 2026 policy. The years have been
used in previous development. Later live cutoffs extend this tested Week 2 contract;
they are not claimed to have separate historical validation. Adding known actuals
does not change total prediction error, so the same errors apply to current
full-season totals, but this is not a new preseason accuracy test.

The selected 2026 half-lives are QB/WR 0, RB 4 and TE 2. Historical remaining-season
RMSE improves for RB, is near unchanged for WR and slightly worse for QB/TE.
The report retains MAE, bias, ranking outcomes, per-year scores and all alternatives.
Coherence does not establish uniform accuracy improvement.

## Publication and refresh contract

The timer and normal refresh both call the shared publisher. It verifies the
accepted snapshot, requires all four long anchors at the same origin, and rejects
anchors made after the locked early capture. All 725 initial players have exact
same-origin anchors; weekly-only newcomers can use an explicitly recorded locked
history proxy. Missing/unusable inputs stop publication instead of retaining stale
season totals under an updated weekly bundle. A policy or anchor hash change also
invalidates the publication no-op check.

The schedule follows the locked weekly team's identity so all views agree on byes.
The v2 anchor may carry a different roster-source team. Publication records those
team and denominator differences explicitly, retaining the anchor's per-game
estimate and applying it to the weekly team's schedule. At the initial origin,
11 players have different remaining-game counts between these two sources.

Publication and replacement require all 12 position/horizon models at one current
cutoff, matching weekly populations and capture receipts, matching current-week
points, reconciled totals, and correct averages. New versions must retain this
contract even if they replace the extrapolation method. Legacy comparison releases
are exempt so their historical artifacts can be reproduced unchanged.
