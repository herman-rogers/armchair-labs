# Opening Drive v3.2: injury, participation and workload

V3.2 starts from the saved v3 point models. V3.1 research and its automatic
prospective tracking are paused at the user's request; all prior runs and
captures remain unchanged. No fitted v3.1 point model is used here.

**Release requirement:** v3.2 must improve v3 while retaining its functionality,
including weekly forecasts, variance/distribution outputs, component forecasts,
longer-horizon views, rankings and API behavior. Returning to v2 with an adjustment
is not the intended architecture. See the
[version-lineage contract](../../../../docs/operations/model-campaigns.md#version-lineage-and-retained-functionality).

**Known integration gap:** the weekly injury/workload models do build on v3, but
the current season/remaining projection bridge is anchored to v2 and exposes only
points in those current views. That bridge does not complete the requested v3-based
extension or preserve all component-output functionality. The preview also withholds
K/DST uncertainty and mixed-timing lineup ranges. These are current limitations,
not a template for future versions; arithmetic checks alone do not resolve them.

## Run

```sh
.venv/bin/python -m experiments.future_player_lab.campaigns.weekly_v32.run
.venv/bin/python -m experiments.future_player_lab.campaigns.weekly_v32.capture --week 6
.venv/bin/python -m pytest experiments/future_player_lab/tests/test_weekly_v32.py
```

Sealed run: `experiments/future_player_lab/runs/opening_drive_v32_20261013_002`.
Report: `data/outputs/opening_drive_v32_2026/index.html`.
Fresh `--output` and `--report` paths are required for another experiment.
Initial `_001` is retained. Revision `_002` uses the frozen identity crosswalk,
rather than membership in any season's snap observations, to establish support
for missing-snap labels. This prevents later snap appearances from determining
whether an earlier label is observable. Candidate design is unchanged.

## Completed historical results

The chronological selector reduced pooled point RMSE versus the unchanged v3
predictions on 61,046 player-weeks from 2021–2025:

| Position | Original early-week cutoff | One hour before kickoff |
| --- | ---: | ---: |
| QB | 0.33% | 3.93% |
| RB | 0.36% | 4.18% |
| WR | 0.30% | 3.55% |
| TE | 0.69% | 3.21% |

The pregame selector improved RMSE in each of the five evaluation seasons for
every position. The simple same-game Out-to-zero control reduced pooled RMSE
by 1.36–2.32%, so the full models add improvement beyond that rule. Explicit
workload transfers were not uniformly beneficial: the learned teammate workload
model beat both fixed transfer strengths for QB in the pooled pregame comparison.
These are historical research results on reused development years, and the
pregame arm benefits from later information. They are not prospective validation.

The first immutable current capture is
`data/research/opening_drive_v32_injury_captures/20261013T193919Z`: 32 team entries,
800 status observations, 798 exact identity matches and two unresolved identities.
These are provider observations usable from collection time, not verified
game-week active/inactive designations.

Verification: 58 related tests passed; the subsequent pause-test refinement passed
its 24-test subset. The sealed run and capture verify, all 1,490 protected files
remain unchanged, and v3 baseline predictions and labels match exactly. All 16
methods use identical rows within each horizon; 32,616 redistribution budget rows
pass conservation checks. The report passed desktop/mobile browser checks.
`data/outputs/opening_drive_v32_2026/verification.json` records the final checks.

## Source audit and timing

The original v3 panel has 3,469 2026 player-week rows for Weeks 1–5 and zero known
dated injury reports. Its injury source contains 2025 and 2026 records with
missing modification timestamps. These remain excluded, rather than assigning
their game dates as publication dates. A separate sealed archive contains
corroborated 2025 report rows using the later of article publication/update time.
Both historical sources are retrospective reconstructions, not original
publication-vintage captures.

The early-week arm uses v3's cutoff. The pregame arm uses one hour before each
game, retaining the original baseline and other features. Thus the later arm
has an information advantage over v3, explicitly labeled throughout. Injury
matching requires the exact player, team, season and week, a timestamp no later
than cutoff, and a seven-day maximum age. Previous-week designations are never
treated as current-week reports. Same-time conflicts are unknown. Practice
participation and game status are separate features; missing rows are not health.

Snaps are joined through source IDs with ambiguous identities withheld. Ambiguous
player-week source rows are quarantined, with counts in the source audit. A
missing snap row becomes zero only with a captured team game, known source
identity, a completed point label and no box appearance. Otherwise it remains
unknown. The participation target is offensive snaps, offensive opportunities
or nonzero points; it is **not** official active-roster status or a medical label.

Lagged workload uses earlier scheduled weeks only. Historical box/snap features
assume completed earlier-week data were available by the next forecast week;
original publication vintages are not reconstructed. The source timestamps,
coverage and hashes are saved separately from model scores.

## Models and evaluation

Fifteen fixed candidates compare unchanged v3, an Out-only rule, direct point
regression, participation/conditional-points mixtures, explicit workload
factorization, two redistribution strengths and half-strength point blends.
Each learned family has no-new-injury, own-injury and own-plus-teammate variants.
The controls still have v3's historical roster features; roster state is not
relabeled as game-day availability.

Fits use the preceding five available seasons with v3 means from 2015 onward.
2019–2020 supply the first chronological validation forecasts. Evaluation is
2021–2025. A selector minimizes equal-year RMSE on the two earlier validation
years, separately by position, and can retain v3. All years have been used in
prior development, and the v3 base policies were originally selected in 2018–20.

The workload factorization estimates participation probability, conditional
attempts/carries/targets, then uses a linear conditional point response evaluated
at those predicted workloads. Participation is applied once. A linear response
avoids interpreting a nonlinear model at mean workload as the expectation of
that nonlinear model. The linear approximation itself is evaluated, not assumed
to beat direct regression.

Teammate features aggregate **other players'** pregame usage and report flags.
Explicit redistribution transfers 50% or 100% of the predicted shortfall versus
the no-injury expected-workload budget. Recipients must have predicted
participation at least 50%; allocation is weighted by original expected usage
and updated participation probability. Budgets cover the v3 eligible player
pool, not the entire team roster. Unassigned volume is recorded and transfer
cannot exceed the shortfall. Transfer calculations use unconditional counts,
so participation is not multiplied twice.

Fresh same-game Out designations set injury-aware predictions to zero. The
Out-only control isolates this rule from learned improvements. Current provider
tags are not automatically treated as such official game-week reports.

Scores include point RMSE, MAE, bias, weekly top-10 QB/TE and top-20 RB/WR
realized points, participation Brier/log loss, and conditional/unconditional
workload errors. Ranking tests use the same eligible pool, not actual league
rosters. Bootstrap intervals resample five whole seasons and are descriptive,
without multiple-comparison adjustment. Cohorts are fixed within each horizon;
report-based cohort membership can differ between horizons.

## Going-forward injury collection

The collector retrieves the public ESPN all-team injury endpoint and retains
the exact response, collection timestamp, provider update timestamp, source
identity match and status separately. It requires all 32 team entries. Each
capture is immutable. A current status is usable only from its collection time;
an older story timestamp is never used to backdate it. Missing players are not
inferred healthy, and provider status is not an official active/inactive list.
Repeated captures can support future observation histories; no timer is installed.

Current observations are not backfilled into historical folds. Repeated dated
practice reports and official inactive/active evidence remain the major source
gap, especially for early-week forecasts. No full practice trajectory or
within-game injury-exit reconstruction is claimed.

The study preserves all v3/v3.1 artifacts, saves fold and 2026 research models,
and does not promote an API model, change scoring, or implement a joint lineup
simulator. K/DST and medical injury diagnosis are excluded.

Upstream reference: [nflverse data availability](https://nflreadr.nflverse.com/articles/nflverse_data_schedule.html).

## Serving preview and prospective comparison

`weekly_v32.integration` seals the existing point weights, the locked selection
policy, baseline replay models and new uncertainty artifacts. No point models are
refitted or reselected during integration. The application exposes
`opening_drive_v32` as **Opening Drive v3.2 preview**, selected as the application
default by the user on 2026-10-13. V3 remains available for comparison. Since
2026-10-14, the eight current season/remaining-season forecasts use a
[shared future-week projection curve](../horizon_projection/README.md), including
the saved v3.2 weekly prediction. Season totals add known actuals; averages and
rankings use the resulting totals. Original preseason forecasts and prior bundles
remain preserved. The historical longer-horizon accuracy comparison is mixed.

```sh
.venv/bin/python -m experiments.future_player_lab.campaigns.weekly_v32.integration
.venv/bin/python -m experiments.future_player_lab.campaigns.weekly_v32.shadow track
.venv/bin/python -m experiments.future_player_lab.campaigns.weekly_v32.shadow score
.venv/bin/python -m experiments.future_player_lab.campaigns.weekly_v32.shadow replay \
  --capture-dir data/outputs/opening_drive_v32_shadow/forecasts/2026_06/early_week
```

The first command requires a fresh output path for another version. Tracking is
also called by the usual campaign refresh, even when the v3 origin already exists.
`--no-capture` reuses archived inputs without network collection. Alternate
registries are isolated from automatic local tracking.

The first early forecast locks before the first kickoff and uses injury evidence
no later than the original v3 feature cutoff. Per-game updates can lock in the
90–60 minute pregame window, before the historical experiment's one-hour cutoff.
They reuse the original weekly baseline/population/workload history. Both teams
must have official weekly report tables collected within 36 hours. The official
NFL page must identify the requested season and week; redirects to an older week
are archived as unusable. Missing players are not presumed healthy. Provider
status tags are archived separately and cannot activate the later model.

Missing late evidence retains the locked early forecast. Completed games never
receive a reconstructed pregame capture. First captures cannot be overwritten;
new API revisions retain the underlying captures and archive the prior bundle.
Each origin binds its own lineup calibration. The scorecard only joins completed
accepted outcomes to the locked players and offense rosters, keeping missing
labels pending. Offensive participation probabilities are diagnostic model
outputs, not confirmations of official active-roster status.

The uncertainty integration uses 256 mean-centered empirical error samples by
position, horizon and fixed point bins (2, 5, 10 and 20). A cell with fewer than
100 errors falls back to its position pool. Calibration always precedes the
forecast season; 2021–22 seed the 2023–25 evaluation. Historical lineup comparisons
re-estimate both versions' dependence from the available earlier evaluation-year
errors; these are matched comparisons, not reprints of the original v3 lineup
study. The live v3 comparison retains its sealed dependence artifact.

Early-week player ranges gain coverage with some extra width. Early-week
artificial lineup RMSE is slightly worse; pregame lineup RMSE and interval scores
improve. These reused historical years do not validate live performance. Actual
offense lineups are captured separately when a usable local league snapshot is
available. K/DST uncertainty and mixed-timing lineup ranges are withheld in the
preview; player ranges and point totals remain available.

Review: `data/outputs/opening_drive_v32_shadow/index.html`. The prospective
scorecard is under `scorecard/`, with immutable evaluations under `evaluations/`.
There is no automatic default promotion. V3.1 stays paused.

Numerical inference files and their dependencies must match the sealed serving
contract exactly. Collection/publication orchestration can receive reviewed fixes
without invalidating locked feature-matrix replays; new captures record the
runtime source hashes. The 2026-10-13 timestamp merge fix is recorded under
`data/outputs/opening_drive_v32_shadow/maintenance/20261013_timestamp_update`.
It preserves the first capture and all point/distribution inference code.

The user service templates in `deploy/opening-drive-monitor.{service,timer}` target
this workspace under `~/dev/armchair-labs`. The timer checks every 15 minutes while
the local user service manager is running. It does not wake a powered-off machine
or recover missed pregame windows. Inspect with
`systemctl --user status opening-drive-monitor.timer`; stop collection with
`systemctl --user disable --now opening-drive-monitor.timer`. A config status other
than `shadow_preview` also stops tracking.
