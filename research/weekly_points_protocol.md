# Dedicated NextGen weekly points protocol — v1

Fixed before the first fit in this work. Target: league-exact QB/RB/WR/TE points
in the next calendar NFL regular-season week, including zero production for known
candidates who do not score. A scheduled bye is known zero. Unknown teams/schedules
are excluded, not silently treated as active. K/DST retain separately labeled
prior-score references; their targets are not in the audited offensive weekly panel.

Candidates and core features reuse NextGen's cutoff-safe player panel: preseason
candidates plus players observed before the forecast week; previous-season and
current-prefix scoring, touches, targets, passing volume, snap share, age and draft
information. Add previous-game/three-game scoring and workload, home/away, and the
opponent's earlier positional points allowed. No current-week production, final
season aggregates, market/provider forecasts, future injuries or future lineups
enter training features. Historical source revisions are a remaining limitation.

Fit each position independently, all available earlier seasons from 2004 onward.
Season folds start in 2008. Fit once per season on strictly earlier seasons;
features update before each week but neither fitting nor recipe selection sees
that season's outcomes. References: prior scheduled-game scoring rate, prefix
scheduled-game rate, four-game-prior blend and trailing-three observed games.
Challengers: median-imputed/scaled ridge (alpha=100), histogram gradient boosting
(100 iterations, learning_rate=.05, 15 leaves, min leaf=40, l2=10, fixed seed).
Use only core/prefix/opponent predictors; avoid large loosely dated feature blocks.
Choose the reference and challenger using earlier out-of-season predictions,
requiring at least three prior evaluation seasons for challenger selection.
Minimize equal-season MAE; do not inspect current-season outcomes to select.

Primary report: 2019–2025; full-history guardrail: 2011–2025. Report annual and
pooled MAE, RMSE, bias, weekly positional top-K point capture (QB10/RB20/WR30/TE10),
and observed-workload and rookie cohorts. Compare the chronological challenger
selection policy against the chronological reference selection policy. Promotion
requires >=2% and >=0.10 point equal-season MAE improvement, a positive lower 95%
season-bootstrap bound, one-sided exact season-sign-flip p-values with Holm <=.05
across four positions, non-worse MSE/top-K capture, nonnegative full-history gain,
and no observed-workload or rookie MAE degradation > max(.25 points, 5%).
If any gate fails, serve the dedicated weekly reference selected using past folds.
Do not tune/retry hyperparameters based on these evaluation results.

Predict 2026 Weeks 1 through the currently available next week using models and
recipes selected with data only through 2025; current-year targets are attached
only for retrospective scoring. Current-season replay is a reconstruction, not a
claim of forecasts published at the time. League replay is conditional on recorded
actual lineups; no lineup optimization. Do not infer historical ownership from
current rosters. Missing K/DST prior scores withhold complete team forecasts.

Save immutable panel shards, fold predictions, current-season predictions, report,
protocol and source-code snapshots, bound to the exact analysis/gold/profile refs.
Serve only a verified product matching the current analysis release and cutoff.
Keep timestamped prospective forecast captures separate from retrospective evidence.
