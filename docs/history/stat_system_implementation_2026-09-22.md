# Stat-system implementation — 2026-09-22

> Historical implementation record. Release IDs, results, commands, and interface descriptions reflect the recorded change; use the [operations guides](../operations/README.md) to run the system today.

Implemented in the `stat-system-review` worktree. The baseline commit records the
reviewed working tree; the original checkout is unchanged. No research model has
been promoted and the frozen Adaptive predictions have not been changed.

## Preserved forecasts

`data/static/draft_2026/` contains the exact V1 JSON, league configuration, scoring,
overrides, projection assumptions, and a SHA-256 manifest. V1 API and league reads
prefer this verified archive. `patron board` copies its exact bytes to both V1
compatibility filenames. A regression pins the original board digest independently
of the manifest. V1 retains its original production-row PPG denominator.

Adaptive remains an explicitly experimental, frozen season-points forecast.
`patron board --experiments` rebuilds its comparison view; ordinary production
builds do not require its snapshot or manifest. `prospective_grading_2026.json`
preserves the original ranking settings and legacy evaluation population, with a
pinned digest. Changing the mutable metric report cannot change that grade.

## Production boundary and publication

Every configured model now declares a population: returner, rookie, or all.
Production PPG and games train and predict only returners. Live fitting applies only
`fitted_ppg`, `fitted_games`, and their season-points product. The existing rookie
research models remain report-only; unmodeled draft players use the declared market
rank-matching fallback where available.

The report exports `production_model.json` separately. Its production fingerprint
excludes research specs. The artifact records training-input/population and scoring/
feature-definition digests, information cutoff, coefficients, and training seasons.
Boards receive a content-addressed manifest tying their bytes to production models,
scoring, overrides, and relevant implementation files. JSON publication is atomic.
The status API and UI expose unverified, stale, and degraded artifacts.

## Evaluation

Ranking outcomes now remain fixed when a ranker lacks scores. Missing predictions
reduce coverage and cannot remove actual winners from the ideal top K. Pairwise
common-player ECR comparisons are separate from complete retained-pool evaluation.
Both report player counts and coverage. The deployment test invokes the same
position/overall ranking and three-neighbor market fallback as production. Historical
manual overrides are excluded because there is no dated archive for those folds.

Newly constructed folds union historical, rookie, and market-only candidates before
joining outcomes. Reanalysis of older retained folds cannot recover omitted players;
the report marks those pools as legacy and requires a fold rebuild before claiming
complete market-only coverage. Refitting retained folds is useful corrected evidence,
not a promotion test with a newly reconstructed universe.

## Forecast and decision semantics

The canonical contract exposes active-game PPG, expected games, season points,
draft VOR, source, cutoff, support, and uncertainty status. Existing API columns
remain compatibility fields. Board and roster tables display the fitted PPG/games
underlying season points. Market fallback season points equal borrowed PPG × games.
V2 historical PPG explicitly uses participation-observed active games.

Adaptive publishes its frozen season total and leaves its unidentified PPG/games
components null. Its retained V2 weekly simulation is explicitly labeled a separate
estimate. All selectable systems remain historical/preseason views.

Waiver replacement is the best freely available player at each position. The best
free agent has zero wire VOR; weaker alternatives have negative wire VOR. The wire
also reports the gain to a legal roster lineup from adding a player, before choosing
a drop, on the selected season-equivalent scale. It is not a current-week forecast
or a complete add/drop optimizer.

Sample support is labeled a heuristic, not a probability. Residual intervals are
calibrated on earlier out-of-fold residuals and checked in later seasons by position
and population. They remain report-only; calibrated live tiers are not claimed.
A separate `candidate_games_nonredundant` removes the duplicate availability input
without changing the incumbent or frozen forecast. The existing availability-on-VOR
scenario statistic is explicitly labeled experimental and uncalibrated.

## Reproduction

Run `uv run patron production-report` to refit only approved models from retained
inputs without research fits. `uv run patron metric-report --reanalyze` refits the
full research comparison separately. Then run
`uv run patron board --experiments` to publish matching boards and manifests. A full
`uv run patron metric-report` reconstructs the history/rookie/market draft universe.
Research remains separate from approval: corrected evidence and the untouched 2026
prospective grade are prerequisites to choosing another production model.


## Regenerated evidence and verification

The production-only refit used retained 2004–2026 fold inputs. All 238 pending 2026
rookies have null production PPG and games, confirming the population boundary.
On 2019–2025 retained outcome pools, fitted season points have 59.52% mean overall
hit rate. Executing the automatic policy with market fallbacks gives 60.24%; its
annual retained-pool coverage ranges from 83.1% to 93.1%. On common players against
overall ECR, mean hit-rate lift is −3.81 percentage points. These are different
questions and populations; neither result establishes market outperformance.
There are 71 position/season residual-interval calibration records. No calibrated
live uncertainty claim or new model promotion follows from them.

The worktree boards were rebuilt entirely from copied caches with socket connections
disabled. The first build correctly rejected a newer August 30 depth chart against
the artifact's August 28 inputs. Rebuilds now select the artifact's recorded date;
the model loader still rejects a mismatch. Production input features retain full
precision in JSON so published models can reproduce exported forecasts.

The final V2 and Adaptive pools contain 610 modeled players and 163 market fallbacks.
Unsupported kicker market rows are excluded from the skill-player forecast pool.
Both board manifests report current. V1 and its compatibility alias match the
archived bytes exactly. Reapplying the saved production model to exported inputs
changes legacy rounded PPG and season-points fields by less than 0.0005. Canonical
PPG × expected games equals canonical season points exactly. Adaptive model scores
match the original snapshot within its legacy three-decimal export rounding.

The complete offline suite passed: 396 tests, with 35 network tests deselected.
Python lint/type checks, frontend build, and frontend lint passed. Frontend lint retains existing warnings; the installed Node 21 runtime
also emits Vite's engine-version warning, although the build succeeds. Full research
refitting was not needed for production publication; the production-only evidence
and rebuilt boards are the verified outputs. A complete reconstruction of older
market-only draft pools is still required before making a promotion decision.
