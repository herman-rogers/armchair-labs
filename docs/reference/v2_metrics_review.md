> Archived system reference. This describes the earlier metric/board system, not the current NextGen serving path.

> Evidence status (2026-09-22): the numerical results below are historical records.
> Earlier rankers were evaluated on different player populations and do not establish
> market outperformance. See [the stat-system review](../research/stat_system_review_2026-09-22.md)
> and [implementation and corrected evidence](../history/stat_system_implementation_2026-09-22.md).
> The frozen Adaptive snapshot remains experimental; no research model was promoted.

# V2 metrics review

Date: 2026-08-29. Scope: the v2 projection model (`src/patron/metrics/projection.py`,
`enrichment.py`, `backtest.py`), its configuration, the data it consumes, and the
rolling backtest.

Two generations of backtest output are cited. Sections 1–6 use the three-fold report
generated 2026-08-29 (`forecast_seasons: [2023, 2024, 2025]`) plus additional cuts of
its `metric_backtest_predictions.parquet`. Section 0 uses the 22-fold rerun
(2004–2025) produced later the same day after `metric_report.yaml` was extended, scored
with the new head-to-head ranking evaluation. The structural findings do not depend on
fold count.

The unit suite passed on the working tree at review time (343 tests after the ranking
evaluation, the shrinkage fix, the active-games denominators, and the fitted ranker
were added).

## 0. The goal, and how the report now measures it

The purpose of these metrics is not to describe last season and not to beat the
market. It is to **rank players correctly in each backtested year** — to put the
players who actually finish best at the top of the board, as often as possible — and
to carry that demonstrated ability into the 2026 prediction. Everything else in this
document is in service of that.

That goal fixes the evaluation:

- The bar is the **naive baseline**: rankings anyone can produce with no model at all.
  `historical_ppg_prior` (recency-weighted three-year PPG, the first step of v2) and
  last-season `ppg`. A model output earns its place only by beating them.
- The score is measured **at the top of the board**, not across the whole population.
  Whole-population Spearman is dominated by separating starters from players who leave
  the league, which any metric does. The report therefore scores every ranker on the
  same top-K slice per position (K = the starters a ten-team league drafts: QB 12,
  RB 24, WR 36, TE 12) with hit rate @K, NDCG @K, and Spearman inside a draftable pool
  (QB 24, RB 60, WR 80, TE 24).
- Every candidate is compared **head-to-head, fold by fold**, against the best baseline
  on that slice. "Beats baseline" requires a non-negative pooled lift and winning at
  least half the folds.

This is implemented in `backtest.analyze_rankings` and configured in the `ranking`
block of `metric_report.yaml` (baselines, candidates, targets, `top_k`, `pool`). The
JSON report carries it as `ranking_results`; the Markdown report opens with the
"Does v2 beat the naive baseline?" table. Rankers, K, and targets are config, so the
same evaluation can score any future sort key.

### What it says today (parity-fixed folds, target-matched baselines, strict W–T–L)

Hit rate @K vs next-season points. Baselines now include last season's `season_pts`
for the season-points outcome; lift is against the single best baseline on the
candidate's own folds, with the paired standard error; W–L counts folds strictly won
or lost (ties broken on top-K actual points). A candidate passes only with positive
lift and more wins than losses.

**Modern window (2019–2025, seven folds):**

| pos | best baseline | `v2_score` | `proj_ppg` | `fitted_ppg` | `fitted_season_points` |
|---|---|---|---|---|---|
| QB | `season_pts` 0.595 | 0.595 (0, 4–3) | 0.595 (0, 4–3) | **0.619 (+0.024 ± 0.028, 5–2) W** | 0.595 (0, 5–2) |
| RB | `historical_ppg_prior` 0.649 | 0.679 (+0.030, 6–1) W | 0.673 (+0.024, 5–2) W | **0.684 (+0.036 ± 0.012, 6–1) W** | 0.667 (+0.018, 4–3) W |
| WR | `season_pts` 0.675 | 0.651 (−0.024, 2–5) L | 0.651 (−0.024, 2–5) L | 0.651 (−0.024, 2–5) L | **0.679 (+0.004 ± 0.012, 4–3) W** |
| TE | `ppg` 0.524 | 0.476 L | 0.476 L | 0.500 (−0.024, 3–4) L | 0.512 (−0.012, 4–3) L |

`component_proj_ppg` is the highest RB number (0.702, +0.054, 6–1) but only 14–8 on
the long horizon, so `fitted_ppg` is the more consistent RB key.

**Long horizon (2007–2025, 19 fitted folds):**

| pos | best baseline | `v2_score` | `fitted_ppg` | `fitted_season_points` |
|---|---|---|---|---|
| QB | `season_pts` 0.576 | 0.580 (+0.004, 9–13) L | **0.610 (+0.031, 11–8) W** | 0.592 (+0.013, 12–7) W |
| RB | `ppg` 0.597 | 0.614 (+0.017, 15–7) W | **0.614 (+0.031, 14–5) W** | 0.603 (+0.020, 13–6) W |
| WR | `season_pts` 0.636 | 0.614 (−0.023, 6–16) L | 0.646 (+0.012, 11–8) W | **0.656 (+0.022, 13–6) W** |
| TE | `season_pts` 0.530 | 0.458 (−0.072, 9–13) L | 0.513 (0, 11–8) W | 0.526 (+0.013, 13–6) W |

**Overall draft value (one top-60 across all positions vs availability-adjusted
actual VOR; each ranker as per-game VOR against its own positional replacement, the
live board's construction):**

| ranker | modern hit @60 | lift | W–L | long-horizon hit @60 | lift | W–L |
|---|---|---|---|---|---|---|
| best naive (`historical_ppg_prior`) | 0.600 | — | — | 0.582 | — | — |
| **`fitted_season_points`** | **0.638** | +0.038 ± 0.006 | **7–0** | 0.599 | +0.021 | 15–4 |
| `fitted_two_stage` | 0.640 | +0.041 ± 0.008 | 6–1 | 0.600 | +0.022 | 14–5 |
| `fitted_ppg` | 0.617 | +0.017 | 5–2 | 0.596 | +0.018 | 13–6 |
| `v2_score` | 0.598 | −0.002 | 3–4 L | 0.554 | −0.027 | 8–14 L |
| `proj_ppg` | 0.593 | −0.007 | 3–4 L | 0.551 | −0.031 | 7–15 L |

Reading it:

1. **With honest baselines the per-position margins are small.** RB is the only
   position where v2 or the fitted ranker wins convincingly (+0.03 to +0.04, 6–1 and
   14–5). QB and WR are positive but within about one standard error in the modern
   window and firmer on the long horizon. TE passes nothing in the modern window;
   last-season `ppg` remains its best ranker.
2. **The overall order is where the fitted ranker earns its place.** Ranking every
   position by `fitted_season_points` VOR beats naive on every one of seven modern
   folds and 15 of 19 long-horizon folds. The hand-built `v2_score` and `proj_ppg`
   overall orders lose to naive in both windows — the original cross-position sort
   was worse than doing nothing.
3. **Season points: PPG × calibrated games wins.** Of the three formulations,
   `fitted_ppg × fitted_games` matches or beats the direct fit and the two-stage
   model at WR/RB overall, and the double-counted `expected_games` input is gone.
4. **Regularization is not binding.** Nested selection picks λ = 0.01 (the grid
   floor) almost everywhere; with 1,300–4,700 training rows and five to seven
   features the ridge is effectively least squares, and coefficient spreads across
   refits remain small (`fitted_ppg`: ≤ 0.2 except QB's prior weight at 0.41).

### The live board (as built)

`league.yaml` → `projection_rank_key`: `fitted_ppg` (QB, RB), `fitted_season_points`
(WR), `ppg` (TE); `projection_overall_key: fitted_season_points`. Live/report parity
is enforced: the board selects the same depth-chart snapshot the pending fold was
scored on, recomputes TD-over-expectation on the same window, and applies the stored
models only if the artifact's fingerprint, season, cutoff, and snapshot match
(otherwise it logs why and falls back to `v2_score`). Re-applying the stored models to
the live inputs reproduces the pending fold within 0.0005 PPG on 609 of 610 rows; the
610th is Ricky Pearsall, removed by a manual override that the backtest deliberately
ignores.

## 1. V2 under-performs the naive baseline on whole-population rank

This section is the diagnostic that led to §0. Whole-population Spearman is a harsher
and less draft-relevant measure than the top-K evaluation, but it isolates *where* the
model loses information, which the top-K table cannot.

Spearman rank correlation vs next-season PPG, players who played, three folds pooled:

| | ALL | QB | RB | WR | TE |
|---|---|---|---|---|---|
| `historical_ppg_prior` (weighted 3-yr PPG, no model) | **0.749** | **0.623** | 0.743 | **0.785** | **0.746** |
| last-season `ppg` alone | 0.731 | 0.550 | **0.752** | 0.773 | 0.733 |
| `individual_prior_ppg` (after shrinkage) | 0.658 | 0.553 | 0.698 | 0.709 | 0.590 |
| `prior_branch_ppg` | 0.660 | 0.557 | 0.708 | 0.697 | 0.619 |
| `component_proj_ppg` | 0.618 | 0.478 | 0.663 | 0.668 | 0.632 |
| **`proj_ppg` (v2)** | 0.642 | 0.526 | 0.700 | 0.694 | 0.642 |
| **`v2_score` (the sort key)** | 0.609 | 0.498 | 0.656 | 0.645 | 0.580 |

The ordering holds in every fold (2023/2024/2025 prior 0.725/0.746/0.773 vs `proj_ppg`
0.648/0.628/0.657).

On the draftable slice (top QB14/RB40/WR50/TE14 by `v2_score` per fold, n=354):

| target | last `ppg` | `historical_ppg_prior` | `proj_ppg` | `v2_score` | `expected_season_points` |
|---|---|---|---|---|---|
| next-season PPG | 0.581 | 0.603 | 0.596 | 0.539 | 0.603 |
| season points (absent = 0) | 0.526 | 0.518 | 0.476 | 0.513 | 0.519 |

Per position on that slice vs season points: QB `component_proj_ppg` is **0.27**
(prior 0.59); RB `proj_ppg` 0.61 is the one place v2 beats last-season PPG (0.59).

### Root cause: the shrinkage target

```
individual_prior = reliability × PPG + (1 − reliability) × position_ppg
```

`position_ppg` is the mean of the *rosterable pool* (top 2×baseline rank, ≥8 games),
i.e. starter-level production (~17 PPG for QB). Every small-sample player is pulled
*up* toward a starter. Examples from the folds:

| player | games | weighted PPG | individual prior | `proj_ppg` | actual |
|---|---|---|---|---|---|
| Kyle Trask (QB) | 1 | 0.9 | 17.0 | 20.8 | −0.05 |
| Chris Oladokun (QB) | 1 | 0.5 | 16.4 | 13.4 | 3.7 |
| Bailey Zappe (QB) | 4 | 9.9 | 16.0 | 16.9 | 6.9 |
| Joe Milton III (QB) | 1 | 21.2 | 18.7 | 18.5 | 2.6 |

This one step costs ~0.09 pooled Spearman (0.749 → 0.658). The same pattern is in
every component prior (`target_share`, `catch_rate`, `route_participation`, TD rates,
…): all are rosterable-pool means with hand-picked shrinkage constants
(k = 40/60/80/120/140/160/220), so the bottom-up branch inherits the bias.

Fix: shrink toward a prior conditional on what is known about the player. At minimum
the all-population positional mean (or replacement level); better, a regression of
next-season PPG on (weighted PPG, effective games) fit on the folds.

### Result of the fix (applied 2026-08-29)

`projection_prior_pool: all` in `league.yaml` (`_prior_rows` in `projection.py`): the
shrinkage target and every component-rate anchor now come from the games-weighted
all-player pool at the position instead of the top-starter pool. `rosterable` is
retained as a config option for comparison. Whole-population Spearman vs next-season
PPG, modern window, before → after:

| | ALL | QB | RB | WR | TE |
|---|---|---|---|---|---|
| `historical_ppg_prior` (unchanged) | 0.756 | 0.666 | 0.754 | 0.771 | 0.754 |
| `individual_prior_ppg` | 0.656 → **0.714** | 0.579 → 0.649 | 0.702 → 0.744 | 0.687 → 0.749 | 0.530 → 0.688 |
| `proj_ppg` | 0.647 → **0.708** | 0.602 → 0.653 | 0.679 → 0.749 | 0.668 → 0.752 | 0.594 → 0.726 |
| `v2_score` | 0.583 → **0.662** | 0.547 → 0.604 | 0.634 → 0.699 | 0.620 → 0.699 | 0.542 → 0.667 |

`proj_ppg` MAE fell from 4.10 to 3.20 and bias from +2.84 to +1.16 (modern window).
The remaining gap to the raw prior is now ~0.05 rather than ~0.11; what is left is the
other adjustments (§4) and the component branch. The top-K effect is in §0.

## 2. Systematic positive bias

`proj_ppg − actual_ppg`, by projected tier within fold and position, played rows:

| tier | QB | RB | WR | TE |
|---|---|---|---|---|
| 1–12 | **+3.9** | −1.1 | −0.6 | +0.5 |
| 13–24 | +3.4 | −0.7 | −0.3 | +0.6 |
| 25–36 | +5.0 | +1.7 | −0.4 | +1.0 |
| 37–60 | +6.7 | +2.5 | +0.8 | +2.3 |
| 61+ | +5.7 | +2.9 | +2.9 | +2.3 |

- After the shrinkage fix (modern window) the tier-37+ bias fell from +2.5/+3.4 to
+0.9/+1.6 at RB/WR and from +2.7 to +0.8 at TE; QB is still +3.3 to +5.5 below rank
24, which is now the component branch (§2 causes below) rather than the prior.

Pre-fix picture: RB/WR at the top are fine (slightly under, which is the expected regression).
- QBs are +3.4 to +6.7 at every tier; the QB component branch is the weakest piece of
  the model (bias +5.6, MAE 6.7). Contributing causes beyond §1:
  `projected_pass_attempts = team pass_volume × role` hands backups the full team's
  attempts; QB `context_delta` uses only pass volume and scoring.
- Tiers 37+ everywhere carry +2 to +7 PPG. That is exactly the population the waiver
  and free-agent views rank.

`expected_games`: bias +5.9, MAE 6.5. The model cannot produce a value below ~11.1
(sample floor 0.5, prior 0.94, 17-game shrinkage) while the actual mean including
retirees and cuts is ~10. In the top 36 tiers the zero-games share is 0–3% for RB/WR,
so this is harmless for the draft; the "strong" assessment for `expected_games` is
mostly the model separating starters from players who wash out of the league.

## 3. Silent data gaps

These make the three folds test three different models, without any error.

### 3a. Participation 2020–2022 produces zero route opportunities

nflverse's old-format participation (≤2022) carries `offense_players` but no
`offense_positions`. The multi-season concat null-fills the column, `_split` returns
`[]`, and the zip in `build_player_usage` yields nothing. Result in the derived cache:

| season | rows | route_opportunities > 0 | active_games > 0 |
|---|---|---|---|
| 2020 | 1984 | 0.0 | 0.0 |
| 2021 | 2082 | 0.0 | 0.0 |
| 2022 | 2007 | 0.0 | 0.0 |
| 2023 | 1986 | 0.26 | 0.32 |
| 2024 | 2029 | 0.26 | 0.31 |
| 2025 | 2046 | 0.26 | 0.31 |

So the 2023 fold has no route/TPRR/`active_games` data at all (`route_weight` drops to
0, availability falls back to `games`), 2024 has one season, 2025 has two. Fix: for
old-format seasons resolve positions from the `offense_players` GSIS IDs via rosters.
Also: fail loudly when a season's participation yields no route opportunities.

### 3b. Depth charts 2023–2024 never load

Those seasons use the old schema (`season/week/club_code/depth_team/depth_position`);
`require_columns` throws and `pipeline.build_metric_report` catches it with a warning.
`depth_chart_rank` coverage by forecast season: 2023 0%, 2024 0%, 2025 91%, 2026 100%.
Everything depth-driven — `depth_role_factor` and the 65%-weight depth-based teammate
availability blend — therefore has **one** out-of-sample fold, which is why the report
labels it "insufficient". The old schema has `week` and `game_type`, so a preseason /
week-1 cutoff is derivable; that gives three folds.

### 3c. `games` counts stat-row weeks, not games played

nflverse weekly stats have no row for a dressed player with no recorded stat. Among
players with participation data, `games < active_games` for 46% of WRs and 73% of
TEs (mean gap −1.3 and −3.3 games). Ross Dwelley 2023: 2 "games" / 12 active, PPG 1.1
rather than 0.18. This inflates PPG for low-usage players in the prior, in the
`actual_ppg` target, and in the TE replacement level. `active_games` is already
computed; use it as the PPG denominator when present and keep `games` as "games with
production".

**Applied 2026-08-29, in two parts.** First `normalize_ppg_for_active_games`
(`enrichment.py`) made v2's PPG `season_pts / max(games, active_games)`. That alone
left every *other* per-game quantity — season weights, `effective_games`, targets and
carries per game, bonus per game, reliability, replacement gating — on stat-row
`games`, so the component branch projected TE targets per stat-game while being scored
per active game. `_games_played` / `_games_column` in `projection.py` now route all of
them through the same denominator. Effect, modern window, before → after that second
step: `proj_ppg` bias TE +0.51 → 0.00, RB +0.89 → +0.53, WR +0.89 → +0.58; MAE TE
2.19 → 1.98; whole-population Spearman `proj_ppg` TE 0.726 → 0.755, ALL 0.708 → 0.716,
`v2_score` ALL 0.662 → 0.679. Top-K hit rates moved by at most ±0.01. TE's remaining
gap is a ranking problem (§0 point 2), not a calibration one.

A lead from the same test: `avg(ppg, proj_ppg)` is the best RB ranker on the long
horizon (0.621, 18/22 folds) and ties the best in the modern window — the projection
is under-weighting last season at RB.

### 3d. Live board vs backtest

The 2026 board is built from `seasons: [2023, 2024, 2025]` and so has participation
and depth charts for every input season. No completed fold ever validated that
configuration.

## 4. Ablations of the prior-branch adjustments

Spearman vs actual PPG on the draftable slice after removing one factor:

| variant | ALL | QB | RB | WR | TE |
|---|---|---|---|---|---|
| full prior branch | 0.576 | 0.561 | 0.577 | 0.463 | 0.514 |
| − team context | **0.601** | **0.695** | 0.574 | **0.520** | 0.522 |
| − age factor | 0.557 | 0.555 | **0.525** | 0.441 | 0.518 |
| − TD regression | 0.582 | 0.561 | 0.607 | 0.450 | 0.534 |
| − bonus regression | 0.576 | 0.511 | 0.585 | 0.459 | 0.520 |
| − depth factor | 0.569 | 0.470 | 0.561 | 0.461 | 0.514 |

- **Team context hurts**, materially for QB and WR. Its rank correlation with the
  residual (actual − prior) is ≈0 at every position (QB 0.04, RB 0.06, WR −0.04,
  TE 0.17). Candidates: drop it from the prior branch (the component branch already
  carries team volume), or apply it only to movers. Movers (n=16 on the slice) carry
  +2.4 bias vs +0.4 for non-movers, so context is not fixing the move case either.
- **Age curve is real** (RB 0.577 → 0.525 without it; residual correlation 0.27 RB,
  0.33 WR). Keep it and fit the slopes.
- **Depth chart helps** (QB 0.561 → 0.470 without it) but on one fold; fix §3b first.
- TD and bonus regression are noise-level at three folds. `projected_bonus_pg` has a
  partial of −0.30 vs next-season VOR and the report calls it "strong" (see §5).
- Blend weight sweep (`proj = (1−w)·prior + w·component`): pooled, w=0 maximizes rank
  (0.660) while w=1 minimizes MAE (3.80); on the draftable slice w≈0.5–0.65 is best
  (0.594–0.596 vs 0.576 at w=0). 0.65 is defensible for the top of the board; the
  component branch adds nothing below it.

## 5. Backtest and report issues

- **`_assessment` ignores sign.** It uses `abs(partial)`, so an anti-predictive metric
  prints "strong". Add `expected_sign` per catalog entry and label wrong-sign results
  `harmful`.
- **No model-vs-baseline row.** Partial correlation vs `historical_ppg_prior` answers
  "is this new information", but the report never shows whether `proj_ppg` or
  `v2_score` beats `historical_ppg_prior` or last-season `ppg` head-to-head, per fold
  and position. That is the number that decides whether v2 should be the default.
- **Market baseline — resolved.** The report now joins the latest archived positional
  FantasyPros redraft ECR on or before August 31 by FantasyPros ID → GSIS ID. The
  archive fully covers 2021–2025, exposed as its own window; sparse ECR cannot become
  the best baseline in older windows by scoring only favorable folds.
- **Full-population Spearman is dominated by starters vs scrubs.** Report the
  draftable slice, plus draft-shaped metrics: top-K hit rate by position, tier
  calibration, NDCG.
- **`actual_availability_value = VOR × games/17`** rewards below-replacement players
  for missing games (negative × <1 is less negative). Consistent under "you would
  start replacement instead", but it produces artifacts such as "latest-season games →
  availability VOR partial −0.18". The same applies to `v2_score`'s availability
  scaling for negative-VOR players in waiver views.
- Fisher intervals treat pooled rows as independent; each player appears in up to
  three folds, so the intervals are optimistic.
- `opportunity_implied_targets = (wtd_opp − carries) / 2.2` is algebraically
  `targets`; `metrics.md` calls it "a third, independent role anchor". The 0.20 + 0.15
  weights are simply 0.35 on historical targets per game.
- `_position_means` is not season-decayed while `_component_priors` is.

## 6. Smaller method notes

- `TeamProfile.target_availability = 1 − max_other/total` barely separates a WR1 from
  a WR3 and is used in both branches (prior at 0.20 weight, component at ^0.35).
- OUT weeks are already missed games, so the 25% injury-designation blend counts the
  same absences twice.
- WR/TE `qb_context` includes the receiver's own production, so it is circular for
  non-movers (harmless as a ratio, but worth knowing).
- Floor and ceiling VOR are near-monotone in `proj_ppg` (partials 0.25 vs 0.27 for
  `proj_vor`); the 75/15/10 blend is mostly a re-weighting of PPG.
- Fold construction is clean: team profiles, priors, TD rates, birth dates, and depth
  cutoffs are all windowed to the history seasons. No leakage found.

## 6a. Cross-player effects: QB quality, team volume, and target competition

Football is a team sport, so the review checked whether team-level effects are real
and whether v2 measures them usefully. Tested on WR/TE across all 22 folds
(n ≈ 5,400 player-seasons), using only information knowable before the season, plus
two oracle signals as upper bounds.

Partial Spearman vs `historical_ppg_prior` (does the signal add anything beyond the
weighted-PPG prior?):

| signal | vs next PPG | vs season points | by era 2004–12 / 13–18 / 19–25 |
|---|---|---|---|
| QB1 yards per attempt, prior season (z within season) | +0.045 | +0.092 | 0.01 / 0.04 / **0.08** |
| QB1 TD−INT rate, prior season (z) | +0.050 | +0.086 | — |
| team pass attempts (volume) | −0.025 | −0.019 | — |
| strongest teammate's target share (v2's competition input) | −0.010 | +0.011 | ≈0 in every era |
| oracle: next season's actual QB1 efficiency | +0.061 | +0.074 | — |
| oracle: QB1 changed next season | residual −0.31 vs −0.30 | no effect | — |

Top-K hit rate with the signal folded into the baseline,
`historical_ppg_prior × (1 + β × QB1_YPA_z)`:

| | baseline | β = 0.02 | β = 0.04 | β = 0.08 |
|---|---|---|---|---|
| WR season points, modern (7 folds) | 0.631 | 0.643 (6/7 W) | **0.651 (5/7 W)** | 0.635 |
| WR season points, long (22 folds) | 0.633 | 0.634 (11/22) | 0.635 (9/22) | 0.629 |
| TE PPG, long (22 folds) | 0.599 | 0.595 | 0.602 (12/22 W) | **0.606 (13/22 W)** |
| TE season points, modern | 0.500 (`ppg`) | 0.476 L | 0.476 L | 0.476 L |

Findings:

1. **QB quality is real but small, and only the per-attempt version carries it.**
   Partial +0.05 to +0.09, strongest in the modern era. Even the oracle (knowing next
   season's QB play exactly) reaches only +0.06/+0.07, so prior-season efficiency
   captures most of what is available. Ceiling: about one extra correct WR3 per season.
2. **Team pass volume carries nothing** (slightly negative partial). v2's
   `pass_volume` context term should go.
3. **Team-level "passing environment" is a competition measure, not a QB measure.**
   Including the receiver's own catches it is circular; excluding them, the rest of the
   team's passing production correlates *negatively* (−0.35, 2023–25) with his PPG.
   v2's `qb_context` is the circular form and is ~1.0 for every non-mover.
4. **v2's competition input is inert.** "Strongest teammate share" has partial ≈0 in
   every era. Only the player's own projected target share matters; a competition term
   should enter through the projected target tree summing to ~1 per team, not as a
   separate multiplier.
5. **QB-change events have no average effect**; direction (better or worse QB) is what
   matters, and the efficiency z-score already carries it.
6. Where cross-player structure does matter and is currently ignored: **week-to-week
   covariance** (a QB and his WR1 boom and bust together). Irrelevant to point
   projections; relevant to the League team-strength rating, which sums player
   variances as if independent.

Retested after the shrinkage fix (same 22 folds): unchanged. QB1 YPA partial +0.046 /
+0.096, team attempts −0.03, strongest-teammate share ≈0, QB-change event no effect.
`historical_ppg_prior × (1 + 0.04 × QB1_YPA_z)` lifts WR modern season-points hit rate
0.643 → 0.659 (5/7 folds) and matches the baseline on the long horizon; TE gains are
within noise. The effect is real, small, and stable.

Recommendation: replace `qb_context` and `pass_volume` in the prior branch with a
single capped multiplier `1 + 0.03 × QB1_YPA_z` (±10%) for WR/TE, where QB1 is the
projected team's depth-chart starter and the z-score is his prior-season yards per
attempt within that season. Land it after the shrinkage fix (§1), since the baseline
it rides on will change, and accept it only if the §0 WR row holds.

## 7. Missing for a draft board

1. **Rookies** — outside the model; several top-40 picks every year.
2. **ADP / market price** — no way to express or measure value versus cost.
3. **Offseason moves** before the depth-chart cutoff live only in `projections.yaml`,
   which is empty for 2026, so the live board is pure depth chart.
4. Cumulative RB touches (`age.py` states this is what the cliff really measures).
5. Weekly head-to-head utility: playoff-week schedule, byes, lineup-level rather than
   player-level floor.

## 8. The fitted ranker: learning the weights from the backtest

The review's central recommendation was to stop hand-tuning ~30 constants and let the
completed folds set the weights. `src/patron/metrics/fit.py` does that as a
walk-forward stage inside `just metric-report`:

- **Features** (config `fit.features`): the signals that survived the ablations —
  `historical_ppg_prior`, last-season `ppg`, `age_factor`, `depth_role_factor`,
  `component_proj_ppg`, `expected_games`. Six, not fifty-three.
- **Model**: ridge regression per position on standardized features, intercept
  unpenalized, `ridge_lambda: 1.0`; falls back to a pooled fit when a position has
  fewer than `min_position_rows`. Missing features are imputed with the training mean,
  so a signal absent for an era carries no weight there.
- **Walk-forward**: the forecast for season *t* is fitted on completed folds strictly
  before *t* (minimum three). 2004–2006 are training-only; 2007–2025 are scored
  out-of-sample; the pending 2026 forecast uses every completed fold. No row is ever
  scored by a model that saw its outcome.
- **Outputs**: `fitted_ppg` and `fitted_season_points = fitted_ppg × expected_games`
  join the ranking table as candidates; learned coefficients per refit are in
  `fitted_models`, with the latest weights and their spread across refits in
  `fitted_model_summary` and the Markdown "Learned weights" table. Lift is measured on
  the candidate's own folds so the 19-fold fitted ranker is not compared against a
  22-fold baseline.

### Learned weights (latest refit, standardized; sd across 19 refits in parentheses)

| pos | train rows | `historical_ppg_prior` | `ppg` | `age_factor` | `depth_role_factor` | `component_proj_ppg` | `expected_games` |
|---|---:|---|---|---|---|---|---|
| QB | 1279 | +0.97 (0.14) | +0.84 (0.11) | +0.07 (0.07) | +0.83 (0.09) | +0.83 (0.07) | +1.05 (0.13) |
| RB | 2359 | +1.12 (0.08) | +1.09 (0.08) | +0.27 (0.04) | +0.49 (0.06) | +0.86 (0.06) | +0.37 (0.10) |
| WR | 3462 | +1.28 (0.03) | +1.24 (0.03) | +0.15 (0.05) | +0.47 (0.03) | +0.31 (0.09) | +0.42 (0.02) |
| TE | 1949 | +0.92 (0.05) | +0.91 (0.05) | +0.10 (0.05) | +0.47 (0.08) | +0.23 (0.10) | +0.31 (0.08) |

Every weight is positive, every weight is stable (no sign flips, sd ≤ 0.14), and the
pattern is readable: last-season `ppg` earns nearly as much weight as the multi-year
prior everywhere (v2's 0.55 decay under-weights recency); the depth chart is worth as
much as the prior at QB; the component branch matters at QB and RB and little at WR/TE;
age is an RB effect; `expected_games` is a QB effect (the starter/backup split).

### Top-K hit rate vs next-season points, fitted vs the rest

| | best baseline | `proj_ppg` | `v2_score` | **`fitted_ppg`** | **`fitted_season_points`** |
|---|---|---|---|---|---|
| QB modern | 0.536 | 0.595 W | 0.595 W | **0.607 W (6/7)** | 0.607 W (4/7) |
| RB modern | 0.649 | **0.691 W** | 0.679 W | 0.673 W (5/7) | 0.643 L |
| WR modern | 0.639 | 0.655 W | 0.655 W | 0.655 W (5/7) | **0.686 W (7/7)** |
| TE modern | 0.524 (`ppg`) | 0.464 L | 0.476 L | 0.488 L | **0.512** L (5/7) |
| QB long (19 folds) | 0.572 | 0.583 W | 0.583 W | **0.605 W (11/19)** | 0.592 L |
| RB long | 0.597 (`ppg`) | 0.616 W | 0.614 W | **0.616 W (16/19)** | 0.601 W |
| WR long | 0.635 | 0.625 L | 0.624 L | 0.652 W (14/19) | **0.659 W (15/19)** |
| TE long | 0.526 | 0.451 L | 0.455 L | 0.513 L | **0.539 W (14/19)** |

Reading it:

1. **The fitted ranker is the best or tied-best candidate at QB, WR, and TE**, and the
   only candidate that beats the baseline at WR across the long horizon. It does so
   with six features and no hand-set constants beyond `ridge_lambda`.
2. **TE finally moves.** `fitted_season_points` reaches 0.512 in the modern window
   (baseline 0.524 — one starter over seven years) and **beats the baseline on the long
   horizon**, 0.539 vs 0.526, 14 of 19 folds. Every hand-built v2 output was 0.45–0.48.
3. **RB stays with `proj_ppg`** (0.691 vs 0.673). The component branch's RB opportunity
   modelling is worth more than a linear re-weighting captures — which is fine; the
   fitted model is a candidate, not a replacement.
4. `fitted_season_points` beats `fitted_ppg` on season-point targets at WR and TE and
   loses at RB — the same availability trade-off seen with `expected_season_points`.

### Status after the 2026-08-29 review round

All six items of the follow-up review are done:

1. **Live/report parity** — one `depth_chart_as_of(season, cutoff)` selector for the
   backtest and the live board, bounded to entries after September 1 of the prior
   season (stale players no longer pollute teammate availability); per-fold
   TD-over-expectation; artifact check on snapshot date. Parity ≤ 0.0005 PPG.
2. **Ranking contract** — target-matched baselines (`season_pts` for the
   season-points outcome), lift on the candidate's own folds, strict W–T–L against
   one fixed baseline with tie-break on top-K actual points, paired SE and CI.
   `patron metric-report --reanalyze` re-scores from the retained parquet.
3. **Season-points modelling** — model specs (`fit.models`) with nested walk-forward
   λ selection; `fitted_ppg` no longer takes `expected_games`; three formulations
   compared (§0 point 3).
4. **Global value** — the OVERALL test (§0) validates the cross-position order;
   `projection_overall_key` is now `fitted_season_points`.
5. **Artifact contract** — schema/model version, fingerprint over the fit specs,
   depth cutoff and snapshot, creation time, full-precision parameters; the loader
   rejects any mismatch. TypeScript types carry `standardization`,
   `baseline_hit_rate`, W–T–L, and the artifact.
6. Documentation, this file.

Guardrails going forward: keep the feature set small; refit only inside
`just metric-report` so every weight is walk-forward validated; treat a coefficient
whose spread across refits approaches its magnitude as a signal to drop; and keep the
acceptance rule — a change ships only if it lifts the modern hit rate without losing
the long horizon.

## 9. Research-signal challenger

The low-hanging-football-metrics review is implemented as a separate walk-forward
challenger rather than hand-wiring more constants into v2:

1. **Market:** dated positional FantasyPros ECR is an external baseline, never a
   feature. This directly tests whether the model adds anything to public consensus.
2. **Expected opportunity:** ffopportunity expected PPR, yards, touchdowns, and first
   downs are aggregated over official regular-season game IDs. The source-season
   transform never reads the following season's outcome.
3. **QB process and rushing decomposition:** passing EPA/attempt, attempt-weighted
   CPOE, passing/rushing first-down rates, and PBP scrambles separated from designed
   carries.
4. **Team tendency:** pass rate, early-down pass rate, plays/game, no-huddle, EPA, and
   pass-over-expected only in quarters 1–3 with the score within eight points; kneels
   and non-scrimmage plays are excluded.
5. **NGS residuals:** official season-summary CPOE, time to throw, depth/aggressiveness,
   receiver separation/YAC over expected, RYOE/attempt, and the share of runs beating
   expected yards.
6. **Return/roster probability:** retained from the preceding review as the
   walk-forward `fitted_return_prob × fitted_games_if_played × fitted_ppg` two-stage
   candidate.

Separate `fitted_opportunity_ppg`, `fitted_qb_process_ppg`,
`fitted_team_tendency_ppg`, and `fitted_ngs_ppg` candidates make the recommended order
independently falsifiable. `fitted_research_ppg` then includes the surviving core fit
plus only xFP, QB scrambles, neutral pass-over-expected, NGS YAC-over-expected, and
NGS RYOE;
`fitted_research_season_points` multiplies it by `fitted_games`. Missing-era values
are training-mean imputed and therefore carry zero weight before a feed exists. Both
are `apply_live: false`: this section is an acceptance experiment, not a production
sort-key change. The report's raw metric catalog provides the ablation-level evidence;
the ranking table decides whether the combined challenger earns another iteration.

### 2026-08-29 result

The full 2004–2026 rebuild produced 12,750 forecast rows (22 completed folds plus the
pending 2026 fold). The new market window is the clearest result:

| position | ECR hit @K, next-season points | best model candidate | candidate lift vs ECR |
|---|---:|---:|---:|
| QB | **0.600** | `fitted_ppg` 0.583 | −0.017 |
| RB | 0.692 | `proj_ppg` **0.700** | +0.008 (2–3; fails) |
| WR | **0.689** | `fitted_research_season_points` 0.689 | 0.000 (2–3) |
| TE | **0.600** | `fitted_research_ppg` 0.583 | −0.017 |

No candidate beats public consensus over the five fully archived seasons. ECR also
has large incremental correlation after controlling for the historical prior
(partial Spearman +0.42 to +0.52 by position), so it is not merely a restatement of
last-season scoring.

The raw and staged experiments say:

- **Expected opportunity:** xFP adds useful residual signal at WR/TE (modern partial
  +0.17/+0.19) but its fitted challenger does not improve top-K season-points rank.
- **QB:** scrambles are the useful process signal (modern partial +0.17); EPA/attempt
  and CPOE are redundant after the prior. `fitted_qb_process_ppg` wins 6–1 against
  naive on the modern PPG target but still trails existing `fitted_ppg` (0.583 vs
  0.595) and market ECR.
- **Team tendency:** neutral pass-over-expected is useful only for QB (+0.10 partial);
  neutral volume/pass rate is weak elsewhere. The family adds no top-K lift.
- **NGS:** RYOE/attempt is strong for RB (+0.21 partial), and YAC over expected is
  useful for WR (+0.12) and strong for TE (+0.29). The standalone NGS fit does not
  beat the production fit—a warning against promoting raw correlation directly.
- **Combined survivors:** `fitted_research_season_points` is the one material move:
  TE next-season-points hit rate rises from production `fitted_season_points` 0.512 to
  **0.548** modern (+0.024 vs naive, 4–3) and 0.526 to **0.535** long horizon (+0.022,
  15–4). It still trails ECR 0.600, so it remains report-only.

Conclusion: retain the current live position keys. The first production-worthy use
would be a TE-specific challenger, but only after it beats ECR—not merely last-season
PPG—in another prospective or expanded-market sample.

## 10. Removed after the review

Per the evidence in §0, §4, and §6a, the following hand-built pieces were deleted
rather than left as alternatives: the prior-branch TD regression, big-play regression,
and team-context scalar; the `v2_score` floor/ceiling/availability blend and its
`floor_vor`/`ceiling_vor`/`availability_adjusted_vor`/`expected_season_points`
outputs; and their config keys. `proj_ppg`, `prior_branch_ppg`, and
`component_proj_ppg` remain because they are fitted-ranker features and report
diagnostics; `projected_floor/ceiling/volatility` remain as descriptive fields and
roster-risk inputs. The board's fallback when no fitted model is available is now
`adj_proj_vor`. The League power rank and head-to-head team comparison both total the
canonical `v2_overall_vor` across the best legal lineup. The supporting expected-points,
floor, and risk fields still read the fitted mean and games (`roster_*_columns` in
`league.yaml`), with volatility rescaled to the fitted mean.

## 10a. Experimental round: market-informed and research candidates (2026-08-29, commit 490bab4+)

Built strictly as `apply_live: false` candidates — the live board keeps `fitted_ppg`
(QB/RB), `fitted_season_points` (WR), `ppg` (TE), and the `fitted_season_points`
overall order. New in the contract: `require_features` / per-spec `min_train_folds`
on model specs, and every candidate is scored head-to-head against the FantasyPros
ECR ranker on the folds it covers (`market_*` fields, `ranking.market_baseline`).

**The market is the real bar.** ECR (preseason snapshots, 2021–2025) hits 0.650 on the
overall top-60, 0.692 RB, 0.689 WR, 0.600 QB/TE. On its own folds our best live
outputs are at parity overall (`fitted_season_points` −0.003, 4–1) and at RB/WR, and
behind at QB (−0.017) and TE (−0.050).

**Market as a feature (`fitted_market_*`)** — core five + `market_ecr_score` +
`market_ecr_sd`, trained only on rows carrying ECR, scored on 2023–2025 (three folds):

| slice | market on those folds | `fitted_market_season_points` | vs market | W–L |
|---|---|---|---|---|
| QB | 0.556 | **0.611** | **+0.056 ± 0.028** | **3–0** |
| WR | 0.685 | **0.704** | +0.018 ± 0.009 | 2–1 |
| Overall | 0.628 | 0.633 | +0.006 ± 0.029 | 2–1 |
| RB | 0.708 | 0.667 | −0.042 | 0–3 |
| TE | 0.583 | 0.528 | −0.056 | 0–3 |

The ECR coefficient is the largest in every position (+4.6 QB, +2.1 WR, +2.0 RB,
+1.2 TE, stable across refits), confirming the market carries information nflverse
cannot see. Three folds is thin: QB and WR are promising, RB/TE are not, and none of
it is adoption evidence yet. The model needs two more seasons of ECR before it can
pass the acceptance rule.

**Core-plus (`fitted_core_plus_*`)** — core five + scrambles, designed carries,
neutral EPA, xFP: the best non-market RB ranker (0.691, +0.042 vs naive, 6–1; at
parity with the market, 2–3) via `designed_carries`; no gain elsewhere.

**Research subset** — `fitted_research_season_points` remains the only TE candidate
above naive (0.536, +0.012, 4–3) but is well behind the market (−0.050).

Retired: `fitted_ngs_ppg`, `fitted_opportunity_ppg`, `fitted_qb_process_ppg`,
`fitted_team_tendency_ppg` (no lift, small unstable weights, ≤ 21% NGS coverage).
All research grids now start at 0.01.

Decision: **no live change.** Re-run this comparison when the 2026 fold completes; if
`fitted_market_season_points` stays ahead of the market at QB/WR over four or more
folds, promote it there.

## 11. Market fallback for unrated players, and the Hunter override

Players the model cannot rate — 2026 rookies, or anyone without usable tape — used to
be absent from the board, the overall order, the power rank, and the waiver views;
the frontend showed ESPN's draft-room order as a labelled placeholder. The FantasyPros
consensus is the ranker the backtest vindicates (0.650 overall, ahead of us at
QB/TE), so it is now the fallback *value*, mapped onto the model's scale rather than
mixed in as a rank:

- `pipeline.load_draft_market` / `attach_market` join the draft-season ECR snapshot
  (2026-08-28, 693 players; 87% of rated rows) onto every row and add market-only
  players as rows (162 for 2026) with identity from the id crosswalk.
- `board.rank._fill_market_fallback` gives each such player the median
  `v2_overall_vor`, `v2_rank_vor`, and roster-simulation inputs (`fitted_ppg`,
  `fitted_games`, `fitted_season_points`, `projected_volatility`, availability) of the
  three model-rated players at his position whose ECR is nearest. He lands where the
  board already places players the market values like him, never above rated
  players the market ranks below him, and is tagged `rank_source = market`
  (`v2_rank_key = market`). Players ranked by neither stay unranked and sort last.
- The League power rank (`ranking_total` on `v2_overall_vor`) and the weekly
  simulation therefore include market-placed players on the model's scale instead of
  excluding them or falling back to a third-party projection; ESPN's draft rank stays
  as an informational column.

`position_overrides` in `league.yaml` maps GSIS ids to board positions before the
position filter; Travis Hunter (`00-0040718`, filed CB by nflverse) is the first
entry and now carries real model values (WR, 7 games).

## 12. Additive structural round (2026-08-30)

Added report-only Week-1 roster/status, team-change and vacated-opportunity, offensive
snap, draft-capital, experience, and historical-contract features. The rebuild covers
12,751 returning-player forecasts and 22 completed seasons (2004–2025). All new
models have `apply_live: false`; no production rank key changed.

Season-points hit rate shows real but position-specific lift over the existing fitted
season-points model:

| position | 2007–25 existing | best additive stage | incremental | 2019–25 existing | best additive stage | incremental |
|---|---:|---:|---:|---:|---:|---:|
| QB | 0.592 | **0.632 status** | **+0.040** | 0.595 | 0.607 status/rostered | +0.012 |
| RB | 0.603 | **0.647 status** | **+0.044** | 0.667 | **0.708 combined** | **+0.042** |
| WR | 0.656 | **0.678 roster** | **+0.022** | 0.679 | **0.698 roster** | **+0.020** |
| TE | 0.526 | **0.561 combined** | **+0.035** | 0.512 | **0.560 combined** | **+0.048** |

The gains are era-stable enough to keep researching: combined RB hit rate is 0.64,
0.58, and 0.71 across 2007–12, 2013–18, and 2019–25; WR is 0.69, 0.66, and 0.68;
TE is 0.58, 0.54, and 0.56. QB is the exception: the full combined stack falls to
0.50 in 2013–18 and does not beat the existing model overall.

**Availability is the main discovery.** A binary Week-1 roster control lowers
games-played MAE from 3.52/4.41/4.43/4.43 to 2.90/3.51/3.50/3.32 for QB/RB/TE/WR
over 2007–25. Graded ACT/INA/RES status lowers it again to
2.73/3.14/3.19/3.00. This is much larger than any efficiency-stat gain, but it has a
cutoff caveat: nflverse weekly rosters have no historical publication timestamp and
Week 1 status can contain information unavailable at an August 31 draft. Membership
is a reasonable post-cutdown proxy; graded game status is not promotion-safe until a
truly dated transaction/PUP/injury feed replaces it.

**What did not survive the paired ranking test:** prior-season snap share is mostly
redundant with PPG/role, late snap-share trend is weak, and raw team vacated target or
carry share is weak. Depth-weighted vacated opportunity has useful residual signal
(modern partial Spearman: RB targets 0.187/carries 0.151, WR targets 0.234, TE targets
0.168), but the standalone linear vacated model does not improve ranking. Contract
guarantees and years remaining have strong raw partial correlations but a
status-independent commitment stack is flat or worse at QB/RB/WR; it helps TE
(0.548 modern versus 0.512 existing), so the broad contract association is mostly
selection rather than portable incremental signal.

Against the five-fold ECR market, only combined RB wins clearly (+0.017, 5–0).
Combined TE is approximately level; QB, WR, and the overall top-60 remain behind.
The market-augmented all-feature model is worse than ECR, indicating severe
over-parameterization on only three trainable folds.

Decision: keep the entire round experimental. The next data pull should be a dated
preseason transaction/roster/PUP/suspension feed; then refit the roster stage using
only information genuinely available at the draft cutoff. Preserve depth-weighted
vacated opportunity for nonlinear/position-specific TE and RB tests. Do not promote
snap trend, raw vacated share, or the market-plus-everything model.

## 13. Hidden-pattern loop and adaptive position models (2026-08-30)

The follow-up was run as a three-stage discovery/confirmation loop rather than one
large feature search. Residual candidates had to keep the same direction in
2007–2014, 2015–2018, and 2019–2025; every fitted candidate remained walk-forward and
report-only; and all final comparisons use held-out forecast seasons. The JSON now
includes both `residual_pattern_results` and a full `ranking_sensitivity_results`
rerun restricted to the Week-1-rostered population.

The durable residual patterns were positional, not a universal new scoring formula:

- QB availability residuals carry contract/depth security, while age itself remains
  weak after production and role are controlled.
- RB residuals retain depth-weighted vacated carries/targets.
- young WRs are under-projected; combine speed interacted with early career is stable
  across all three eras.
- young TE combine burst has a repeatable PPG residual, while established high-role
  WR/TE players are slightly over-projected on next-season games.

Historical combine data was added to test those young-player interactions. The
standalone career, athletic, security, and lean residual models did **not** improve
the overall board against ECR. This is an important rejection: stable residual
correlation was not enough to justify a broad additive model.

The first meaningful movement came from honest model selection by position. For each
forecast and position, `adaptive_select` chooses among six already-backtested outputs
using only folds before the forecast (minimum three); config order gives the incumbent
an exact-tie advantage. No modern-era position choice is hard-coded.

| overall ranker | 2007–25 hit | 2019–25 hit | 2021–25 hit | vs ECR | market W–L |
|---|---:|---:|---:|---:|---:|
| existing `fitted_season_points` | 0.599 | 0.638 | 0.647 | −0.003 | 4–1 |
| existing `fitted_two_stage` | 0.600 | 0.641 | 0.653 | +0.003 | 3–2 |
| full ridge stack | 0.612 | 0.641 | 0.653 | +0.003 | 3–2 |
| adaptive PPG-quality selector | 0.604 | **0.650** | **0.660** | **+0.010** | **4–1** |
| adaptive season-NDCG selector | **0.621** | 0.643 | 0.653 | +0.003 | 3–2 |

On common folds, the PPG-quality selector improves modern overall hit rate by 0.0095
over `fitted_two_stage` (3 wins, 0 losses, 4 ties) and the season selector improves
long-horizon hit rate by 0.0211 (9–3). The rostered-only sensitivity rerun preserves
the result: PPG-selector hit rate is 0.655 modern and 0.663 market, +0.013 versus ECR
on the market folds. A ridge blend of both selectors does not improve them and is
rejected.

The latest expanding-window choices explain the pattern: PPG quality selects athletic
QB/TE models, the roster-membership RB model, and the full WR stack; season NDCG
selects the full stack at QB/WR and the additive stack at RB/TE. The improvement is
therefore a regime/position effect, not evidence that every discovered feature belongs
in one equation.

Decision: **retain the adaptive selectors in the experimental layer, with no live
promotion yet.** The market advantage is only five folds and was discovered during
an iterative search; NDCG still trails ECR. In addition, the winning RB/WR sources use
the Week-1 roster proxy. A genuinely dated preseason transaction/PUP/cut feed is now
the gating data pull for promotion. Combine data is sufficient for continued
athleticism research; another broad efficiency-stat pull is lower value.

## 14. Cutoff-safe availability and 2026 prospective freeze (2026-08-30)

The gating pull is complete. nflverse does not publish a historical transaction
dataset, so the report now caches ESPN's dated team transaction archive from January 1
through the configured August cutoff for every 2004–2026 forecast. It applies cuts,
signings, trades, retirement, PUP/NFI/IR, suspension, activation, and practice-squad
language to each returning player's prior-season team state. Ambiguous name matches
are rejected unless the transaction team uniquely identifies the player. The old
Week 1 roster remains only as a named sensitivity proxy.

The rebuilt report contains 25,734 source transaction rows and 12,751 player
forecasts. A dated event matches 42.07% of forecasts; 86.81% remain rostered at the
cutoff. Week 1 proxy coverage is 74.59%, and membership agrees on 88.41% of rows where
that proxy exists. The deliberate limitation is conservative carry-forward: a player
with no matched retirement/release can remain active. This can dilute the feature but
cannot inject a post-cutoff event.

With the selector definitions and candidate order unchanged, the overall result is:

| overall ranker | 2007–25 hit | 2019–25 hit | 2021–25 hit | vs ECR | market W–L | market NDCG |
|---|---:|---:|---:|---:|---:|---:|
| existing `fitted_season_points` | 0.599 | 0.638 | 0.647 | −0.003 | 4–1 | 0.667 |
| existing `fitted_two_stage` | 0.600 | 0.641 | 0.653 | +0.003 | 3–2 | 0.666 |
| cutoff roster stage | 0.606 | 0.643 | 0.647 | −0.003 | 3–2 | 0.660 |
| full cutoff-safe ridge stack | 0.607 | 0.636 | 0.637 | −0.013 | 1–4 | 0.636 |
| adaptive PPG-quality selector | **0.610** | **0.652** | **0.657** | **+0.007** | **4–1** | 0.670 |
| adaptive season-NDCG selector | 0.602 | 0.633 | 0.633 | −0.017 | 1–4 | 0.634 |
| ECR | — | — | 0.650 | — | — | **0.692** |

The PPG selector improves hit rate over the incumbent by +0.0105 across 19 long
folds (9 wins, 8 ties, 2 losses) and +0.0143 across seven modern folds (4 wins, 2
ties, 1 loss). It
also improves over `fitted_two_stage` by +0.0096 long and +0.0119 modern. The market
hit lift survives at +0.0067, but its standard error is 0.0201 and its empirical
95% interval spans −0.0328 to +0.0461. More importantly, it loses 0.022 NDCG to ECR
and wins only one of five folds on that measure. The cutoff-safe rebuild therefore
confirms a useful top-K inclusion pattern, not a production-ready ordering win.

The old season-NDCG selector's long-horizon advantage disappears after replacing the
Week 1 proxy (0.621 to 0.602). That failure is useful: it identifies the earlier gain
as timing-sensitive and rejects promotion. The surviving PPG selector now chooses
athletic QB/TE, the full stack at RB, and the cutoff-rostered stage at WR. No single
wide feature stack wins; position/regime selection remains the hidden pattern.

Decision: **wire dated cutoff state into the experimental metrics and fitted report,
but do not change the live rank key.** The two definitions, their 2026 selected
sources, and a digest of all 611 pending predictions are frozen in
`experimental_freeze_2026.yaml` (fit fingerprint `4d65a1b8905e267b`). Production
stays on the existing fitted measurements until the untouched 2026 outcome is graded.
Another broad returning-player data pull is deferred; the next material data project
is the separate rookie/college population, not another round of tuning on these five
ECR folds.

The freeze now includes the complete 611-row CSV, not only a configuration digest.
`patron grade-prospective` verifies its SHA-256, refuses to grade before January 15,
2027 or when outcomes appear partial, and attaches only fresh actual PPG, games,
season points, VOR, and availability value. Passing requires prospective overall hit
rate above both the incumbent and dated ECR plus NDCG at least equal to ECR. The
command writes a review artifact and never promotes a model automatically.

## 15. Focus round: purge of no-signal metrics (2026-08-31)

A Moneyball-style pass with one instruction: state the goal sharply, and remove every
metric the evidence had already condemned but the pipeline still computed. The goal
statement now opens `docs/reference/metrics.md`; the consolidated removal record is that
document's graveyard table. What changed:

**Dead code and display (no model change):** the `qb_context` arithmetic (export-only
since the §10 removals), the `BUY`/`TD-luck` flags and the whole
TD-over-expectation module (§4: noise-level), `projected_ceiling` (monotone in
PPG + volatility, no renderer), the stale `v2_score` fallback references, the
`market_rank_delta` join, and the unread enrichment columns (rushing/receiving EPA
rates, neutral early-down/no-huddle rates, expected yards/TDs per game, six NGS
columns nothing consumed). The fixture validation's advisory flag stage now compares
only the surviving flag vocabulary (`age`, `Ngms`, `ESPN-only`).

**Component-branch surgery (changes fitted feature values, harness-gated):**

1. The "opportunity-implied targets" anchor was collapsed algebraically — it *is*
   targets (§5) — folding its 0.15 weight into the historical target rate (0.35
   total). Verified as a numeric no-op (max |Δ| 3.6e-15) before the real cuts.
2. The 25% WOPR role blend was removed (§5: double-counted target share).
3. The teammate-competition multipliers (`target_availability`/`backfield` ratios and
   their depth-chart blend) were removed (§6a: inert in every era). Competition now
   enters only through the player's own shrunk share and depth-role factor, and the
   `target_availability_multiplier`/`backfield_availability_multiplier` override
   levers are gone; team volume/scoring multipliers remain.
4. The dedicated big-play bonus projection was removed (§4/§5: anti-predictive
   partial −0.30). League bonus *scoring* is untouched; bonus points now reach the
   component branch only through the heavily shrunk residual, keeping the branch an
   estimate of full league-scoring PPG.

**Experimental prune (rejected-only):** deleted the fit specs the record explicitly
rejected and that are neither frozen-selector factors nor awaiting the 2026 fold:
snap (§12 "do not promote"), standalone vacated, capital/commitment,
hidden-residual (§13 rejection), cutdown stack, the adaptive blend (§13 "rejected"),
both hard-coded position hybrids, and the market-plus-everything pair (§12
over-parameterized). 46 specs → 28; the two frozen adaptive selectors, their six
factor models, and every spec those factors depend on are untouched, and
`tests/test_prospective_freeze.py` passes unchanged. The metric catalog dropped the
nine no-signal entries (`td_over_exp`, `projected_bonus_pg`, `projected_wopr`,
`wtd_opp`, `qb_context`, `team_scoring_context`, `team_pass_volume`,
`team_rush_volume`, `teammate_competition`) plus `projected_ceiling`.

**Acceptance (to the rule in §0):** the full 22-fold rebuild after the surgery and
prune (12,751 forecast rows) shows the cuts cost nothing at the top of the board:

| slice | before | after |
|---|---|---|
| OVERALL `fitted_season_points`, modern | 0.638 (7–0) | **0.638 (7–0)** — identical |
| OVERALL `fitted_season_points`, long | 0.599 (15–4) | **0.599 (15–4)** — identical |
| OVERALL `fitted_two_stage`, long | 0.600 (14–5) | **0.601 (15–4)** |
| QB/RB `fitted_ppg`, WR `fitted_season_points` (live keys) | — | hit rates unchanged to the third decimal in both windows |
| `component_proj_ppg` WR | 0.706 / 0.601 (modern/long) | **0.710 / 0.614** |
| `component_proj_ppg` TE long | 0.466 | **0.473** |
| `proj_ppg` OVERALL modern / long | 0.612 / 0.567 | 0.614 / 0.566 (±1 fold slot, noise) |
| vs market (ECR 0.650) | −0.003 | −0.003, unchanged |

Reading: four condemned terms were live inside a fitted feature and their removal
moved no top-K membership anywhere, while the receiver component branch — the piece
that carried the WOPR blend, the competition multiplier, and the bonus projection —
got slightly *better* at WR and TE. The model is meaningfully smaller (two fewer
input modules, ~10 fewer exported columns, 18 fewer fit specs, one fewer override
lever) at zero cost to the goal. `tests/test_prospective_freeze.py` passes untouched;
the fit fingerprint changed with the spec prune, so this rebuild also refreshed the
live-board artifact.

## Recommended order (updated)

Every step is accepted or rejected by the §0 tables: positive lift and more folds won
than lost against the best baseline, modern window, without losing the long horizon.

Done: shrinkage prior (§1), active-game denominators (§3c), participation-position
fallback and legacy depth charts (§3a/3b, other session), ranking evaluation (§0),
fitted ranker and live wiring (§8), parity, contract, overall validation, and the
market/xFP/QB/team/NGS/return research pass (§9), additive structural round (§12),
combine/career residual screen, adaptive position-model experiment (§13), dated
cutoff availability, the versioned 2026 prospective freeze (§14), and the no-signal
purge and goal refocus (§15).

Next, by expected impact:

1. **Grade the frozen 2026 selectors** — require the untouched prospective forecast
   to retain its incumbent and market lift before any live promotion. Do not tune a
   new selector family on the same five historical ECR folds.
2. **Rookie/college model** — returning-player backtests cannot solve the largest
   missing population; add draft capital, age, college production, and athletic data.
3. **Longer dated ADP/ECR history** — the market is still the strongest missing-data
   aggregator, but three trainable folds cannot support a 20-feature blend.
4. **TE/RB nonlinear opportunity challenger** — keep it outside the frozen selector;
   revisit only with new folds or a separately held-out hypothesis.
5. Complete the dated soft-injury/recovery archive, then cumulative RB touches and the
   weekly re-rank. Market folds now stop at the exact archived ECR date and use official
   club IR/PUP/NFI/suspension transactions for 31 teams. That structured status raises
   value-capture precision over the plain season model, but the next-gen challenger
   still loses ECR on net swap VOR; adding same-date ECR to the games leg does not turn
   parity into independent alpha. Do not add the earlier QB-efficiency multiplier: the
   direct EPA/CPOE experiment was redundant (§9).

## Reproducing the cuts

All tables above were computed from `metric_backtest_predictions.parquet` with Polars
only (`rank()` + `corr()` for Spearman). Tiers are ranks of `proj_ppg` within forecast
season and position; the draftable slice is the top QB14/RB40/WR50/TE14 by `v2_score`
per fold; ablations recompute `max(individual_prior + td_adj + bonus_adj, 0) ×
age × team_context × depth_role` with one factor removed.
