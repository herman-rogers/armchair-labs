# Patron metrics: the goal, the surviving catalog, and the graveyard

This document is the contract for the player-ranking metrics. It states the one goal
every metric serves, separates what the engine observed from what it projects, and
records — permanently — every metric the backtest disproved, so a dead idea cannot be
reinvented as a fresh insight.

## The goal

Everything in this system serves one question:

> **Rank the top of the draft board correctly on next-season season points, where
> season points = per-active-game PPG × games played.**

Concretely, the produced ranker is `fitted_season_points = fitted_ppg × fitted_games`
(walk-forward ridge models over a handful of proven features), and the score is hit
rate @K on the slice a ten-team league actually drafts (QB12/RB24/WR36/TE12, top-60
overall VOR) — never whole-population correlation, which any metric passes by
separating starters from players who leave the league.

Two bars must be beaten, per fold, in both the modern (2019–2025) and long (2004–2025)
windows:

1. **The naive baselines** — recency-weighted three-year PPG and last-season PPG,
   rankings anyone can produce with no model.
2. **The market** — dated preseason FantasyPros ECR. A model at parity with the
   market earns nothing at the draft table; the edge lives only where the model
   disagrees with the market and is right.

What twenty-two seasons of folds say about where that edge is: **availability, not
efficiency.** Roster/status features cut games-played error by more than any
efficiency stat ever moved a hit rate, while every glamour signal tested (EPA, CPOE,
NGS separation, snap trends, team-context narratives) was redundant or noise once
recent production and role were known. The per-active-game half of the product still
carries most top-of-board variance — availability is half the product, and the
underpriced half.

The acceptance rule for any model change: positive lift and more folds won than lost
against the best baseline in the modern window, without losing the long horizon
(`just metric-report` produces the tables). Anything that fails is removed, and
recorded in [the graveyard](#the-graveyard-removed-after-backtesting).

## Executive review

The original model is a strong league-specific retrospective board. Its best ideas
are exact Sweaty Plays scoring, per-game production rather than season totals,
position-aware value over replacement, explicit long-touchdown bonus scoring, weekly
score distributions, and written manual overrides for facts that are not in the prior
season's data.

The primary weakness is the question v1 answers. It ranks last season's production;
it does not independently estimate next season's production. A four-game breakout can
therefore rank on its raw PPG, and a trade can leave the old team environment
embedded in the number. The `Ngms` and override annotations warn about some of this
uncertainty but do not make the ranking itself respond to it.

V2 preserves v1 as evidence and changes the ranking question to projected PPG and
projected season points. What remains after the ablation rounds is deliberately
plain:

- recency-weighted multi-year production;
- small-sample shrinkage toward the all-player positional prior;
- gradual position-specific age curves;
- the current-depth-chart role factor;
- a bottom-up component stat line from shrunk shares of official team volume;
- explicit future assumptions for trades and role changes;
- nflverse route-opportunity participation, targets per route opportunity, official
  pass attempts/dropbacks, high-value rushing/receiving usage, and injury-report
  history;
- expected games, so injury risk affects draft rank without contaminating
  per-active-game PPG;
- confidence fields measuring support for both production and availability estimates.

The hand-set constants themselves no longer decide the board: the walk-forward fitted
rankers (§10) learn the weights from completed folds, and the board sorts on their
outputs.

## V1: historical production metrics

### League points and PPG

Every player-week is scored under the league's rules:

- full PPR;
- 0.1 points per rushing or receiving yard;
- 0.04 points per passing yard;
- four-point passing touchdowns and six-point rushing/receiving touchdowns;
- minus two for interceptions and lost fumbles;
- the configured two-point conversions and special-teams touchdowns;
- plus two for a 40–49-yard touchdown and plus three for a 50+-yard touchdown. A
  passing bonus goes to both passer and receiver; a rushing bonus goes to the rusher.

`PPG = total league points / games`

PPG is preferable to a season total when comparing rates, but it is not an
availability projection. Games remain visible and small samples are flagged.

### Historical VOR

`VOR = PPG - replacement PPG at the player's position`

Replacement is configured at QB12, RB25, WR35, and TE12 for this ten-team league.
Only players meeting the baseline games threshold can define replacement; small-
sample players can still appear on the board. Manual deltas produce `adj_vor`, which
is the v1 sort key.

This is the right positional-scarcity concept, but the baseline ranks should be
revisited if league size, starting slots, flex rules, benches, or waiver behavior
change. In season, actual free-agent availability is a better replacement pool than
fixed positional ranks.

### Target share and air-yards share

V1 retains nflverse's weekly target share and air-yards share. The v1 aggregator
weights weekly shares by the player's own targets. That is better than a flat weekly
mean for an injury-shortened week, but it is not a true season share.

V2 therefore adds `season_target_share = player targets / team targets` and
`season_carry_share = player carries / team carries`, and the projection itself works
from shrunk shares of official team volume. Weighted opportunity (carries + 2.2 ×
targets) and WOPR were removed — see the graveyard: the first was an algebraic
restatement of the targets and carries the projection already uses separately, the
second double-counted target share.

### Weekly floor and volatility

Floor is the 25th percentile of weekly league points. Volatility is their sample
standard deviation. Both include the league's long-touchdown bonuses.

These are useful lineup descriptors, especially when choosing floor as a favorite or
ceiling as an underdog. They are noisy in short samples and do not measure a player's
future range by themselves. V2 leaves them as historical evidence rather than using
them directly in projected PPG.

### Age and flags

V1 flags an RB at age 27.5 or later, measured on September 1 of the draft season, and
the number of games in a small sample (`Ngms`). Age is a useful warning but the
binary RB cliff is too coarse to be a projection. V2 applies a bounded gradual curve
by position while retaining the flag. The former `BUY`/`TD-luck`
touchdown-regression flags are gone (graveyard).

### Manual overrides

V1's injury and situation deltas are an important strength. A delta is expressed in
PPG/VOR rather than as a hard rank and always carries a reason. Unmatched overrides
fail loudly. V2 applies the same delta to projected VOR, then adds a separate
projection-context file for future teams and roles.

## V2: combined historical and projected-stat-line model

V2 retains every v1 metric as evidence but does not merely average the two finished
rankings. It combines two explainable PPG estimates:

1. a normalized historical branch built from v1 league-scored PPG; and
2. a bottom-up branch that projects the next-season stat line from player role,
   efficiency, team volume, and scoring environment.

The bottom-up branch receives 65% of projected PPG by default and the historical
branch receives 35%. Both branches are features of the fitted rankers, which set the
board's actual sort (§10).

### 1. Historical PPG branch

Each player-season receives:

`season weight = 0.55 ^ years old × games`

The weighted PPG becomes `historical_ppg_prior`. It is shrunk toward the rosterable
positional PPG prior based on effective games:

`reliability = effective games / (effective games + 8)`

`individual prior = reliability × historical PPG + (1 - reliability) × position PPG`

The age curve and the depth-chart role factor are then applied. This yields
`prior_branch_ppg`. Nothing else touches the prior branch: TD regression, big-play
regression, and the team-context scalar were all removed after the ablations
(graveyard).

### 2. Team context

For every team-season, v2 calculates official pass attempts, dropbacks, and carry
volume per game; offensive TDs per game; and passing TDs per target and rushing TDs
per carry.

Team context enters only the component branch, as guarded ratios (0.75–1.25) of the
projected team's volume and TD environment to the environments already embedded in
the player's weighted history. There is no quarterback-quality term and no teammate
competition term — both tested inert or negative across every era since 2004
(review §6a).

### 3. Projected opportunity

V2 calculates target share against official team pass attempts and carry share against
team carries. Each is recency-weighted and shrunk according to its target/carry
sample. The share itself, plus the depth-chart role factor, is the whole role
estimate: the former WOPR blend and the teammate-availability multipliers are gone
(graveyard).

The nflverse participation feed identifies which offensive players were present on a
dropback, but it does not prove that every eligible RB/TE released into a route. V2
therefore names this input `route_opportunities`, not charted routes. It separately
shrinks route-opportunity participation and targets per route opportunity (TPRR).

When participation is available, projected targets use:

`route targets = projected team dropbacks × route-opportunity participation × TPRR`

The route estimate receives 55% weight. The remaining branch blends target share
against official attempts (65%) with the historical target rate (35%). Without
participation data, v2 falls back cleanly to that remaining branch.

Projected carries use 75% team carry-share volume and 25% normalized historical
carries/game. All historical anchors respond to projected team pass/rush volume, and
the explicit opportunity multiplier applies to volume rather than efficiency.

### 4. Projected efficiency and stat line

The component model separately shrinks:

- catch rate;
- receiving yards per target;
- rushing yards per carry;
- receiving TDs per target;
- rushing TDs per carry.

Receiving touchdowns are decomposed into projected end-zone targets, end-zone
conversion, and non-end-zone TD rate. Rushing touchdowns similarly separate carries
inside the five, goal-line conversion, and non-goal-line TD rate. Red-zone targets and
carries are exported alongside the narrower high-value opportunities. General TD
rates retain 30% weight so sparse charting samples cannot dominate.

Air-yards share is recency-weighted and shrunk. Its ratio to target share measures
role depth relative to the position, producing a bounded `air_yard_factor` for yards
per target. Receiving and rushing TD rates are also scaled by the projected team's
passing- and rushing-TD environments.

For QBs, official attempts are projected from team passing volume. Yards per attempt,
passing-TD rate, and interception rate are shrunk separately, then converted into the
passing stat line. QB rushing yards and rushing TDs remain separately projected.

The exported component stat line contains targets, carries, receptions, passing/
rushing/receiving yards, split TDs, and interceptions per game. A small shrunk
residual preserves the scoring not explicitly decomposed: two-point conversions,
fumbles, special-teams production, and — since the 2026-08-31 round — big-play bonus
points, whose dedicated projection tested anti-predictive (graveyard). Scoring those
components under the league rules produces `component_proj_ppg`.

### 5. Combining the branches

`projected PPG = 35% prior_branch_ppg + 65% component_proj_ppg`

The weight is configurable. `proj_ppg` and its VOR (`proj_vor`, `adj_proj_vor` with
overrides) remain as the hand-built projection, but since the 2026-08-29 review they
are **inputs and diagnostics, not the sort**: `component_proj_ppg` and
`historical_ppg_prior` are features of the walk-forward fitted rankers, and the board
ranks on the fitted outputs (see §10).

### 6. Floor and volatility

Floor and volatility describe the distribution of the same weekly points, so they do
not get added to expected PPG. Their multi-year ratios to PPG are shrunk toward
positional profiles and applied to the forward mean:

- `projected_floor`: projected 25th-percentile score;
- `projected_volatility`: projected weekly standard deviation.

These are descriptive fields and roster-risk inputs (the League team-strength rating
rescales `projected_volatility` to the fitted mean). `projected_ceiling` was removed
as a redundant restatement of the same two numbers.

### 7. The graveyard: removed after backtesting

Every metric here was a believed edge that the walk-forward backtest disproved. It is
recorded so the same story cannot be re-sold to a future session as a fresh insight.
Full evidence lives in `docs/v2_metrics_review.md` (section cited per row).

| Removed | The story we believed | What the folds showed | When |
|---|---|---|---|
| `td_over_exp` + `BUY`/`TD-luck` flags | "The regression engine — the single most exploitable signal" | Noise-level next-season signal; players repeatedly earn goal-line work (§4) | 2026-08-31 |
| `td_regression_adjustment` (prior branch) | Partial TD regression improves the prior | Noise-level at every horizon (§4) | 2026-08-29 |
| `projected_bonus_pg` + `bonus_regression_adjustment` | Big-play bonus rate is projectable, and it's this league's edge | Anti-predictive (partial −0.30 vs next-season VOR); bonus scoring itself stays in league points and reaches the component via the shrunk residual (§4, §5) | 2026-08-31 |
| `team_context_factor` (±18% prior-branch scalar) | A better destination environment lifts the player | Actively hurt — QB draftable Spearman 0.561→0.695 with it removed (§4) | 2026-08-29 |
| `qb_context` | Quarterback quality is a rankable input | Circular for non-movers; the fitted experiments found only a tiny per-attempt effect, redundant after the prior (§6a, §9) | 2026-08-31 |
| Team pass volume as a prior-branch signal | More attempts, more receiver points | Slightly negative partial (−0.025); survives only as component-branch volume ratios (§6a) | 2026-08-29 |
| Teammate-competition multiplier (`target_availability`, strongest-teammate share) | A target hog on the roster caps your share | Inert (partial ≈0) in every era since 2004 (§6a) | 2026-08-31 |
| WOPR blend (25% of receiving role) | Air-yard role adds to target share | Double-counted target share; air-yard information already carried by `air_yard_factor` (§5) | 2026-08-31 |
| `wtd_opp` (carries + 2.2 × targets) | One number for volume | Algebraically the targets and carries the model already uses; its "independent anchor" weight was just 0.35 on targets (§5) | 2026-08-31 |
| `v2_score`, `floor_vor`, `ceiling_vor`, `availability_adjusted_vor`, `expected_season_points` composites | Blending floor/ceiling/availability makes a better sort | The hand-built overall order lost to *doing nothing*; fitted rankers superseded it (§0) | 2026-08-29 |
| Rosterable-pool shrinkage prior | Shrink small samples toward starters | Pulled every 1-game cameo toward QB12 production; cost ~0.09 Spearman (§1) | 2026-08-29 |
| `projected_ceiling` | A distinct upside number | Monotone restatement of PPG + volatility | 2026-08-31 |
| Snap-share trend, raw vacated share, EPA/CPOE/NGS glamour stats, market-plus-everything stack | More signals, better model | Redundant after production+role, or over-parameterized on five market folds (§9, §12) | 2026-08-30 |

What survived every round: the all-player shrinkage prior, last-season PPG at nearly
full weight, the age curve (RB above all), the depth-chart role factor, shrunk shares
of official team volume, high-value (end-zone/goal-line) usage splits, and
availability — the last being the largest single discovery in the project's history
(§12: status features cut games-played MAE from ~4.4 to ~3.0).

### 8. Current depth charts and explicit future assumptions

The latest published nflverse depth-chart snapshot automatically supplies current NFL
team, position, and depth rank. Current team supersedes the player's last historical
team unless a manual projection override is present. Depth rank applies a bounded
role adjustment: rank one is neutral, while ranks two and below progressively reduce
opportunity. The snapshot date is exported because a dated depth chart is evidence,
not timeless truth.

[`projections.yaml`](../src/patron/config/projections.yaml) supplies facts historical
data cannot know. A player entry can set `projected_team`, `opportunity_multiplier`,
and a visible reason. A team entry can adjust pass/rush volume or scoring for every
player on that team. Unmatched player entries fail the build so a misspelled trade or
role change cannot quietly disappear.

### 9. Injury-based expected games

PPG remains an estimate of ability per active game. Expected games are modeled
separately from nflverse offensive participation and official injury-report history.
Observed availability receives 75% of the player sample and injury designations
receive 25%, with OUT, DOUBTFUL, and QUESTIONABLE carrying decreasing missed-game
equivalents. The result is shrunk toward a configurable 94% availability prior over a
17-game shrinkage sample.

V2 exports `expected_games` and `projected_availability`. They are features of the
fitted games model (`fitted_games`), which is what turns fitted PPG into fitted season
points; injury risk therefore affects draft rank without contaminating
per-active-game `proj_ppg`.

### 10. Fitted rankers and the board sort

`src/patron/metrics/fit.py` fits walk-forward per-position ridge models on the
completed backtest folds (`metric_report.yaml` → `fit.models`): `fitted_ppg`
(per-active-game), `fitted_games` (calibrated availability on the full population),
`fitted_season_points = fitted_ppg × fitted_games`, plus the direct and two-stage
season-points variants that compete in the report. The live board applies the stored
2026 models only when the report's artifact matches the current fit config, season,
depth-chart cutoff, and snapshot.

`league.yaml` → `projection_rank_key` picks each position's within-position key from
the report's evidence (`fitted_ppg` QB/RB, `fitted_season_points` WR, last-season
`ppg` TE); `projection_overall_key` (`fitted_season_points`) sets the cross-position
order as per-game VOR against its own replacement (`v2_overall_vor`). The League
team-strength rating reads the same fitted mean and games (`roster_*_columns`).

### 11. Consensus fallback and position overrides

Players without usable tape enter the board from the FantasyPros consensus snapshot
for the draft season, valued by rank-matching to model-rated players at the same
position and tagged `rank_source = market`; see `docs/v2_metrics_review.md` §11.
`league.yaml` → `position_overrides` corrects nflverse positions by GSIS id before the
board filter (two-way players).

### 12. Frozen Adaptive PPG-selector board

The `adaptive` system is a third selectable 2026 forecast, not a replacement for v2.
It joins the immutable `data/static/experimental_2026_predictions.csv` snapshot onto
the current player pool and ranks every returning player by
`fitted_adaptive_ppg_hybrid`. The score is expressed as season points and then as
per-scheduled-game VOR against its positional replacement, exactly like the overall
backtest comparison. The source selected before the season remains visible by
position: athletic QB/TE, full-stack RB, and cutoff-rostered WR.

Players without a frozen model score retain the same explicitly tagged consensus
fallback as v2. Current team, ownership, and injury display can update, but the model
score cannot: the build verifies the snapshot's SHA-256 against
`experimental_freeze_2026.yaml`. Using the shadow board therefore does not mutate the
predictions that `patron grade-prospective` will evaluate after the season.

### 13. Next-generation shadow challenger

The next-generation track does not reinterpret the frozen Adaptive board. Its
returner path preserves the aligned target decomposition—PPG from quality and role,
games from availability and cutoff roster status, season points as their product—and
removes `projected_availability`, the exact scaled duplicate of `expected_games`, from
the games feature list. Its rookie path is trained only on rookie rows. The first
scorecard showed that age, combine measurements, round buckets, and a coarse
major-conference flag all made the ranking worse than NFL draft capital, so the core
uses draft capital plus an undrafted indicator and leaves the other fields as
reportable hypotheses. The two mutually exclusive populations are coalesced only
after each PPG × games forecast exists. All `fitted_nextgen_*` models are report-only.

The report's `market_disagreement_results` is the promotion gate that broad accuracy
could not provide. It compares each ranker with dated overall FantasyPros ECR on the
same players and exact archived cutoff. The primary scores are precision among
model-top-60/ECR-outside-60 calls, recovery of actual top-60 players ECR missed, net
realized VOR of model-only versus ECR-only swaps, and false-positive cost. A player who
only beats ECR's rank directionally but remains outside the actual draftable tier is
not a hit. FantasyPros ECR is an expert-consensus price proxy, not ADP. ESPN's observed
ADP is written as one daily panel under `data/outputs/market_snapshots/`, creating a
true prospective ADP archive. The historical series is backfilled only where the
vintage is provable (2026-08-31): dated ECR now reaches back to 2011 — 2020 recovered
from the archive's combined-offense pages, 2011–2019 from timestamped Wayback captures
— and dated observed ADP (MFL real-league drafts 2011+, FFC 2008–2010) sits beside it
in `data/static/market_adp_backfill.csv`. Provenance, derivations, and the honest gaps
(no 2010 ECR, nothing for 2007) are in `data/static/market_backfill_manifest.md`. The
market comparison therefore rests on fifteen ECR folds rather than five, and
2011–2020 served exactly once as held-out market folds for every spec tuned on
2021–2025.

Market-era availability uses the ECR snapshot date as its information boundary.
Official NFL club transaction archives supply dated IR, PUP/NFI, suspension, and
roster state for 31 teams in 2020–2026; Dallas and older seasons retain an explicitly
labelled ESPN fallback. A same-date ECR-conditioned games challenger measures the
effect of soft-injury/recovery information missing from the public archive, but is not
presented as independent model signal.

### ESPN's deliberately limited role

ESPN is not a statistical projection source. It remains useful for private-league
state that nflverse cannot know: which fantasy manager owns a player, who is a free
agent, fantasy lineup placement, ESPN's live injury display, and its current PPR
draft-room order. Those fields power roster and waiver views, but ESPN projected
points, lineup slots, ownership rates, and ranks do not feed V2's football projection.
Historical statistics, usage, teams, depth charts, and injury modeling are
nflverse-first.

An ESPN skill player with no usable NFL production remains visible in the connected
player and roster tables as an `ESPN-only` row. Its model fields stay null rather than
receiving a fake zero; `espn_draft_rank` places it into the otherwise unchanged Patron
order, while `espn_adp`, `rank_source`, and the row note make that fallback auditable.

### League team-strength rating

The League dashboard evaluates the complete roster rather than trusting the lineup a
manager happens to have saved in ESPN. For each of 12,000 deterministic scenarios, it:

1. samples every player's active/inactive state from `projected_availability`;
2. selects the highest-projected legal active lineup using the ESPN league shape—one
   QB, two RBs, three WRs, one TE, and one RB/WR/TE flex;
3. sums the selected V2 per-active-game means and their projected variances; and
4. records whether every starting slot was filled and how many points came from a
   player outside the full-strength lineup.

The scenario average is `expected_weekly_points`. It inherently values bench depth:
a reserve contributes only in scenarios where that player actually enters the best
active lineup. `bench_rescue_points` reports that expected contribution explicitly,
and `lineup_coverage` is the probability that all eight skill-position slots can be
filled by active rostered players.

Weekly uncertainty uses the law of total variance:

`team variance = average(selected player variances) + variance(scenario lineup means)`

The first term is normal week-to-week scoring volatility. The second is availability
and replacement risk—the stars-and-scrubs penalty when an absent star exposes a weak
bench. `weekly_risk` is the square root of that variance, in fantasy points, and
`weekly_floor = expected weekly points - 0.674 × weekly risk` approximates a 25th-
percentile outcome. Player outcomes are currently treated as independent, so shared
QB/receiver and game-environment covariance remains a future refinement.

The League board also reports **risk-adjusted VOR**: in every scenario the best legal
*active* lineup is chosen on the board's overall VOR (`v2_overall_vor`), and
`risk_adjusted_total = mean(scenario lineup VOR) − 0.674 × sd(scenario lineup VOR)`,
an approximate 25th percentile. It penalises stars-and-scrubs construction — an absent
star replaced from a weak bench — on the same scale as the power rank, without mixing
in ordinary week-to-week scoring noise that every roster carries. Market-placed
players (§11) participate on the model's scale rather than via ESPN projections.

Overall `team_rank` sorts the full-strength lineup VOR (`ranking_total`). Risk is intentionally separate rather
than silently penalizing or rewarding volatility. The displayed `team_score` maps the
expected-points distribution to a league-relative 0–10 index centered at 5.0, with 1.5
rating points representing one league standard deviation.

Separately, an ESPN full-season projection is used only inside the league team-strength
simulation for a rostered rookie or returnee with no nflverse tape; its volatility is
estimated from the position's V2 ratio and its availability is set to the configured
94% prior. This does not populate the player's blank V2 fields or change model-backed
ranks. The UI reports the number of these fallbacks. Kicker and defense points are
excluded because the V2 player projection does not produce their weekly mean and
variance. ESPN division IDs and names group the cards but do not change the overall
rank.

## V2 additions and updates

| Area | V1 | Combined v2 |
|---|---|---|
| Production | Latest-season PPG | 35% normalized PPG branch + 65% projected stat line, both features of the fitted rankers |
| Sample size | Flag only | Shrinkage toward the all-player pool, plus confidence |
| VOR | Historical mean VOR | Fitted per-game and season-points VOR (`v2_overall_vor`) |
| Touchdowns | Season totals | Component-level split TD rates from end-zone/goal-line usage |
| Big-play bonuses | Historical scored points (kept) | Reach the component branch through the shrunk residual only |
| Target share | Target-weighted weekly average | True, shrunk share directly projects targets |
| Carry share | Missing | True, shrunk share directly projects carries |
| Air-yards share | Display only | Bounded `air_yard_factor` on yards per target |
| Floor/volatility | Historical descriptors | Projected descriptors and roster-simulation risk inputs |
| Team context | Embedded in results | Component-branch volume and TD-rate ratios only |
| Age | Binary RB flag | Gradual position curve plus original flag |
| Future changes | Manual VOR delta | Projected team, team volume/scoring multipliers, role multiplier |
| Dashboard/API | One board | Selectable v1/v2/frozen-adaptive boards with component metrics |
| League comparison | Opponent positional weaknesses | Overall roster rank/strength, rookie fallback visibility, and ESPN divisions |
| Routes/TPRR | Missing | nflverse route-opportunity participation and regressed TPRR project targets |
| High-value usage | TD rate only | Red-zone/end-zone targets and carries inside the five project split TDs |
| Passing volume | Target proxy | Official attempts/dropbacks project QB and receiver opportunity |
| Depth charts | Manual only | Latest nflverse team/rank automatically changes context and role |
| Availability | Games flag | Expected games and `fitted_games`, the season-points multiplier |

## Implementation coverage and remaining roadmap

The percentages below are approximate implementation coverage, not claims about
forecast accuracy. "Direct" means the input is represented in the projected stat
line. "Proxy" means v2 responds to a related signal but cannot distinguish the more
specific football usage we ultimately want.

### Bottom-up projection core

| Area | Approx. coverage | What v2 does now | Important gap |
|---|---:|---|---|
| Targets | 90% | Projects route opportunities × TPRR, blended with share of official attempts and historical target rate | Participation cannot distinguish pass protection from a released route |
| Carries | 90% | Projects carry share and team carry volume, with current depth rank, red-zone work, and goal-line work | Situation-neutral run rate and two-minute usage remain implicit |
| Routes | 65% | Derives eligible-player dropback participation and TPRR from nflverse play participation | This is a route-opportunity proxy, not a charted route for every player |
| Red-zone opportunities | 80% | Separates red-zone/end-zone targets, carries inside the five, and their conversion rates | Quarterback sneak competition and expected red-zone trips are not yet separately forecast |
| Efficiency | 75% | Separately projects catch rate, receiving yards per target, rushing yards per carry, and TD rates | No yards per route, separation by route/coverage type, or explicit efficiency aging beyond the overall age factor |
| League scoring | 90% | Scores the projected components under the league rules and retains a shrunk residual for rare scoring | Fumbles, two-point conversions, and special-teams scoring are not projected as separate events |
| Weekly range | 75% | Projects floor and volatility from shrunk historical distribution shapes around the new mean | No matchup, injury, or role-state mixture distributions |

The core is therefore roughly two-thirds to three-quarters of the way to a complete
transparent stat-line projection. The missing enrichment inputs below are less
complete and are the main reason the model should not yet be treated as a full
play-by-play projection system.

### Remaining enrichment inputs

| Upgrade | Data availability | Projection use today | Assessment and next step |
|---|---:|---:|---|
| Route participation | 75% proxy | 70% | nflverse participation now drives projected route opportunities. Replacing on-field dropbacks with charted route releases would remove blocking false positives. |
| Targets per route | 75% proxy | 75% | Regressed targets per route opportunity is a primary target estimate. True charted routes would upgrade the denominator. |
| Goal-line/end-zone usage | 90% | 80% | nflverse play-by-play supplies red-zone targets/carries, end-zone targets, carries inside the five, and conversions. Next add team red-zone-trip forecasts and QB sneak competition. |
| Official pass attempts | 100% | 90% | nflverse team attempts and attempts-plus-sacks dropbacks now set team/QB volume and target-share denominators. Situation-neutral pace remains missing. |
| Current depth chart | 90% | 80% | Latest nflverse team/rank automatically changes projected team and role, with timestamp and manual override support. Camp-battle probabilities and formation-specific depth remain missing. |
| Injury-based games played | 85% history | 80% | nflverse participation/history plus point-in-time official IR/PUP/NFI/suspension transactions project expected games, season points, and availability-adjusted VOR. Complete historical soft-injury, recovery-stage, and practice-state archives remain missing. |

This distinction now appears directly in rankings: `proj_ppg` remains a per-active-
game estimate, while `expected_season_points = proj_ppg × expected_games` and
availability-adjusted VOR carry season-long durability.

### Next refinements after upgrades 1–5

1. Replace route opportunities with fully charted routes if a stable, licensed source
   becomes available; retain the nflverse proxy as a fallback.
2. Add situation-neutral pace, red-zone trips, two-minute roles, designed QB runs, and
   quarterback sneak competition.
3. Model depth charts as role probabilities during camp rather than treating each
   published rank as certain.
4. Add injury type, surgery/recovery stage, cumulative workload, and current preseason
   availability to the expected-games model.
5. Backtest all new coefficients against held-out seasons and calibrate expected games,
   targets, carries, touchdowns, and PPG separately.

After those inputs, the most valuable model-level work is held-out-season backtesting,
calibration by position and confidence bucket, cumulative RB-touch aging, ADP/auction
price deltas, and draft tiers with uncertainty bands. Differences of 0.1 projected
VOR should not be presented as materially precise rank gaps.

## Version selection and artifacts

`patron board` writes:

- `board.json`: compatibility alias for v1;
- `board_v1.json` and `board_v1.md`: historical board;
- `board_v2.json` and `board_v2.md`: production forward-projection board;
- `board_adaptive.json` and `board_adaptive.md`: frozen 2026 Adaptive PPG shadow board.

The API accepts `version=v1|v2|adaptive` on `/api/board`, `/api/players/{id}`, and
`/api/positions`. `/api/status` reports availability and counts for all systems. The
dashboard tabs call those endpoints; switching tabs changes both the sort key and the
visible metric columns.
