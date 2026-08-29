# Patron metrics: model review, v1 catalog, and v2 projections

This document is the contract for the player-ranking metrics. It separates what the
engine observed from what it projects, explains how team context enters the model,
and names the places where the numbers are useful without pretending they are more
precise than the inputs allow.

## Executive review

The original model is a strong league-specific retrospective board. Its best ideas
are exact Sweaty Plays scoring, per-game production rather than season totals,
position-aware value over replacement, explicit long-touchdown bonuses, opportunity
signals, weekly score distributions, and written manual overrides for facts that are
not in the prior season's data.

The primary weakness is the question v1 answers. It ranks last season's production,
with flags around it; it does not independently estimate next season's production.
A four-game breakout can therefore rank on its raw PPG, a player can carry a volatile
long-touchdown rate forward, and a trade can leave the old quarterback and teammate
environment embedded in the number. The `Ngms`, `TD-luck`, and override annotations
warn about some of this uncertainty but do not make the ranking itself respond to it.

V2 preserves v1 as evidence and changes the ranking question to projected PPG and
projected VOR. It adds:

- recency-weighted multi-year production;
- small-sample shrinkage toward a rosterable positional prior;
- partial rushing/receiving touchdown regression;
- regression of volatile big-play bonus production;
- gradual position-specific age curves;
- projected-team context: quarterback quality, offensive volume, scoring environment,
  and the opportunity occupied by the strongest teammate;
- explicit future assumptions for trades, quarterback changes, and role changes;
- nflverse route-opportunity participation, targets per route opportunity, official
  pass attempts/dropbacks, high-value rushing/receiving usage, current depth charts,
  and injury-report history;
- expected games and availability-adjusted VOR alongside per-active-game PPG;
- confidence fields measuring support for both production and availability estimates.

V2 is intentionally labeled a transparent projection, not a trained forecasting
model. Its coefficients are bounded heuristics and need backtesting as additional
seasons accumulate. The formulas are visible so a surprising rank can be explained.

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

### Weighted opportunity

`weighted opportunity = carries + 2.2 × targets`

Targets receive extra weight because they include a full PPR reception opportunity
and generally carry more expected fantasy value than a carry. This is a useful volume
summary, not a complete role model: goal-line carries, routes, two-minute work, and
end-zone targets are not equivalent to ordinary opportunities.

### Target share, air-yards share, and WOPR

V1 retains nflverse's weekly target share, air-yards share, and WOPR fields. The v1
aggregator weights weekly shares by the player's own targets. That is better than a
flat weekly mean for an injury-shortened week, but it is not a true season share and
the previous documentation overstated it as team-pass-attempt weighting.

V2 therefore adds `season_target_share = player targets / team targets` and
`season_carry_share = player carries / team carries`. A later data upgrade should
derive air-yards share from player and team air-yard totals and calculate WOPR from
those season-level shares rather than averaging weekly WOPR.

### Touchdowns over expectation

V1 uses:

`TD over expectation = rushing + receiving TDs - (carries + targets) × position TD rate`

At +4 it raises `TD-luck`; at −2.5 with at least 150 weighted opportunities it raises
`BUY`. The signal correctly questions noisy touchdown totals, but "luck" is stronger
language than the metric proves. Players can repeatedly earn valuable goal-line and
end-zone work. V2 treats it as partial regression rather than removing the entire
difference.

The next refinement should estimate rushing and receiving touchdowns separately from
goal-line carries, red-zone/end-zone targets, team scoring expectation, and the
quarterback's rushing competition.

### Weekly floor and volatility

Floor is the 25th percentile of weekly league points. Volatility is their sample
standard deviation. Both include the league's long-touchdown bonuses.

These are useful lineup descriptors, especially when choosing floor as a favorite or
ceiling as an underdog. They are noisy in short samples and do not measure a player's
future range by themselves. V2 leaves them as historical evidence rather than using
them directly in projected PPG.

### Age and flags

V1 flags an RB at age 27.5 or later, measured on September 1 of the draft season. It
also exposes positive/negative touchdown-regression flags and the number of games in
a small sample. Age is a useful warning but the binary RB cliff is too coarse to be a
projection. V2 applies a bounded gradual curve by position while retaining the flag.

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
   efficiency, team volume, scoring environment, and teammate competition.

The bottom-up branch receives 65% of projected PPG by default and the historical
branch receives 35%. V2 then projects floor and ceiling, calculates VOR for all three
outcomes, and sorts on a balanced `v2_score`.

### 1. Historical PPG branch

Each player-season receives:

`season weight = 0.55 ^ years old × games`

The weighted PPG becomes `historical_ppg_prior`. It is shrunk toward the rosterable
positional PPG prior based on effective games:

`reliability = effective games / (effective games + 8)`

`individual prior = reliability × historical PPG + (1 - reliability) × position PPG`

For RB/WR/TE, half of latest-season rushing/receiving TD over-expectation is regressed,
bounded at ±2.5 PPG. Historical big-play bonus PPG is partially regressed toward the
position. The age curve and relative team-context factor are then applied. This yields
`prior_branch_ppg`.

### 2. Team and teammate context

For every team-season, v2 calculates:

- passing fantasy context per game;
- official pass attempts, dropbacks, and carry volume per game;
- offensive TDs per game;
- passing TDs per target and rushing TDs per carry;
- the largest teammate target share;
- the largest competing RB's weighted backfield share.

The projected team is compared with the contexts already embedded in the player's
weighted history. Ratios are guarded at 0.75–1.25 and the historical branch's combined
team effect is capped at ±18%, preventing one unstable team statistic from dominating.

### 3. Projected opportunity

V2 calculates target share against official team pass attempts and carry share against
team carries. Each is recency-weighted and shrunk according to its target/carry
sample.

Receiving role blends correlated metrics deliberately:

`receiving share = 75% shrunk target share + 25% WOPR-equivalent share`

WOPR therefore contributes air-yard role information without counting target share
twice at full weight. Teammate availability then adjusts the share. RB carry share is
adjusted by the strongest competing back.

The nflverse participation feed identifies which offensive players were present on a
dropback, but it does not prove that every eligible RB/TE released into a route. V2
therefore names this input `route_opportunities`, not charted routes. It separately
shrinks route-opportunity participation and targets per route opportunity (TPRR).

When participation is available, projected targets use:

`route targets = projected team dropbacks × route-opportunity participation × TPRR`

The route estimate receives 55% weight. The remaining branch blends target share
against official attempts, historical target rate, and raw weighted opportunity.
Without participation data, v2 falls back cleanly to that remaining branch.

Projected carries use 75% team carry-share volume and 25% normalized historical
carries/game. All historical anchors respond to projected team pass/rush volume, and
the explicit opportunity multiplier applies to volume rather than efficiency.

### 4. Projected efficiency and stat line

The component model separately shrinks:

- catch rate;
- receiving yards per target;
- rushing yards per carry;
- receiving TDs per target;
- rushing TDs per carry;
- big-play bonus points per opportunity.

Receiving touchdowns are decomposed into projected end-zone targets, end-zone
conversion, and non-end-zone TD rate. Rushing touchdowns similarly separate carries
inside the five, goal-line conversion, and non-goal-line TD rate. Red-zone targets and
carries are exported alongside the narrower high-value opportunities. General TD
rates retain 30% weight so sparse charting samples cannot dominate.

Air-yards share is recency-weighted and shrunk. Its ratio to target share measures
role depth relative to the position, producing a bounded `air_yard_factor` for yards
per target and long-TD bonus expectation. Receiving and rushing TD rates are also
scaled by the projected team's passing- and rushing-TD environments.

For QBs, official attempts are projected from team passing volume. Yards per attempt,
passing-TD rate, and interception rate are shrunk separately, then converted into the
passing stat line. QB rushing yards, rushing TDs, and bonus points remain separately
projected.

The exported component stat line contains targets, carries, receptions, passing/
rushing/receiving yards, split TDs, interceptions, and bonus points per game. A small
shrunk residual preserves rare scoring events not explicitly decomposed, such as
two-point conversions, fumbles, and special-teams production. Scoring those components
under the league rules produces `component_proj_ppg`.

### 5. Combining the branches

`projected PPG = 35% prior_branch_ppg + 65% component_proj_ppg`

The weight is configurable. `proj_ppg` and its VOR (`proj_vor`, `adj_proj_vor` with
overrides) remain as the hand-built projection, but since the 2026-08-29 review they
are **inputs and diagnostics, not the sort**: `component_proj_ppg` and
`historical_ppg_prior` are features of the walk-forward fitted rankers, and the board
ranks on the fitted outputs (see §10).

### 6. Floor, volatility, and ceiling

Floor and volatility describe the distribution of the same weekly points, so they do
not get added to expected PPG. Their multi-year ratios to PPG are shrunk toward
positional profiles and applied to the forward mean:

- `projected_floor`: projected 25th-percentile score;
- `projected_volatility`: projected weekly standard deviation;
- `projected_ceiling = projected PPG + 0.674 × projected volatility`.

These are descriptive fields and roster-risk inputs (the League team-strength rating
rescales `projected_volatility` to the fitted mean). The former floor/ceiling VOR blend
(`v2_score`) was removed after the backtest showed it lost to both the naive baseline
and the fitted rankers on cross-position draft value.

### 7. Removed after backtesting

The 2026-08-29 review (`docs/v2_metrics_review.md`) ablated every prior-branch
adjustment against 22 seasons of folds. The following were removed because they
carried no repeatable signal, and their config keys are gone:

- `td_regression_adjustment` (partial regression of last season's TD over-expectation);
- `bonus_regression_adjustment` (big-play bonus regression toward the position);
- `team_context_factor` (the ±18% capped scalar comparing destination and source
  team environment — team context still enters the component branch through projected
  team volume, TD rates, and teammate availability);
- `v2_score`, `floor_vor`, `ceiling_vor`, `availability_adjusted_vor`,
  `expected_season_points`, and `availability_factor` (composites superseded by the
  fitted outputs).

What survived the ablation and remains in the prior branch: the all-player shrinkage
prior, the age curve, and the depth-chart role factor.

### 8. Current depth charts and explicit future assumptions

The latest published nflverse depth-chart snapshot automatically supplies current NFL
team, position, and depth rank. Current team supersedes the player's last historical
team unless a manual projection override is present. Depth rank applies a bounded
role adjustment: rank one is neutral, while ranks two and below progressively reduce
opportunity. The snapshot date is exported because a dated depth chart is evidence,
not timeless truth.

[`projections.yaml`](../src/patron/config/projections.yaml) supplies facts historical
data cannot know. A player entry can set `projected_team`, `opportunity_multiplier`,
and a visible reason. A team entry can adjust QB context, pass/rush volume, scoring,
target availability, or backfield availability for every player on that team.
Unmatched player entries fail the build so a misspelled trade or role change cannot
quietly disappear.

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

### ESPN's deliberately limited role

ESPN is not a statistical projection source. It remains useful for private-league
state that nflverse cannot know: which fantasy manager owns a player, who is a free
agent, fantasy lineup placement, and ESPN's live injury display. Those fields power
roster and waiver views, but ESPN projected points, lineup slots, and ownership rates
do not feed V2's football projection. Historical statistics, usage, teams, depth
charts, and injury modeling are nflverse-first.

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

Overall `team_rank` sorts expected weekly points. Risk is intentionally separate rather
than silently penalizing or rewarding volatility. The displayed `team_score` maps the
expected-points distribution to a league-relative 0–10 index centered at 5.0, with 1.5
rating points representing one league standard deviation.

An ESPN full-season projection is used only for a rostered rookie or returnee with no
nflverse tape; its volatility is estimated from the position's V2 ratio and its
availability is set to the configured 94% prior. The UI reports the number of these
fallbacks. Kicker and defense points are excluded because the V2 player projection
does not produce their weekly mean and variance. ESPN division IDs and names group the
cards but do not change the overall rank.

## V2 additions and updates

| Area | V1 | Combined v2 |
|---|---|---|
| Production | Latest-season PPG | 35% normalized PPG branch + 65% projected stat line |
| Sample size | Flag only | Shrinkage plus confidence |
| VOR | Historical mean VOR | Expected, floor, ceiling VOR and balanced score |
| Touchdowns | TDOE flag | Historical regression plus component-level split TD rates |
| Big-play bonuses | Historical points | Opportunity-regressed rate adjusted by air-yard role |
| Target share | Target-weighted weekly average | True, shrunk share directly projects targets |
| Carry share | Missing | True, shrunk share directly projects carries |
| WOPR | Display only | 25% of receiving-role estimate, avoiding full double count |
| Air-yards share | Display only | Adjusts yards per target and bonus expectation |
| Weighted opportunity | Display only | Per-game receiving-role anchor |
| Floor/volatility | Historical descriptors | Projected range, floor VOR, and ceiling VOR |
| Team/QB | Embedded in results | Explicit volume, scoring, QB, TD-rate, and competition context |
| Age | Binary RB flag | Gradual position curve plus original flag |
| Future changes | Manual VOR delta | Projected team, team-wide context, and role multiplier |
| Dashboard/API | One board | Selectable v1/v2 with component metrics |
| League comparison | Opponent positional weaknesses | Overall roster rank/strength, rookie fallback visibility, and ESPN divisions |
| Routes/TPRR | Missing | nflverse route-opportunity participation and regressed TPRR project targets |
| High-value usage | TD rate only | Red-zone/end-zone targets and carries inside the five project split TDs |
| Passing volume | Target proxy | Official attempts/dropbacks project QB and receiver opportunity |
| Depth charts | Manual only | Latest nflverse team/rank automatically changes context and role |
| Availability | Games flag | Expected games, season points, and availability-scaled VOR |

## Implementation coverage and remaining roadmap

The percentages below are approximate implementation coverage, not claims about
forecast accuracy. "Direct" means the input is represented in the projected stat
line. "Proxy" means v2 responds to a related signal but cannot distinguish the more
specific football usage we ultimately want.

### Bottom-up projection core

| Area | Approx. coverage | What v2 does now | Important gap |
|---|---:|---|---|
| Targets | 90% | Projects route opportunities × TPRR, blended with share of official attempts, historical target rate, WOPR, and weighted opportunity | Participation cannot distinguish pass protection from a released route |
| Carries | 90% | Projects carry share and team carry volume, with current depth rank, backfield competition, red-zone work, and goal-line work | Situation-neutral run rate and two-minute usage remain implicit |
| Routes | 65% | Derives eligible-player dropback participation and TPRR from nflverse play participation | This is a route-opportunity proxy, not a charted route for every player |
| Red-zone opportunities | 80% | Separates red-zone/end-zone targets, carries inside the five, and their conversion rates | Quarterback sneak competition and expected red-zone trips are not yet separately forecast |
| Efficiency | 75% | Separately projects catch rate, receiving yards per target, rushing yards per carry, TD rates, and bonus rate | No yards per route, separation by route/coverage type, or explicit efficiency aging beyond the overall age factor |
| League scoring | 90% | Scores the projected components under the league rules and retains a shrunk residual for rare scoring | Fumbles, two-point conversions, and special-teams scoring are not projected as separate events |
| Weekly range | 75% | Projects floor, volatility, and ceiling from shrunk historical distribution shapes around the new mean | No matchup, injury, or role-state mixture distributions |

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
| Injury-based games played | 85% history | 75% | nflverse participation and injury reports now project expected games, season points, and availability-adjusted VOR. Injury type/recovery-stage models and a current preseason availability feed remain missing. |

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
- `board_v2.json` and `board_v2.md`: forward-projection board.

The API accepts `version=v1|v2` on `/api/board`, `/api/players/{id}`, and
`/api/positions`. `/api/status` reports availability and counts for both versions.
The dashboard's v1/v2 tabs call those endpoints; switching tabs changes both the sort
key and the visible metric columns.
