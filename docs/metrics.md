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
- a confidence field measuring how much historical sample supports the estimate.

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
- team target and carry volume per game;
- offensive TDs per game;
- passing TDs per target and rushing TDs per carry;
- the largest teammate target share;
- the largest competing RB's weighted backfield share.

The projected team is compared with the contexts already embedded in the player's
weighted history. Ratios are guarded at 0.75–1.25 and the historical branch's combined
team effect is capped at ±18%, preventing one unstable team statistic from dominating.

### 3. Projected opportunity

V2 calculates true target and carry share from player opportunities divided by team
opportunities. Each is recency-weighted and shrunk according to its target/carry
sample.

Receiving role blends correlated metrics deliberately:

`receiving share = 75% shrunk target share + 25% WOPR-equivalent share`

WOPR therefore contributes air-yard role information without counting target share
twice at full weight. Teammate availability then adjusts the share. RB carry share is
adjusted by the strongest competing back.

Projected targets use three anchors:

`targets/game = 65% team-share estimate + 20% historical target rate + 15% raw weighted-opportunity estimate`

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

Air-yards share is recency-weighted and shrunk. Its ratio to target share measures
role depth relative to the position, producing a bounded `air_yard_factor` for yards
per target and long-TD bonus expectation. Receiving and rushing TD rates are also
scaled by the projected team's passing- and rushing-TD environments.

For QBs, passing yards, passing TDs, interceptions, rushing yards, rushing TDs, and
bonus points are projected directly from shrunk per-game history and team context.

The exported component stat line contains targets, carries, receptions, passing/
rushing/receiving yards, split TDs, interceptions, and bonus points per game. A small
shrunk residual preserves rare scoring events not explicitly decomposed, such as
two-point conversions, fumbles, and special-teams production. Scoring those components
under the league rules produces `component_proj_ppg`.

### 5. Combining the branches

`projected PPG = 35% prior_branch_ppg + 65% component_proj_ppg`

The weight is configurable. Keeping both branches prevents an incomplete stat feed or
unstable efficiency estimate from replacing the strong signal in multi-year league
PPG, while the majority component weight allows an excellent role with poor prior
results to move meaningfully.

### 6. Floor, volatility, and ceiling

Floor and volatility describe the distribution of the same weekly points, so they do
not get added to expected PPG. Their multi-year ratios to PPG are shrunk toward
positional profiles and applied to the forward mean:

- `projected_floor`: projected 25th-percentile score;
- `projected_volatility`: projected weekly standard deviation;
- `projected_ceiling = projected PPG + 0.674 × projected volatility`, an approximate
  75th-percentile outcome.

Replacement floor and ceiling are recomputed by position to produce `floor_vor` and
`ceiling_vor` alongside expected `proj_vor`.

### 7. Balanced v2 score

The configurable default is:

`v2_score = 75% projected VOR + 15% floor VOR + 10% ceiling VOR + manual override`

Expected value remains dominant, with a slight floor preference appropriate to a
weekly head-to-head league. The weights are normalized at calculation time.
`adj_proj_vor` remains available as pure expected VOR plus the override; `v2_score` is
the actual v2 sort key.

### 8. Explicit future assumptions

[`projections.yaml`](../src/patron/config/projections.yaml) supplies facts historical
data cannot know. A player entry can set `projected_team`, `opportunity_multiplier`,
and a visible reason. A team entry can adjust QB context, pass/rush volume, scoring,
target availability, or backfield availability for every player on that team.
Unmatched player entries fail the build so a misspelled trade or role change cannot
quietly disappear.

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

## Important missing inputs and planned refinements

The next improvements with the highest expected value are:

1. Snap share, route participation, targets per route, designed QB rushes, two-minute
   RB routes, goal-line carries, and end-zone targets.
2. Official team pass attempts, situation-neutral pace, red-zone trips, and implied
   scoring rather than target/carry proxies.
3. Current roster/depth-chart ingestion so teammate competition is projected from the
   future roster rather than latest-season usage plus manual multipliers.
4. Injury-specific games-played distributions and cumulative RB touches.
5. Actual player/team air-yard totals and yards per route. V2 currently uses the
   available air-yards share as a bounded role-depth factor.
6. Backtesting each coefficient against held-out seasons, including calibration by
   position and projection-confidence bucket.
7. ADP and auction-price deltas. Market price should not determine player ability, but
   it is essential for deciding when a model edge is actionable.
8. Draft tiers and uncertainty bands. Differences of 0.1 projected VOR should not be
   presented as materially precise rank gaps.

## Version selection and artifacts

`patron board` writes:

- `board.json`: compatibility alias for v1;
- `board_v1.json` and `board_v1.md`: historical board;
- `board_v2.json` and `board_v2.md`: forward-projection board.

The API accepts `version=v1|v2` on `/api/board`, `/api/players/{id}`, and
`/api/positions`. `/api/status` reports availability and counts for both versions.
The dashboard's v1/v2 tabs call those endpoints; switching tabs changes both the sort
key and the visible metric columns.
