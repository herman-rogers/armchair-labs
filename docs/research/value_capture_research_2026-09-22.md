# Research: where our fantasy analysis captures value — and where it does not

> Dated research record. Findings and roster advice apply to the stated data and cutoff; follow-up ideas are not an active implementation commitment.

Date: 2026-09-22. Scope: retained historical forecasts through 2025, saved research
reports, current data definitions, and outside research. Draft value is the primary
focus; in-season applications are identified separately. No model was refitted or
promoted, no board was rebuilt, and neither V1 nor the frozen 2026 forecast was changed.

## Bottom line

We have useful forecasting infrastructure, but the retained evidence does **not**
demonstrate a reliable ability to find bargains better than the market. When our full
rankings disagree with consensus, the players we leave behind have generally been
more valuable than the players we elevate. This also happens against observed ADP,
not just expert rankings.

The most useful next research direction is **forecasting changes in opportunity at
an actionable price**, especially RB contingent roles, rookie/young-player breakouts,
and returning players with misleading prior-season averages. This is a hypothesis
motivated by our errors, not an already-proven edge.

Keep the existing scoring and historical-production foundation. Add a separately
evaluated decision layer that asks: “What can I acquire, what changes its role, and
how much usable lineup value does that add?” More accurate season rankings and
better fantasy decisions are related, but they are not the same objective.

## 1. What we tested

The accompanying [audit script](../../research/value_capture_audit.py) reuses the saved
walk-forward prediction table; it does not train new candidates. It compares:

- V2 fitted season points, the recomputed Adaptive-family selector, next-generation
  season points, and the market-conditioned availability challenger.
- Expert consensus rankings (ECR) and preferred observed preseason ADP.
- All covered players and returning players separately.
- Top-24, top-60, and top-120 selections.
- The full model ranking and fixed 90/10 and 80/20 market/model ordinal-rank blends.
- Expanded 2011–2025 and modern 2019–2025 windows.

This produces 288 diagnostic comparisons, plus 12 ECR-versus-ADP comparisons. These
are not 300 independent experiments or independent confirmations of a finding.

Each model-policy/market comparison uses exactly the same players. A “call” is a
player selected in the model's top K but not the market's; the “market alternative”
is one of the equal number of players selected only by the market. A hit finishes
in the actual top K of that same comparison pool.

The actual target is the app's saved availability-adjusted value:

`max(actual active-game PPG − actual positional replacement PPG, 0) × actual games / 17`

Net value is the summed value of model-only selections minus market-only selections,
averaged across seasons. It is **not** a team's weekly scoring gain, draft profit,
or championship probability. Top-K selection does not enforce a legal roster or
simulate which players remain available at each pick.

## 2. How well are we capturing value?

### Our disagreements usually hurt

Modern window, 2019–2025, overall top 60, ECR comparison:

| Saved candidate | Model-only calls | Model-only hits | Market-alternative hits | Net value / season | Season-bootstrap 95% interval |
|---|---:|---:|---:|---:|---:|
| V2 fitted season points | 89 | 20 / 89, 22.5% | 41 / 89, 46.1% | -13.80 | [-22.75, -5.16] |
| Adaptive-family selector | 88 | 25 / 88, 28.4% | 41 / 88, 46.6% | -11.37 | [-19.83, -3.29] |
| Next-generation season points | 90 | 25 / 90, 27.8% | 38 / 90, 42.2% | -9.17 | [-16.16, -2.52] |
| Market + availability, returners only | 70 | 25 / 70, 35.7% | 29 / 70, 41.4% | -2.99 | [-10.90, 5.23] |

The first three candidates share the same modern comparison pool. The availability
candidate covers a different, returning-player pool: its smaller deficit is promising,
but not a clean all-player victory over the other systems. “Adaptive-family” here is
a historical recomputation, **not an evaluation of the frozen 2026 forecast**.

ECR missed 150 actual top-60 player-seasons in the first three candidates' common
pool. V2 recovered 20 (13.3%); Adaptive and next-generation each recovered 25 (16.7%).
The modern common pool excludes two true top-60 player-seasons from the larger saved
forecast universe, one in 2020 and one in 2023. Coverage is recorded rather than
silently treating those omitted players as successes.

Against observed ADP, modern V2 calls hit 19/88 while the alternatives hit 36/88;
next-generation hits 23/87 versus 34/87. Their net values remain negative, -12.81 and
-9.47 respectively. The problem is not just which consensus source we benchmark.

The existing August 31 metric report reaches the same broad conclusion: none of its
38 candidates has positive net swap value in either the modern window or the
expanded market history. Its older feature-discovery report tests 17 families,
including nonlinear, temporal, latent, symbolic, weekly-context, and stacked models;
none passes its ECR promotion gate. Those reports have different vintages/populations
and should not be pooled as if they were one clean trial.

### Small adjustments look safer, but have not established an edge

An exploratory 80% ECR / 20% model rank blend among returners yields these expanded
2011–2025 results:

| Model contribution | Model-only hits / calls | Market-alternative hits | Net value / season | 95% interval |
|---|---:|---:|---:|---:|
| V2 | 24 / 50 | 13 | +1.27 | [-0.53, 2.83] |
| Next-generation | 23 / 51 | 11 | +1.41 | [-0.39, 3.02] |

Both have positive net value in 12 of 15 seasons. However, their intervals include
zero; switching the baseline to observed ADP changes net value to -0.24 and -0.26.
Modern-window gains also vary with the population and blend weight. These are useful
pilot candidates, not a basis for silently changing the production ranking.

ECR alone versus observed ADP is also inconclusive: expanded all-player top-60 net
value is +0.05, interval [-2.74, 2.97]; modern is +0.27, interval [-4.75, 5.15]. We
have not demonstrated that merely buying consensus/ADP discrepancies solves value
detection.

## 3. What works — and what that evidence actually supports

- **League-specific scoring is a real capability.** The app scores full PPR and
  exclusive +2/+3 bonuses for 40–49/50+ yard touchdowns. That supports a useful
  scoring adjustment to generic rankings, although the size of any resulting market
  advantage still needs testing. Do not simply extrapolate last year's long TDs.
- **Historical production remains a useful starting point.** It captures established
  players, is interpretable, and is an appropriate baseline. Outside TE research also
  finds prior production informative, alongside team receiving share and route volume;
  that is predictive evidence, not proof of value after price.
  [Fantasy Points' summary of its TE research](https://newsletter.fantasypoints.com/p/killer-stats-to-know-roundup).
- **Some good disagreements are real.** Examples include 2024 George Kittle and
  2023 Alvin Kamara/Evan Engram. V2's modern WR calls hit 7/21 versus RB's 2/21.
  This describes where its hits occur; it does not establish WR outperformance over
  the corresponding market choices.
- **Explicit rookie modeling can solve otherwise inaccessible cases.** The
  next-generation candidate ranks 2021 Ja'Marr Chase 37th in the common overall
  pool, versus V2's 305th and the market's 64th; he finishes eighth by the target.
  That is a useful case study, not validation of the whole rookie model.
- **Availability and market information are useful directions.** Their combined
  challenger has the smallest modern deficit in the retained report. A clean ablation
  is still needed to separate market information, population filtering, and true
  availability-model contribution.
- **The evaluation infrastructure is worth retaining.** Walk-forward predictions,
  recorded misses, source metadata, and a frozen prospective forecast make it possible
  to reject attractive but unsuccessful ideas.

## 4. What we miss

### A. Changes in role, not just changes in efficiency

V2's 89 modern model-only calls contain 72 fourth-year-or-later players, and 88 have
at least ten games in the prior season. It mostly finds established production that
the market discounted.

Of the market's missed top-60 player-seasons:

| Group | Market misses | Recovered by V2 | Recovered by next-generation |
|---|---:|---:|---:|
| Rookies | 27 | 0 | 2 |
| Returning players with fewer than ten prior-season games | 12 | 0 | 0 |
| Second-/third-year players | 42 | 5 | 5 |
| RBs, all career stages | 35 | 2 | 3 |

Rows overlap and must not be summed. This is not evidence that all young players or
all injured players are bargains. It identifies where our existing approach fails to
recover bargains that actually occurred.

Lamar Jackson in 2019, Deebo Samuel in 2021, and Jalen Hurts in 2022 are notable misses.
For Lamar, a season average containing backup usage is an unsuitable estimate of a
future starter's workload. We need a probability of entering a new role and a forecast
conditional on that role, not just a longer average of old roles.

For RBs, distinguish early-down carries, receiving work, two-minute work, goal-line
opportunities, and the probability of inheriting them. A cheap backup may have little
median season value but meaningful value if promoted. Conversely, high yards per
carry without a path to more valuable work need not be a buying signal.

### B. Historical context is not yet trustworthy enough

Two retained examples warrant a source/identity audit:

- Lamar's 2019 row has `team=BAL` and `projected_team=BAL`, but
  `cutoff_preseason_team=IND` and `team_changed=1` at a July 17 cutoff.
- Deebo's 2021 row has `cutoff_preseason_status=off`, no cutoff team, and
  `cutoff_preseason_rostered=0` at an August 27 cutoff. The next-generation games
  forecast is 4.86; the actual result is 16 games.

These observations do not by themselves identify the exact upstream bug or prove
that correcting a field would fix the forecast. They demonstrate why identity,
transaction chronology, and missing-data behavior need checking before adding more
weight to status/context features. The working-tree repair effort is separate from
this audit of retained artifacts.

Availability should also separate known as-of-cutoff absences, probability of playing,
workload when active, and genuinely unpredictable future injuries. An “active” roster
label is not a complete estimate of expected playing time.

### C. “Routes” and timely routes are different data products

Our enrichment explicitly measures presence on a dropback, not charted routes run.
An RB or TE staying in protection can count in the denominator. Treating this as true
targets per route can confuse blocking usage with receiving opportunity.

The public participation dictionary's `route` describes the primary receiver, not
every eligible player's route. More importantly, the documented 2023+ participation
feed arrives after the postseason and does not update in-season. Prior-season
preseason research can use it; a live Tuesday waiver model cannot assume that same
feed exists for Sunday's game. [Participation dictionary](https://nflreadr.nflverse.com/articles/dictionary_participation.html),
[official update schedule](https://nflreadr.nflverse.com/articles/nflverse_data_schedule.html).

Start an in-season pilot with genuinely available snaps, targets, carries, play
situations, and roster/practice timestamps. Add true routes and first-read targets
only with verified access, delivery timing, definitions, and historical coverage.
Commercial tools document teammate on/off analysis and injury-status splits; those
are useful categories to evaluate in a small data trial, not grounds for buying a
large subscription before incremental value is measured.
[Fantasy Points Data feature documentation](https://fantasypointsdata.com/whats-new).

### D. Consensus is not our acquisition price

The repo already has an observed-ADP archive, but inspection found no application
caller of `load_backfill_adp`; existing disagreement results use ECR. The research
script explicitly joins the archive and exposes that comparison.

Preferred ADP here comes from actual MFL PPR drafts after August 15. It is not our
ESPN room, and its final preseason window can contain news later than the model/ECR
cutoff. Some older ECR captures also use standard scoring. Therefore this is a useful
sensitivity test, not an exact-date, exact-league trading simulation.

The eventual decision benchmark must include platform rank, actual pick cost,
available players, roster needs, and the chance a player survives until our next pick.
A player can be accurately ranked and still be a poor selection at the current price.

### E. The outcome does not yet measure how we win

Season points and top-60 hits reward outcomes differently from a managed ten-team
lineup. The bench has limited space; replacement players can cover absences; weekly
start decisions are imperfect; breakout timing matters. A rank-59 finish counts as
a hit just as a rank-three finish does, despite very different impact.

Keep forecast accuracy as a diagnostic, but evaluate usable starter weeks, incremental
lineup points, waiver/drop cost, and eventually team wins under the league's rules.
Use decisions made from information available before the games, not hindsight-optimal
lineups. The open-source ffsimulator documentation is a useful example of separate
weekly, lineup, and season outputs, not a validation of our current simulator.
[ffsimulator basic simulations](https://ffsimulator.ffverse.com/articles/basic).

## 5. The research program I would prioritize

### Experiment 1: Find role changes before the price adjusts

**Hypothesis:** correctly identifying a transition into valuable opportunity adds
information beyond market price and trailing production.

Build a small, interpretable role-state model: reserve, rotational, primary, limited
by injury. For each player, forecast role probabilities and production conditional
on each role. Keep rookie and returning-player models explicit; use hierarchical
shrinkage so tiny samples do not masquerade as certainty.

Allocate opportunities within a team's passing/rushing totals: every teammate cannot
gain target share simultaneously. Record why a role changes and when that evidence
became available. Do not retrospectively discard poor games as “injury-limited”
without a consistently applied, dated rule.

Position-specific inputs to test:

| Position | Opportunity mechanism | Important control |
|---|---|---|
| RB | Receiving/two-minute work, inside-five carries, teammate availability, next-in-line probability | Separate contingent upside from an existing starting role |
| WR | Route share, targets earned per true route, competition on/off, QB and formation changes | Do not extrapolate teammate-absence splits as if the absence is permanent |
| TE | Routes versus blocking, share of team passing work, competition ahead of the TE | On-field snaps are not receiving routes |
| QB | Starting-role probability, designed rushing/scrambles, team dropbacks | Backup appearances must not define full-start PPG |

For a preseason experiment, compare role-conditioned forecasts against ECR, ADP,
prior PPG, and the same model without role information. For an in-season pilot,
predict next-four-week opportunity and realized lineup value at the waiver deadline,
using only information published by then. Compare against last-three-game usage and
available weekly market projections. Archive those market snapshots prospectively
if historical vintages cannot be established.

Measure whether improvement concentrates in the missing groups above, but score the
full eligible pool too. Otherwise the model could “improve” by simply declining to
predict difficult players.

### Experiment 2: Price-aware, league-specific decisions

**Hypothesis:** our own scoring and draft-room behavior provide more actionable
opportunities than broad disagreement with national consensus.

Use a strong consensus prior, a separately tested custom-scoring adjustment, and
only validated model adjustments. Map that value to the actual room's acquisition
prices. An 80/20 rank blend is a research comparator, not a chosen production formula;
equal rank gaps do not imply equal point gaps or equal uncertainty.

Run an ablation separating:

1. Generic consensus.
2. Consensus adjusted only for our PPR/long-TD scoring and replacement rules.
3. The same forecast with role/availability adjustments.
4. A draft policy incorporating roster needs and availability at the next pick.

Long-TD bonuses should use shrunk opportunity/rate estimates and show uncertainty,
not simply reward last year's rare long scores. Do not reintroduce generic TD
regression flags as proven signals: the existing project already tested and removed
versions that did not repeat.

Use legal draft/lineup replay with the same opponent policy and random seeds for
each comparison. Report incremental usable points and the cost of reaching early,
not only successful sleeper names. In-season, replace ADP with actual availability,
FAAB/drop cost, and improvement over our current lineup.

### Experiment 3: Availability, uncertainty, and contingent upside

**Hypothesis:** differentiating absence mechanisms and role-dependent outcomes is
more useful than a single expected-games number or uncalibrated “confidence” score.

Audit dated transactions first. Then estimate separately: probability active, workload
conditional on active, and alternative role states. Known absence, returning from
injury, and random future injury should not share one undifferentiated penalty.

Produce calibrated probabilities of becoming a useful starter, downside outcomes,
and weekly production intervals. Check interval coverage and probability calibration
by position, price range, and career stage. A broad interval is not itself upside;
beneficial upside must be achievable at the roster-slot and acquisition cost.

Late-round/waiver contingent-value policies should pay for holding the bench slot and
for false-positive stashes. Do not assume a best-ball roster where every spike week
is automatically captured. Only test portfolio correlation or playoff-tail objectives
after the weekly decision simulation is credible.

## 6. Guardrails before claiming we found an edge

- **Use a decision-specific baseline.** Last-season PPG measures modeling progress;
  matched-date ECR/ADP and an executable acquisition policy measure market value.
- **Fix population and feature provenance first.** The prior stat-system review
  identified rookie-training coupling and unequal headline evaluation pools. This
  audit corrects comparison pools but does not refit those historical models.
- **Version every feature as of its availability date.** Annual historical tables
  can be valid for next year's draft but invalid for a within-year waiver backtest.
- **Audit derived-model vintages, too.** ffopportunity documents models trained on
  2006–2020 data. If those outputs inform earlier simulated forecasts, assess the
  embedded training look-ahead; use cutoff-specific models or a suitably later test
  window. This is an audit requirement, not a confirmed leak in every app candidate.
  [ffopportunity model documentation](https://ffopportunity.ffverse.com/).
- **Pre-register a small set of hypotheses and primary outcomes.** The 2011–2025
  history has already informed many choices. Nested training does not erase research
  selection over the same outcomes. Keep an explicit experiment ledger and count
  failed trials; do not promote the best-looking slice out of hundreds.
- **Report economic size and uncertainty together.** Use paired season results,
  sensitivity to eras and price sources, and concentration in a few players/years.
  Seven modern seasons are a small sample. These bootstrap intervals do not adjust
  for all model-search or era dependence.
- **Require a complete deployment-policy test.** Include rookies, missing features,
  fallback forecasts, legal lineups, and acquisition constraints. Abstention may be
  useful, but its fallback must be scored.
- **Preserve a prospective test.** Keep V1 as the draft reference and leave the
  existing frozen 2026 forecast intact. New weekly candidates can enter a separately
  timestamped shadow trial after this report; the next untouched full preseason for
  a newly designed system is 2027. Do not relabel already-inspected history a holdout.

## 7. Reproduction and limitations

Run from the repository root:

```sh
.venv/bin/python research/value_capture_audit.py
.venv/bin/ruff check research/value_capture_audit.py
```

Generated artifact: `data/outputs/value_research_2026-09-22.json` (ignored generated
output). It includes fold results, bootstrap intervals, coverage, cohorts, examples,
and input hashes. Stable player-ID tie-breaking can produce a one-hit difference
from the older report at a tied cutoff without changing the overall conclusion.

Input prediction SHA-256:
`2bb6089480805e18abf91c3bd5577efb0e827fd97b7d5b1d4f3af30fd2798811`.

Input ADP SHA-256:
`6347d552e0bf7ab9903f11001b5400155a199635bbc2bb59253d9aead7139403`.

The script checks these inputs did not change during execution. It excludes pending
2026 rows, compares paired player pools, and uses 10,000 season-bootstrap resamples
with seed 20260922. Model replacement ranks mirror the retained report: QB12, RB25,
WR35, TE12, using players with at least eight historical games to set the threshold.

Verification passed: Ruff checks; synthetic gain/loss, no-op, tie, and bootstrap
checks; all 300 comparisons' accounting and date invariants; and reproduction of
the four candidates' modern saved-report hit counts and net values within rounding.
Application tests were not rerun for this research-only addition.

The saved forecast table defines the observable population. Players absent from it
cannot be measured here. Pairwise common-pool results are not a replacement for a
complete draft-pool deployment evaluation. Player ranks in case studies are ranks
within that common overall pool, not literal published ADP or positional finish.

Working-tree fixes underway are not evaluated by this research; these results concern
the retained August 31 forecasts. No positive blend, subgroup, outside correlation,
or compelling player example in this report constitutes a validated new advantage.

Local evidence: `data/outputs/metric_report.json`,
`data/outputs/feature_discovery_report.md`,
`data/static/market_backfill_manifest.md`, `src/engine/config/scoring.yaml`,
`src/engine/metrics/enrichment.py`, and the
[earlier stat-system review](stat_system_review_2026-09-22.md).
