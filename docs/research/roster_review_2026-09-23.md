# Roster and waiver review — September 23, 2026

> Dated research record. Findings and roster advice apply to the stated data and cutoff; follow-up ideas are not an active implementation commitment.

**Recommended first moves: Adonai Mitchell for Travis Hunter; Baker Mayfield for
Jaxson Dart. C.J. Stroud is the alternative if the next four weeks matter more
than longer-term QB depth.** These are recommendations only; no claims, drops,
lineup changes, or messages were submitted.

## Data and interpretation

- League: Sweaty Plays, Just Say No, team 3; 10 teams, full PPR with the league's
  own scoring bonuses, one QB, two RBs, three WRs, one TE and one RB/WR/TE flex.
- Ownership: September 23, **3:56 p.m. EDT** snapshot. $145 budget remains. All
  recommended additions were unrostered in that snapshot. The snapshot does not
  distinguish an immediate free-agent add from a pending waiver claim.
- Corrected gold: `canonical_nextgen_20260923_r1`; observations through Week 2.
- Completed analysis: `nextgen_system_20260923_r1`, generated at 4:03 p.m. EDT.
  Its manifest, corrected gold dependency and experiment files passed hash checks.
- Matching outlook: `nextgen_products_20260923_r1_outlook`, Weeks 3–6. Its saved
  current predictions are exactly equal to the previously published canonical
  outlook; source/output hashes were checked. No unfinished model results are used.
- Injury context: `injuries_20260923_r1`, described in
  [the capture notes](../history/current_injury_capture_2026-09-23.md). External reporting
  supplies availability facts only. No external rankings or projections enter
  this review.
- Retain **both kickers and both defenses**, following the existing roster rule.

The new analysis keeps the historical baseline as its default. Ridge and boosted
models are **shadow research**, and the old fitted NextGen season model is archived
pending revalidation. This review explicitly compares the new research outputs;
it does not promote them, blend their scores, or claim a validated waiver edge.

The recommendations are a judgment using current opportunity, roster fit,
availability, and model comparisons. They are not a saved optimized waiver policy.

In the tables, **usage / outlook** are two separate own-data estimates of total
league points across **Weeks 3–6**, including scheduled byes. They are neither a
range nor a confidence interval. **Ridge / boost** are separate **preseason
full-season** predictions with their original cutoff. They are not remaining
season points and are not directly subtracted from four-week forecasts. Their
disagreement is uncertainty, not independent confirmation.

The short-horizon WR outlook does not reliably improve on the simpler usage
model; its historical error is about 13 points per four-week total. Differences
of one or two points among these WRs are not decisive. QB error is about 18 points
over the same horizon. No calibrated player-level confidence claims are made.

## Hunter replacement order

| Priority | Available WR | Weeks 3–6 usage / outlook | New preseason ridge / boost | Weeks 1–2 targets | Week 2 snaps | Reason |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | **Adonai Mitchell** | **39.6 / 40.4** | 83.6 / 74.0 | **15** | **82%** | Best immediate own-data opportunity screen. Twelve targets in Week 2; a substantially more useful role than Hunter's. |
| 2 | **Carnell Tate** | **35.1 / 32.7** | **117.0 / 151.8** | 11 | 75% | Preferred developmental alternative. Strong draft-capital signal and actual offensive involvement, with meaningful model disagreement. |
| 3 | **Tre Tucker** | **37.3 / 37.3** | **128.8 / 133.3** | 11 | 59% | Strongest balance of established production and both new preseason models among this shortlist; lower snap exposure than Tate/Mitchell. |
| 4 | Dontayvion Wicks | 38.4 / 39.4 | 78.0 / 79.8 | 10 | 62% | Strong short-horizon alternative; less preseason support. |
| 5 | Rashod Bateman | 38.2 / 38.9 | 57.4 / 50.7 | 10 | 93% | Nine targets in Week 2 and substantial field time; current opportunity is more encouraging than historical model value. |
| Compare | Travis Hunter | **8.8 / 7.0** | 89.2 / 88.5 | **1** | **10%** | Insufficient offensive workload for this roster. |

Mitchell is my recommendation for a roster needing usable WR depth now. Choose
Tate instead if the intended use is a patient upside stash. Tucker is close
enough to Tate that preferring his more established production is reasonable;
neither ordering is a demonstrated model certainty.

**The old “Tate WR5” claim is not supported by the latest models.** In the new
full-season research outputs he is WR49 in ridge and WR26 in boost, versus the
archived model's WR5. The current rookie reference baseline is a broad historical
rookie mean, not a player-specific assessment of Tate; it is not a sensible reason
to call him a 25-point talent.

Denzel Boston is already owned by **Pause**, as is Emanuel Wilson. Yesterday's
Boston-first waiver plan is no longer executable. Troy Franklin leads the
available WRs in the new boosted preseason model, but only has three targets
through two games and about 20% Week 2 snaps; I would not elevate that old forecast
over a current role. IR/out players and players absent from ESPN's captured pool
are not treated as executable healthy acquisitions.

As alternative claims, every WR above can use **the same Hunter drop**; the list
does not assume five open spots. A second receiver would need a separate drop.
I would keep Doubs for now: both new preseason models value him above Mitchell,
and he has a meaningful offensive role. Replace Hunter before using Doubs' spot.

## Dart replacement order

The September 23 NFL Network report says Dart is expected to undergo
[season-ending knee surgery](https://www.nfl.com/news/giants-qb-jaxson-dart-season-ending-knee-surgery).
Treat him as unavailable for the remainder of this redraft decision. His saved
four-week and preseason numbers predate that report and are not usable estimates
of his remaining production. Historical Week 1–2 points remain intact.

| Priority | Available QB | Weeks 3–6 usage / outlook | New preseason ridge / boost | Bye | Reason |
| --- | --- | ---: | ---: | ---: | --- |
| 1 | **Baker Mayfield** | **55.5 / 60.5** | **222.4 / 249.1** | **10** | Best longer-term depth choice. Both new preseason models favor him over the other principal healthy candidates here. Complements Stafford's bye. |
| 2 | **C.J. Stroud** | **63.9 / 65.5** | 193.0 / 167.3 | 8 | Best immediate four-week choice. Ninety-three attempts through two games, including 55 in Week 2. |
| 3 | Bryce Young | 61.2 / 64.1 | 172.1 / 138.6 | 5 | Strong current screen despite only three scheduled games in Weeks 3–6; less full-season support. |
| 4 | Jordan Love | 59.6 / 60.0 | 197.2 / 214.5 | **11** | Good preseason alternative, but shares Stafford's bye and does not fill that coverage need. |
| 5 | Cam Ward | 57.5 / 63.2 | 143.6 / 151.5 | 9 | Viable fallback with some rushing involvement; weaker preseason support than Mayfield/Love/Stroud. |
| Compare | Matthew Stafford | 55.3 / 60.8 | 252.3 / 187.2 | **11** | Keep. The models disagree about Stafford versus Mayfield, so adding Mayfield is not an automatic starting-QB change. |

Because Stafford is already rostered, I prefer Mayfield as a second QB. If the
goal is maximizing projected Weeks 3–6 lineup points, put **Stroud first** instead.
Do not add both and carry three QBs in this one-QB league.

Geno Smith and Kirk Cousins remain usable fallbacks; their four-week outlooks are
61.0 and 59.8 respectively, but both new preseason models favor Mayfield, Love and
Stroud. Deshaun Watson's usage estimate is high, yet the expanded four-week model
falls to 52.3 and preseason estimates are much lower. That disagreement does not
justify making him the preferred replacement.

Do not mistake Drew Lock's recent starts for a secure season-long job. Seattle
was [planning Darnold's return to practice](https://www.seahawks.com/news/seahawks-planning-on-sam-darnold-returning-to-practice-this-week).
Darnold's injury-shortened sample also makes his low saved four-week estimate
unrepresentative of a fully healthy starter. Jameis Winston and other newly
installed starters need a role-aware update: low forecasts trained on their
previous backup usage are not trustworthy estimates for the new job. None is
necessary while the primary shortlist is free.

## Other roster improvements

**Consider Dalton Schultz for Juwan Johnson if immediate TE/flex help is the
priority.** Schultz has 22 targets through two games and estimates of 44.3 / 43.5
for Weeks 3–6, versus Johnson's 30.0 / 29.7. This is not a recommendation to carry
three TEs. Keep McBride as the primary TE. Schultz's 14-target Week 2 came with
Nico Collins absent; Collins is now questionable in the league snapshot. That
concentration can unwind. The new preseason models actually favor Johnson over
Schultz, so this is a short-term workload decision, not a clear permanent upgrade.

**Woody Marks for Kaelon Black is an optional depth upgrade.** Marks has seven
targets through two games, including six in Week 2, and estimates of 34.3 / 34.5
over Weeks 3–6 versus Black's 26.4 / 27.1. Both new preseason models also prefer
Marks (115.6 / 119.6 versus 95.2 / 76.1). With Pollard and Dowdle both questionable,
Marks offers another currently active option. Black's specific value is CMC
insurance; keep Black if protecting that injury contingency matters more than
current standalone workload. This review does not assign a made-up probability
to McCaffrey missing games.

**Keep the core:** McCaffrey, Etienne, McBride, Rice, Adams, Moore and Sutton.
Rice and Sutton's two-game usage is a monitoring issue, not a sufficient reason
to discard their stronger historical outlooks. Moore's injury-limited Week 2
should not be interpreted as a pure demotion. Keep Pollard and Dowdle for now,
but do not treat the saved point forecasts as proof they will play this week.
Retain Doubs, both kickers, and both defenses. Injury statuses must be rechecked
before setting the actual lineup.

The main roster weakness is **usable receiver depth**, with a secondary need for
healthy QB coverage and RB depth while two backs carry injury tags. There are
enough unrostered options to address these without first paying a trade premium.

## Estimated lineup effect and limits

Comparing legal offensive lineups under the same four-week totals, treating Dart
as unavailable but otherwise assuming the saved forecasts apply:

| One move at a time | Additional optimized lineup points, usage / outlook |
| --- | ---: |
| Mitchell for Hunter | +9.4 / +9.3 |
| Tate for Hunter | +4.8 / +1.6 |
| Tucker for Hunter | +7.0 / +6.2 |
| Stroud for Dart | +8.5 / +4.7 |
| Mayfield for Dart | +0.2 / +0.0 |
| Schultz for Johnson | +12.0 / +10.9 |
| Marks for Black | +2.1 / +1.9 |

Hunter is on the bench: replacing him does **not** automatically add the entire
Mitchell-minus-Hunter forecast difference to the starting lineup. Mayfield's
near-zero immediate estimate is also expected when Stafford already starts;
the purpose is depth, an alternate starter, and Week 11 coverage. These are
mechanical research scenarios, not guaranteed gains or individual weekly start
recommendations. They do not adjust for new injury durations or specific matchup
difficulty, cannot optimize different weekly lineups without weekly forecasts,
and are not additive when multiple additions compete for the same starting spot.

## Reproducible evidence

Saved in `data/research/roster_review_20260923_r1/`:

- `manifest.json`: pinned gold, NextGen, matching outlook and injury references.
- `league_snapshot.json`: the exact ownership snapshot used in the review.
- `players.parquet` / `players.csv`: all captured players, availability, current
  workloads, separate forecast families and ownership.
- `lineup_scenarios.json`: matched legal-lineup comparisons.

All captured skill-position players mapped to canonical player IDs. Conflicting
identity records outside the league pool were not allowed to overwrite mappings;
no name-only guesses were needed for these recommendations.
