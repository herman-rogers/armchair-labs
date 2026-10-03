# Roster opportunity review — September 22, 2026

> Dated research record. Findings and roster advice apply to the stated data and cutoff; follow-up ideas are not an active implementation commitment.

Advice only; no claims, lineup changes, or trade offers submitted.

User correction: both kickers and both defenses must be retained. Treat these as roster constraints in future recommendations. The revised claims below supersede the original three-add plan.

## Evidence and scope

- Live ESPN snapshot: September 22, 2026, 23:45:13 UTC (7:45 p.m. EDT), Just Say No, team 3, Sweaty Plays. Ten teams, full PPR, three starting WRs plus a flex, $145 FAAB remaining from $150.
- Accepted repaired research: `historical_v2_20260922_r5`, accepted for research, not promoted. See [acceptance](../../data/research/historical_v2_20260922_r5/acceptance.json) and [repair findings](../history/historical_data_repairs_2026-09-22.md).
- Repaired 2026 season forecasts use an August 28 cutoff. They are preseason estimates, not rest-of-season forecasts incorporating Weeks 1–2. Positional ranks below are calculated from fitted season totals; ESPN draft ordering is not an executable trade price.
- Current opportunity screen joins the live ESPN ownership snapshot to canonical player IDs, repaired predictions, and Weeks 1–2 statistics/snaps in the rookie-watch input snapshot. Scratch evidence: `/tmp/fantasy_roster_review_2026-09-22.json` and `/tmp/fantasy_roster_review_screen.parquet`; these temporary files are not permanent research artifacts.
- [Rookie research](../history/rookie_watch_2026-09-22.md) provides exploratory next-four-week historical analog estimates. Boston: 42.5 points, analog spread 18.4–70.3; Tate: 31.4, spread 4.6–61.5. These are not calibrated confidence intervals or injury/matchup-adjusted forecasts.

The repaired backtests do not establish a market-beating ranking edge. Standalone models lose to consensus in the modern returner/top-60 comparison. The 80% consensus/20% next-generation blend has a positive point estimate but an interval crossing zero. Recommendations therefore emphasize actual availability, recent workload, roster fit, and acquisition cost.

## Roster assessment

WR depth is the clearest repair: Rice, Adams, Moore, Sutton, Hunter, Doubs. Moore is questionable, Sutton has started slowly, and Hunter has one target through two games and a 10% offensive snap share in Week 2. Hunter is expendable in this ten-team format.

CMC and McBride remain anchors. Keep Pollard while Dowdle is questionable; Etienne also has competition for receiving work. Kaelon Black has 21 carries through two games and provides CMC insurance. Stafford provides a starter while Dart's knee timeline develops. Both kickers (Reichard and McLaughlin) and both defenses (Bears and Falcons) must be retained under the user-provided roster requirement. They are not available drop slots. Retain Doubs initially: his Week 2 line was three catches for 96 yards, and his role has room with A.J. Brown out.

## Suggested claims

All players below were unrostered in this league at the snapshot time. These are alternative claims for the same Hunter slot, not four additions. Retain the other bench players for now; further adds require a separate justified drop or a trade that creates space.

| Order | Add | Drop | Suggested FAAB | Basis |
| --- | --- | --- | ---: | --- |
| 1 | Denzel Boston | Travis Hunter | $9; discretionary ceiling $12 | 11 targets through two games; seven targets and 93% offensive snaps in Week 2. Strongest combination of immediate role and rookie upside. |
| 2, if Boston missed | Carnell Tate | Travis Hunter | $5 | 11 targets through two games and 75% Week 2 snaps. Longer-term rookie bet; current production trails Boston. |
| 3, if above missed | Adonai Mitchell | Travis Hunter | $5 | 15 targets through two games, including 12 in Week 2; 82% Week 2 snaps. One spike week is not proof of a settled target hierarchy. |
| 4, if above missed | Dontayvion Wicks | Travis Hunter | $3 | 10 targets through two games; six in Week 2. |

The original $17 Boston recommendation was an aggressive discretionary allocation: 11.3% of the initial $150 budget, or 11.7% of the $145 remaining. It was not estimated from competing bids or a validated points-to-FAAB model, and its exact dollar precision was unjustified.

The available snapshot contains five winning waiver bids: Caleb Douglas $3, Tyjae Spears $5, Devaughn Vele $3, MarShawn Lloyd $11, and Kaelon Black $5. It also records Tate being dropped on September 16. This limited sample supports trying a smaller bid, but includes different player situations and no losing bids; it cannot establish Boston's clearing price. Revised amounts remain judgment calls. A successful $9 Boston claim leaves $136.

Boston moves ahead of Tate for immediate help based on current opportunity, not the next-generation preseason ranking. Verified against the repaired forecast and the UI ranking implementation: Tate is **24th across all positions by raw predicted season points**, which is **5th among WRs**, with **218.39 projected season points**. These are the same model output under different position filters. Boston is 171st overall/WR69 with 79.54 predicted points; Hunter is 174th overall/WR70 with 78.75. The overall rank is not a value-over-replacement draft ranking or a current rest-of-season ranking. The earlier explanation failed to make this distinction explicit.

The next-generation preseason model strongly favors Tate over Boston. Preferring Boston for immediate help is a judgment that weights current usage and the separate in-season rookie feed more heavily; it is not the model's recommendation. Tate's rookie analog distribution is wide, and the repaired integrity checks do not validate the model's extreme preseason WR5 prediction. A manager prioritizing the preseason model's full-season upside can reasonably choose Tate for Hunter instead. The rookie rows have no populated forecast-cutoff date; do not imply their individual rows have a verified August 28 timestamp simply because returner rows do.

Current external usage checks: [Sharp Week 3 waiver analysis](https://www.sharpfootballanalysis.com/fantasy/nfl-waiver-wire-pickups/), [Doubs Week 2 recap](https://www.fantasypros.com/nfl/news/609287/romeo-doubs-rebounds-after-tough-week-1.php).

## Trade opportunities

**Jameson Williams, team 6:** Most practical first inquiry. His manager has Lamb, Olave, Coker and other WR depth, with Kyle Pitts and Brenton Strange at TE. Offer Juwan Johnson for Jameson; Johnson plus Doubs is a conditional exploratory package only if the full resulting roster, including any required TE replacement, can be filled acceptably. With only one clear waiver slot, do not assume both Boston and Tate will be added. Acceptance is uncertain. Jameson has 13 targets through two games and played 97% of Week 2 snaps. The repaired preseason rankings place him WR18 in next-generation and WR16 in core season totals. His low recent output does not imply loss of role, but weekly volatility remains.

**DK Metcalf, HOUSE OF M:** Strong usage-based buy-low inquiry. He has 19 targets through two games, 97% Week 2 snaps, and only 14.7 league points. Repaired preseason ranks are WR20/WR18. Explore Sutton for Metcalf straight up; the manager may decline and lacks an obvious RB shortage. Do not add a meaningful RB starter simply to force a deal. His passing environment may continue to limit conversion. [Week 2 report](https://www.fantasypros.com/nfl/news/609307/dk-metcalf-leads-team-targets-week-2-loss.php).

Josh Downs is a secondary usage-based target (13 targets, nine in Week 2), but his repaired rank does not show a major discount. Free WR additions reduce the need to pay for him.

## Other positions and misleading discounts

**Dalton Schultz is available:** 22 targets through two games; repaired preseason TE6 next-generation/TE8 core. Consider $2–4 to replace Johnson if Johnson is traded, or as an optional backup upgrade. Do not carry three TEs or divert substantial WR spending with McBride already rostered. His 14-target Week 2 benefited from Nico Collins' absence, so that workload should not be extrapolated unchanged. [Schultz recap](https://www.fantasypros.com/nfl/news/609414/dalton-schultz-explodes-12-catches-sunday.php).

**Emanuel Wilson and Woody Marks:** Cheap RB contingencies, behind the WR claims. Wilson's 21 Week 2 carries depend partly on Jadarian Price's availability; Seattle described Price's shoulder injury as not serious/long term. Marks had six Week 2 targets but shares the backfield. [Seattle injury update](https://www.seahawks.com/news/seahawks-planning-on-sam-darnold-returning-to-practice-this-week).

Avoid acting solely on favorable old model ranks: Tyrone Tracy has only two carries through two games; Jakobi Meyers has three targets; Jauan Jennings has one target and an injury designation. Those workload concerns outweigh apparent draft-rank discounts.

Retain both kickers and both defenses. The prior recommendation to reduce these positions to one each is withdrawn.
