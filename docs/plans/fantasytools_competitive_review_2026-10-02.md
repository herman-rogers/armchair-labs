# FantasyTools.ai comparison and adoption priorities

> Planning document. Proposed work is not a statement of shipped behavior; use the [plans index](README.md) for scope and status.

Reviewed October 2, 2026. This review compares FantasyTools.ai's public product with the Patron Saints repository and recommends features to adopt. It covers the current frontend, league integration, scoring, forecast code, documentation, and selected published release. It is a product and code review, not a head-to-head prediction benchmark or a test of every competitor interaction. The competitor's “My team this week” page could not be fetched, so its integration capabilities were not verified.

**Recommendation: borrow their weekly decision workflows.** That is where their public product is ahead of this repo. This review found no evidence that their predictions are better than ours. Our best first feature is a league-specific waiver-risers board, paired with a weekly briefing.

FantasyTools.ai organizes information around what a manager needs to do this week. Our app organizes much of it around players, rankings, and research. We have useful foundations for turning that analysis into more direct league decisions.

## What FantasyTools.ai is

FantasyTools.ai is a free fantasy-football toolkit operated by Nolix LLC. It advertises 67 tools covering rankings, waivers, trades, streaming, scoring, and game-day context. Its named AI assistant, “Chalk & Sharp,” is still marked coming soon. The existing product centers on statistical boards and calculators using sources including nflverse, Fantasy Football Calculator, and National Weather Service. [Source: FantasyTools.ai](https://fantasytools.ai/)

## Feature comparison

The final column contains recommendations, rather than claims about features already implemented here.

| Area | What they have | What this repo has | What to adopt |
| --- | --- | --- | --- |
| Weekly briefing | A [weekly cheat sheet](https://fantasytools.ai/weekly-fantasy-football-cheat-sheet/) connecting their boards | [League overview](../../web/src/components/LeagueOverview.tsx), matchups, FAAB, transactions, and alerts | A personalized “Your week” screen: lineup problems, available risers, upcoming byes, and changes since refresh |
| Waiver discovery | [Risers](https://fantasytools.ai/waiver-wire-risers/) explain changes in snaps, targets, and carries; availability is approximated using draft ADP | Actual ESPN free-agent ownership plus published remaining-season ranks in [LeagueWorkspace](../../web/src/components/LeagueWorkspace.tsx) | Highest-value feature: opportunity changes filtered to players actually available in our league |
| Start/sit | [Player comparisons](https://fantasytools.ai/who-should-i-start/) with reasons, ranges, and historical success rates for different projection gaps | Captured ESPN weekly projections; our own published forecasts cover remaining season and next four weeks | A focused comparison screen, initially using clearly labeled ESPN estimates; our own recommendations require a weekly forecast |
| FAAB | A [bid calculator](https://fantasytools.ai/faab-bid-calculator/) using remaining budget, player tier, need, and week | Remaining budgets and reported transaction bids | Editable bid scenarios incorporating our budget and roster need; later investigate league bidding tendencies |
| Trade analysis | A [trade calculator](https://fantasytools.ai/fantasy-football-trade-analyzer/) based on remaining points above replacement | Ownership, roster slots, forecasts, and archived lineup/value machinery | Evaluate the actual before/after roster, including the required drop and available replacement |
| Streaming and game day | [DST streaming](https://fantasytools.ai/dst-streaming-rankings/), kicker boards, [weather](https://fantasytools.ai/nfl-weather-report/), implied totals, and line movements | K/DST scoring and ESPN observations; no comparable current weather/odds workflow found in the reviewed source | Add league-filtered streaming candidates and dated game context |
| Accuracy reporting | A [public projection tracker](https://fantasytools.ai/fantasy-projection-accuracy/) freezing forecasts and comparing results with season averages | Extensive historical evaluation, immutable releases, and [experimental prospective grading](../../src/patron/metrics/prospective.py) | A simple current-season scorecard attached to the forecasts people actually use |

## Existing advantages to preserve

Our strongest asset is league specificity. We know actual ownership, roster slots, remaining budgets, and captured lineups. Their waiver board explicitly substitutes ADP for ownership. That gives us a better starting point for answering “Who can I pick up?” [Their waiver methodology](https://fantasytools.ai/waiver-wire-risers/), [our league workspace](../../web/src/components/LeagueWorkspace.tsx).

We also handle this league's big-play touchdown bonuses in the [scoring engine](../../src/patron/scoring/bonuses.py). Their start/sit tool explicitly excludes long-TD and yardage bonuses. That supports keeping our scoring foundation, although correctly scoring historical bonuses does not automatically solve forecasting them. [Their start/sit methodology](https://fantasytools.ai/who-should-i-start/)

Our profiles, career comparisons, college/rookie analysis, and forecast provenance are substantial foundations. However, existing research code is not the same as a usable current feature. Our [README](../../README.md) explicitly archives the older waiver/trade/matchup experiments, and the [published ranking contract](../history/nextgen_rankings_2026-09-23.md) says these ranks are not weekly lineup recommendations or FAAB values.

## Recommended implementation order

1. **Freshness and “Your week.”** At review time, the local [current catalog](../../data/current.json) selected a release whose [ranking report](../../data/research/nextgen_qb_variations_20260924_r2/ranking_report.json) used production through Week 2. Their [weekly rankings](https://fantasytools.ai/weekly-fantasy-football-rankings/) advertised Week 3 inputs. Check the existing refresh/publish workflow first, then combine our alerts, matchup, budget, and refresh changes into one screen. Newer research folders do not establish that newer forecasts are published. This finding describes the local selected release on the review date, not every deployment.

2. **Available players whose role changed.** Use corrected snap, target, and carry measurements, show the latest week against a recent baseline, and intersect with ESPN ownership. Each row should explain the change and show coverage. This can be useful immediately as descriptive evidence without pretending that a usage jump guarantees future points. The [gold data definitions](../../src/patron/data/gold.py) already include relevant opportunity measurements; current coverage and trustworthy denominators still need checking during implementation.

3. **Start/sit comparison plus prospective logging.** Show two players, their weekly estimate source, availability, opponent, and supporting evidence. Freeze estimates before kickoff and grade them afterward. Do not divide our four-week forecast by four and call it a matchup projection. Existing prospective grading code is experimental infrastructure to review and adapt, not proof that a current weekly tracker already exists.

4. **Acquisition scenarios.** Connect a candidate to an explicit drop, roster need, and editable FAAB range. Add trade evaluation on the same foundation. The old [lineup code](../../src/patron/espn/lineup.py) is a starting point to review, but it explicitly lacks weekly matchup/bye handling and treats unranked players as zero. Those limitations must be addressed before using it for current recommendations.

Streaming and game context remain useful follow-on work from the comparison. They require additional current data and, for our own projections, validation beyond the existing K/DST scoring functions. Handcuff scenarios can follow once role assumptions and scenario presentation are explicit.

## Competitor assumptions to avoid copying blindly

- **FAAB percentages are heuristics.** Their calculator discloses a formula based on budget, tier, need, and season urgency. Those percentages are not demonstrated auction-winning prices. Borrow the transparent controls and ranges. [FAAB calculator](https://fantasytools.ai/faab-bid-calculator/)

- **Handcuff upside is conditional.** Their handcuff board models an upper-end transfer of the starter's workload and lacks a depth-chart feed. Borrow the scenario interface and make workload assumptions editable. Do not present the upper-end scenario as the expected result of an absence. [Handcuff methodology](https://fantasytools.ai/fantasy-football-handcuff-rankings/)

- **Trade drops have real opportunity costs.** Their trade analyzer assumes a required drop has replacement-level value. We can use the actual player being dropped and the actual available replacement. [Trade methodology](https://fantasytools.ai/fantasy-football-trade-analyzer/)

- **Accuracy claims need matching scope.** Their accuracy tracker had no completed prospective week when checked. It also discloses that two QB settings were chosen after examining 2024. Their reported backtest edge is modest; it does not establish superiority over our different forecast horizons. [Accuracy methodology](https://fantasytools.ai/fantasy-projection-accuracy/)

The recommended first development choice is the league-specific waiver-risers board, paired with a weekly briefing. It uses data we already understand, fixes a clear usability gap, and benefits directly from our ESPN integration.
