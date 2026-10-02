# Data integrity audit and RB role research

Date: 2026-09-22. Audited checkout: `c80b346` plus the existing uncommitted historical
fold repairs. This is a new audit of the repaired code, not a repetition of the
earlier review. Production code and artifacts were not changed by this work.

## Verdict

**Base scoring and the main player identifiers pass the checks below. Historical
team/status, contract timing, and the rich weekly route features are not yet sound
enough to certify for role-transition research.** Some prior errors are fixed;
other errors remain. A full corrected model-performance claim must wait for these
repairs and a rebuilt historical dataset.

We can still make progress using a restricted input set. An exploratory RB pilot
using scored weekly results, raw offensive snaps, and preseason ADP finds useful
information in current usage beyond recent fantasy points. It does not yet show
that complex role-transition features beat simple usage, or that small-sample
returners are a profitable subgroup.

## 1. Updating our understanding of the codebase

The merged stat-system repairs implement the V1 archive, returner/rookie training
boundaries, production/research artifact separation, fixed outcome universes,
canonical forecast fields, and improved waiver replacement/lineup semantics. The
working-tree historical repairs improve full-name transaction matching, route-source
coverage, configurable history windows, and adaptive-selector fold comparisons.

At the start, **431 offline tests passed; 35 network tests were deselected**. A final
rerun after concurrent API/UI edits had **430 passed, one failed, 35 deselected**:
`test_board_version_selects_the_matching_artifact` expects an exact old status-response
shape, while the changed API adds build/forecast metadata. Those unrelated edits were
not changed by this research. Four additional research tests pass, including a test
that changing future scores and snaps cannot change an earlier decision's features.
Research-file Ruff checks pass. Passing tests validate tested behavior; they do not
certify every real-world transaction interpretation.

There is an important publication distinction:

- The main checkout's `metric_report.json` is still dated August 31, and retained
  predictions still have SHA-256 beginning `2bb608948080`. Its boards lack the new
  publication manifests, and `data/outputs/production_model.json` is absent here.
- The `fantasy-stat-review` worktree has separately regenerated production artifacts
  and board manifests, consistent with its implementation report. A code merge does
  not copy ignored generated output into this checkout.
- The archived V1 passes its manifest verification; the main V1 compatibility files
  match its pinned `ad726895d694…` digest. The API now prefers the verified archive.

The earlier market-performance numbers remain evidence about the old saved forecasts,
not a grade of the merged repairs. Nothing here republishes the other worktree's
artifacts or changes the frozen 2026 forecasts.

## 2. What passes

| Check | Result | What it establishes |
|---|---|---|
| Component scoring vs source PPR, 2001–2025 | 133,802 skill-player weeks; zero discrepancies above tolerance | Scoring arithmetic agrees with the cached source |
| Long-TD bonus reconciliation | 15,047 credited points; zero missing player-ID credits | Bonus credits balance at the play-to-player boundary |
| Bonus join to weekly results | Zero unjoined bonus rows/points | Credits also survive the weekly join |
| Raw skill-player/week keys | Zero duplicate player/season/week rows | These raw weekly joins do not multiply observations |
| Retained forecast keys | Zero duplicate player/forecast-season rows | No duplicate forecast observations |
| Player identity/PFR crosswalk | Zero duplicated non-null GSIS or PFR mapping keys | Mapping is unique in the examined tables |
| Snap identity coverage | 101 unmatched of 87,479 skill/FB/HB snap rows | About 99.88% map; exceptions remain visible |
| Completed-target arithmetic | PPG × games matches season points | The target fields are internally consistent |
| Replayed transaction date limits | Zero last-event dates beyond cutoff | Event-date filtering works, independently of event interpretation |

This is reconciliation against cached sources, not independent film adjudication of
NFL stat corrections. Source snapshots may have been revised after historical
decision dates. That limitation matters more to an exact Tuesday-waiver replay than
to descriptive prior-season scoring.

## 3. Remaining problems, in repair order

### P1: Transaction identity improved, but event meaning still fails

Replaying the repaired matcher on completed 2004–2025 returner folds changes **420
team values, 213 status values, and 618 transaction counts** relative to the retained
inputs. These are differences, not 1,251 independently confirmed corrections.
The earlier repair note's 467/234/690 counts also included pending 2026 rows; this
audit excludes those from the replay evaluation.

Confirmed old examples now behave correctly:

- Lamar Jackson 2019: IND becomes BAL; the false transaction match disappears.
- Lamar Jackson 2020: NE becomes BAL.
- Deebo Samuel 2021: off/no team becomes active/SF.

However, current-code replay still leaves **Deebo's 2025 cutoff team as SF**, despite
the source transaction describing a trade to Washington. The normalized input uses
the publishing team's code as `to_team`; the state transition then treats it as the
destination. The clause splitter also treats suffixes such as `Sr.` as sentence
boundaries, potentially separating a player from the rest of the action.
[49ers' own transaction archive](https://www.49ers.com/team/transactions/2025).

Reversing input order while preserving all dates changes five completed player-season
states: Aaron Rodgers 2023, Malik Willis 2024, and Ameer Abdullah, Skyy Moore, and
Brian Robinson 2025. Brian Robinson ends up WAS versus SF depending on which same-day
team announcement is processed last. Malik Willis changes both team and status.
This is input-order dependence for tied dates, not a claim that identical runs are
random. Mixed-action announcements also allow another player's reserve/activation
language to affect the selected player's status.

Repair contract:

1. Preserve `source_team` separately from `from_team` and `to_team`.
2. Normalize one player/action per event; protect initials, suffixes, and clauses.
3. Reconcile both sides of trades and duplicate announcements into one event.
4. Resolve same-day conflicts using event semantics or an explicit unknown state,
   never alphabetical team order.
5. Store the raw source, date, identity evidence, action, and resolution confidence.
6. Distinguish an observed active status from “no matching event, carried forward
   from last year's team.” Missing evidence is not evidence that a player is active.

Until then, quarantine transaction-derived status, team changes, and downstream
vacated-opportunity features from the new research model. Do not overwrite the frozen
2026 snapshot; corrections belong in new historical/research artifacts.

### P1: Contract features contain confirmed post-cutoff information

`build_contract_features` filters on signing **year**, not signing date. Two examples
survive a rebuild with current code:

| Player/fold | Forecast cutoff | Deal incorporated | Actual agreement date |
|---|---|---|---|
| Ezekiel Elliott, 2019 | July 17 | Six-year extension | September 4 |
| Dak Prescott, 2024 | August 30 | Four-year extension | September 8 |

The dates are confirmed by the club's announcements:
[Elliott extension](https://www.dallascowboys.com/news/done-deal-zeke-signs-6-year-contract-extension),
[Prescott extension](https://www.dallascowboys.com/news/dak-prescott-cowboys-agree-to-terms-on-massive-contract-extension).

There are **3,634 completed returning-player seasons with a same-year contract row**
whose exact signing date is absent from this join. They require date resolution;
they are not all proven leaks. Contract terms can also begin after signing, so
`signing year + term` is not automatically the effective expiration date.

Require exact agreement/publication/effective dates or exclude same-year deals from
strict point-in-time tests. A prior-year-only sensitivity run is conservative, not
a complete repair. Retrospectively updated guarantees and termination fields also
need their own provenance. The source dictionary itself documents `year_signed`
only as a year. [nflreadr contract dictionary](https://nflreadr.nflverse.com/articles/dictionary_contracts.html).

This finding applies to candidates consuming these contract features; it is not a
claim that the simpler production PPG model uses every affected field.

### P2: Route-participation numerator and denominator do not match

The repaired v2 rich weekly panel correctly marks 2001–2015 as lacking participation
coverage. That fix is present. Nevertheless, **2,441 route-participation values
exceed 1.0**, with a maximum of 1.3333.

A concrete reconciliation: San Francisco's 2016 Week 1 has 40 play-by-play dropbacks,
including five scrambles. The panel's team denominator is 35 attempts-plus-sacks.
A player present on all 40 dropbacks therefore gets 40/35 = 1.1429. The numerator
and denominator must use the same play population; clipping to 1.0 hides the error.

Even after that fix, presence on a dropback remains a route-opportunity proxy, not
proof that a TE/RB ran a route. Public 2023+ participation is documented as arriving
after the postseason, so it is unsuitable as an assumed live weekly input.
[nflverse update schedule](https://nflreadr.nflverse.com/articles/nflverse_data_schedule.html).

The panel also has 18,665 nonfinite air-yard-share values. Its descriptor logic masks
nonfinite observations, so this is not evidence that all fitted models ingest NaNs.
Signed air-yard shares can legitimately be negative or above one; do not apply a
blanket probability-style range constraint to them. Separate feed availability,
player observation, true zero, and invalid ratio in the data contract.

### P2: Rookie population and outcome positions are inconsistent

Rebuilding season points from weekly scoring finds **47 discrepancies**, all rookie
rows labeled RB whose raw outcome position is FB. The rookie builder maps FB to RB,
but the outcome pool excludes FB, assigning zero to production that actually exists.
Six occur in 2019–2022; examples include Andrew Beck, Alec Ingold, and Connor Heyward.

This does not explain the top-player misses by itself, but it contaminates rookie
training labels. Apply the same position policy to candidate construction, scoring,
and outcome joins. Either include FB production under RB consistently or exclude
those candidates consistently. Preserve V1's archived definition separately.

### P2: Price and publication provenance limit what can be claimed

Preferred historical ADP keys have no duplicates, but MFL's final preseason window
does not match earlier ECR snapshot cutoffs and is not an ESPN-room price. Our local
ESPN snapshots are dated 2026-08-31, 09-01, 09-03, and 09-22; they do not provide
historical 2019–2025 room prices or waiver availability. A September 22 capture must
not be silently substituted for a preseason snapshot.

Before publication, rebuild historical folds from the repaired sources, refit in an
isolated output location, and verify artifact manifests against the exact code/input
versions. A reanalysis/refit of old columns alone does not repair the underlying
transaction labels or contract dates.

Additional timing limitations remain un-certified: legacy Week-1 depth charts use
an August 31 proxy rather than a real publication timestamp, and derived xFP values
need the underlying learner's training vintage checked as well as the play dates.
ffopportunity documents training on 2006–2020 data; applying that derived model to
earlier simulated decisions requires special treatment. Neither source enters this
pilot. [ffopportunity documentation](https://ffopportunity.ffverse.com/).

## 4. First restricted-input research experiment

The pilot deliberately does **not** use transaction, contract, route, injury, roster,
or xFP features. It uses the audited bonus-inclusive weekly score table, directly
mapped offensive snaps, carries/targets, and preseason ADP.

This is an **in-season research watchlist**, not a preseason ranking replacement or
an executable waiver simulation. It is a first way to test the role hypothesis while
the richer preseason context is quarantined.

Design:

- Walk forward through 2019–2025; training uses only earlier seasons, starting with
  2013 snaps. Means, scales, and ridge coefficients use training rows only.
- Decide after Weeks 3–13. Forecast total points in the next four calendar weeks,
  counting absences and byes as zero.
- Candidates: RBs with offensive snaps in the just-completed week, preseason ADP
  greater than 100 and at most 300, and trailing three-calendar-week PPG below 12.
  Preseason ADP is available by these decision dates; it is not their current price.
- Select three candidates per decision from the same pool for every policy.
- Compare trailing points, preseason ADP, simple recent snaps, simple weighted
  opportunity, price/production regression, usage regression, and a role-change
  extension. Ridge penalty 10 is fixed; no parameter grid is searched.
- Separately examine returners with 1–9 prior-season box-score appearances.

The usage regression adds recent snaps, carries, and targets. The role extension adds
snap-change, observation support, and prior-season sample size. Stronger simple-usage
controls were added after seeing the first points-baseline result, to test whether
the apparently promising gain actually requires the more complicated model.

### Results: usage is promising; incremental role-change edge is not established

All late-price candidates: 77 decision windows, 231 selections per policy. The typical
pool has 28 RBs. Players can be selected repeatedly; these are not 231 independent
players or acquisitions. The role policy selects 102 distinct player-seasons.

| Selection policy | Realized next-four-week points per selection |
|---|---:|
| Lowest preseason ADP in eligible pool | 25.66 |
| Highest recent fantasy PPG | 29.38 |
| Price + production regression | 30.73 |
| Highest recent snap share | 32.30 |
| Price + production + usage regression | 32.64 |
| Role-change extension | 32.88 |

The role extension gains **3.50 points across four weeks**, not per week, versus
recent PPG; the season-bootstrap 95% interval is [1.55, 5.73], with positive results
in seven of seven seasons. However, its incremental gain is only **0.23** versus
the usage regression, interval [-1.53, 2.00], and **0.57** versus simply sorting on
recent snaps, interval [-0.97, 2.31]. This does not establish an advantage from the
transition-specific features or over contemporary weekly market projections.

Small prior samples do not improve: role extension 23.67 versus recent-PPG baseline
24.17, a -0.50 difference with interval [-1.27, 0.30]. There are 69 eligible windows;
the median candidate pool is only four, and 23 windows have exactly three candidates,
leaving no selection choice. This narrow slice needs more informative eligibility
and better transition evidence, not a promotional claim from this result.

### Acquisition-price research: progress and remaining evidence gap

Using a known preseason cost band makes this a price-aware screening experiment, but
we still lack the ownership/FAAB/available-player history needed to value an actual
add. Do not label the observed difference “waiver profit” or a market-beating edge.

The custom-scoring adjustment also needs realistic scale. Among 2019–2025 players
with at least eight box-score appearances, 90th-percentile observed bonus PPG is
0.93 for QB, 0.30 for WR, and 0.21 for RB. TE's 90th percentile is zero. This is
descriptive, not a next-season forecast; it argues for measured scoring adjustments
rather than assuming long-TD bonuses move every sleeper multiple rounds.

Next acquisition test: timestamp actual ESPN ranks/ADP, available players, roster,
pick position or waiver budget, and the alternatives at each decision. Benchmark
against a simple consensus/usage policy with identical information and constraints.
Evaluate cost of acting now versus waiting, not just prediction accuracy.

### Usable-weekly-value research: progress and remaining evidence gap

As a diagnostic, the role policy produces 1.15 weeks scoring at least 12 points in
the next four, versus 0.94 for recent PPG. Its four-week points above a fixed
12-point weekly threshold average 6.44 versus 5.03. These measures illustrate how
total scoring and usable upside differ.

Twelve is a fixed screening threshold, **not** the best free agent, our actual lineup
replacement, or a league-derived starter cutoff. Next, replay legal start/sit and
add/drop choices using forecasts available before kickoff, actual roster slots,
bench/IR cost, and realistic acquisition constraints. Hindsight-optimal lineups are
only an upper bound. The recent wire-lineup repair provides a useful implementation
starting point, but its preseason season-equivalent values are not weekly forecasts.

## 5. Priority gates before expanding the research

1. **Data repair:** normalize transaction actions/direction and tie resolution;
   enforce contract dates; unify FB/RB outcome policy; fix route denominators.
2. **Rebuild and certify:** regenerate folds into a separate artifact version;
   reconcile known-player timelines; report unresolved records and missing coverage;
   require exact scoring/outcome checks and point-in-time source contracts.
3. **Research iteration:** retain simple snap/usage baselines; test position-specific
   RB receiving, short-yardage, and teammate-absence role transitions using admissible
   feeds. Small-sample players remain a hypothesis, not an approved preference.
4. **Decision validation:** archive room/waiver prices and weekly forecasts, then score
   executable roster decisions. A prospective shadow run is required before promotion.

The historical results above are exploratory. Seven seasons, repeated selections,
overlapping horizons, retrospective source revisions, and research choices prevent
treating these intervals as proof of a deployable edge. V1 remains the draft
reference; the original frozen 2026 forecast remains untouched.

## Reproduction

```sh
.venv/bin/python research/data_integrity_audit.py
.venv/bin/python research/role_transition_pilot.py
.venv/bin/pytest research/test_research_audits.py
.venv/bin/ruff check research/data_integrity_audit.py research/role_transition_pilot.py research/test_research_audits.py
```

Generated, ignored research outputs:

- `data/outputs/data_integrity_audit_2026-09-22.json`
- `data/outputs/research_audited_weekly_points.parquet`
- `data/outputs/research_transaction_replay.parquet`
- `data/outputs/role_transition_pilot_2026-09-22.json`
- `data/outputs/research_rb_role_predictions.parquet`
- `data/outputs/research_rb_role_selections.parquet`

The audit records code hashes, source file metadata, prediction hash, coverage,
reconciliation exceptions, and protected artifact hashes. The pilot records hashes
of its scored outcomes, ADP, player identity, and raw snap sources. Inputs and
protected boards are checked for changes during the audit; no production data is
refreshed, rebuilt, or overwritten. No 2026 outcomes enter either analysis.
