# Historical absence evidence and coverage

> Historical implementation record. Release IDs, results, commands, and interface descriptions reflect the recorded change; use the [operations guides](../operations/README.md) to run the system today.

This follow-up addresses missing public information at the forecast date. The
repaired July 17, 2019 input for Kareem Hunt still had no suspension evidence, and
its fitted forecast allowed 12.486 games and 220.057 points. His eight-game
suspension was already public in March. The relevant constraint is at most eight
available games, regardless of a regression's estimate.

## Completed rebuild

The completed version is `historical_absences_20260922_r3`, accepted for exploratory
research only. Its 18,289 forecast rows passed all 13 integrity gates, including
zero evidence-replay mismatches, zero future-evidence rows, and zero violations
across the baseline and 11 fitted games outputs. All 13 protected artifacts are
byte-identical. The manifest records 232 copied source files and 11 build outputs;
the separate acceptance record also binds the before/after comparison.

| Forecast | Prior fitted games | New fitted games | Prior points | New points |
| --- | ---: | ---: | ---: | ---: |
| Hunt, July 17, 2019 | 12.486 | 8.000 | 220.057 | 140.994 |
| Ridley, August 26, 2022 | 9.271 | 0.000 | 130.932 | 0.000 |

Hunt's fitted per-game rate remains 17.624. This removes the impossible availability
forecast; his inferred prior team and incomplete role evidence remain unresolved.
The other cases come from a full refit with constrained baseline availability
inputs, so the games forecasts can fall below their maximum available games.

Validation: **553 offline tests passed, 35 network tests deselected**; targeted Ruff,
mypy on four edited modules, and diff-whitespace checks passed. The first two
development runs were interrupted before completion and remain unaccepted.

- [Acceptance record](../../data/research/historical_absences_20260922_r3/acceptance.json)
- [Final integrity audit](../../data/research/historical_absences_20260922_r3/outputs/data_integrity_audit_2026-09-22.json)
- [Before/after comparisons](../../data/research/historical_absences_20260922_r3/outputs/availability_changes.json)
- [Coverage report](../../data/research/historical_absences_20260922_r3/outputs/availability_coverage.json)
- [Review queue](../../data/research/historical_absences_20260922_r3/outputs/availability_review_queue.parquet)

## Evidence, separate from inferred roster state

The new [evidence ledger](../../data/static/historical_absences.json) contains 14 dated
records for 12 player-seasons. Each has a stable GSIS player ID, season, incident
ID, publication date, date the fact became known, source URL, retrospective
verification date, and unavailable regular-season team-game numbers. These are
manually verified public facts, not numbers inferred from actual games played.

The initial ledger covers Hunt, Herndon, Edelman, Hopkins, Ridley, Watson, Kamara,
Williams, Tate, Rice, Jones, and Chubb. Eleven player-seasons have usable evidence
by their saved forecast dates. Tate's July 27, 2019 announcement is deliberately
excluded from the July 17 forecast. Williams's September revision is likewise
excluded from his August forecast. Watson's later 11-game decision replaces his
earlier six-game decision when the cutoff permits it.

Source examples:

- [Hunt announcement](https://www.nfl.com/news/nfl-suspends-browns-rb-kareem-hunt-eight-games-0ap3000001022826)
- [Herndon announcement](https://www.nfl.com/news/roundup-jets-chris-herndon-suspended-4-games-0ap3000001035664)
- [Watson settlement](https://www.clevelandbrowns.com/news/nfl-announces-11-game-suspension-for-qb-deshaun-watson)
- [Chubb's minimum PUP absence](https://www.nfl.com/news/browns-rb-nick-chubb-knee-to-start-2024-season-on-pup-list-miss-at-least-first-four-games)

The implementation uses end-of-day dates, matching the existing historical
forecast convention. It does not establish intraday publication availability.
Collection/verification in 2026 never becomes the historical knowledge date.

## Forecast behavior

`known_available_games_cap` equals the season's scheduled games minus the union
of known unavailable team games. Byes do not count, overlapping incidents are not
double-counted, and conflicting latest revisions are retained for review. A later
revision replaces only its own incident. Clearing one incident does not certify
general health or full availability. Missing evidence remains null.

Baseline expected games retain their original values in `expected_games_before_absences`.
The constrained baseline and every fitted games model are capped before downstream
season-point products, stacks, and selectors consume them. The same rule applies
inside validation folds and when serialized models are reapplied. Forecasts below
the ceiling stay below it; the known absence is not subtracted a second time.

A full-season known absence also forces direct season-point and return-probability
models to zero when they produce a forecast. Partial absences do **not** supply a
mathematically justified points ceiling for direct season-point regressions that
have no games/per-game decomposition. Those remain experimental comparators;
their outputs must not be described as availability-constrained forecasts.

The ledger does not manufacture forecasts for unsupported candidates. Edelman
2018 and Watson 2022 are `market_only` rows with no canonical returner-model score;
their availability caps are recorded without inventing a point estimate. Their
existing missing display names are also a separate identity/coverage issue.

## Coverage is still incomplete

The initial backfill is deliberately identified as partial. It adds absence
evidence for 11 of 18,289 rows; the other 18,278 have no curated absence evidence.
That does not mean all 18,278 players were absent, or that none were available.
It means this evidence source has not established their availability.

The earlier roster coverage counts remain unchanged:

| Roster evidence | Forecast rows |
| --- | ---: |
| Observed | 1,656 |
| Inferred prior team | 11,255 |
| Unresolved action | 370 |
| Conflicting same-day evidence | 13 |
| No returner transaction feature | 4,995 |

Of the inferred-team rows, **9,020 occur in 2004–2019**. Those seasons have zero
observed cutoff roster states in the accepted predecessor. The trusted official
club-transaction loader begins in 2020; previously rejected ESPN transaction dates
cannot fill that gap safely. Dated league and club announcements supply another
evidence path, but this first ledger is not a complete historical news archive.

`availability_coverage.json` reports these separate dimensions by season,
position, and population. `availability_review_queue.parquet` includes missing
transaction-feature rows as well as inferred, conflicting, and unreviewed rows.
It prioritizes ambiguous restriction labels and then prior production, without
consulting forecast-season outcomes. Report metadata also includes absence coverage.
`coverage_complete` remains false independently of passing integrity checks.

A further review finding: `cutoff_suspended` is an inherited transaction parser
label that can include exemption text or another player's clause. The Chubb 2024
source clause lists several players and reserve mechanisms together; Tonyan 2024
shares prose with an exempt practice-squad player. Oliver's January 2022 record
mentions both COVID exemption and return to the active roster. These are explicit
classification/review issues, not evidence of a known suspension duration. Chubb's
new cap comes from his player-specific PUP report, not that parser label.

## Reproduction and scope

```sh
.venv/bin/python research/rebuild_history.py --version YOUR_NEW_VERSION
.venv/bin/python research/data_integrity_audit.py \
  --data-dir data/research/YOUR_NEW_VERSION --require-clean
```

The source ledger is copied and hashed with each create-only rebuild. The rebuild
exports historical inputs, a coverage report, a review queue, refitted forecasts,
and the implementation snapshot. The integrity audit replays evidence selection
and checks games forecasts against their caps. It rejects dropped evidence
columns when the version contains an absence ledger. Old accepted versions remain
unchanged; a new acceptance record identifies the new research version.

Future backfill should prioritize fixed suspensions and mandatory PUP/NFI/IR
minimum absences, then dated retirement/opt-out decisions and roster transitions.
Use the rule and announcement in effect at the cutoff: training-camp PUP status,
an injury headline without a confirmed minimum absence, or an indefinite exemption
does not justify inventing a number of missed games. Expand dated source coverage
before treating the remaining unknowns as a regression problem.
