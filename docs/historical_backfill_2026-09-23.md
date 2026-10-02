# Systematic historical public-evidence backfill

The first pass runs dated roster and absence reconstruction across all 18,289
historical forecast rows. The isolated version is
`historical_backfill_20260923_r2`, compared with
`historical_absences_20260922_r3`. The full refit and final audit passed, and the
version is accepted for research. All 13 protected live/frozen artifacts remain
unchanged. The preliminary `r1` rebuild remains unaccepted.

## Measured coverage

| Measure | Before | After |
| --- | ---: | ---: |
| Observed roster state | 1,656 | 7,161 |
| Club established, status still unknown | 0 | 301 |
| Only inferred prior team | 11,255 | 8,679 |
| Forecasts excluded from transaction feature construction | 4,995 | 0 |
| Forecasts with a reviewed, applicable absence | 11 | 91 |
| Missing player display name | 723 | 0 |
| Unresolved transaction action | 370 | 224 |
| Conflicting same-day states | 13 | 46 |

The observed roster states cover 4,266 returners, 2,530 rookies, and 365
market-only players with observed states. Removing the population exclusion does
not create evidence: 1,878 candidates still have neither a prior team nor a
resolved dated event. Another 301 have club evidence without an active/reserve
status. Releases count as observed roster states; they do not establish a current
club or an absence for the whole season.

The main gain is before 2020: inferred-prior-team rows fell from 9,020 to 6,456.
For 2020 onward they fell from 2,235 to 2,223. New conflicts are retained as review
items, not settled using archive row order.

## Sources collected

The collector attempted every official club/year route for 32 clubs and
2004–2026: 736 pages. It accepted 595 archives containing 69,681 transaction rows.
An explicitly dated later activation inside one compound record adds one derived
event, making 69,682 exported rows. These are full-year collection totals; each
forecast uses only its eligible dates.

The other archive attempts remain explicit gaps: 95 returned no records, 23
Philadelphia pages did not establish the requested canonical year, and 23 Dallas
routes returned errors. Six clubs have usable records in 2004; 30 have records in
2018–2026. A page passing the route checks does not prove the club recorded every
event. Repeated tables across different years are rejected.

The NFL sitemap pass attempted 184 January–August month pages, found records in
154, and indexed 95,163 articles. The current index starts in 2007. Two candidate
passes and targeted follow-ups captured 1,032 distinct article bodies. The legacy
pass includes surname-only headlines, which require manual identity resolution.
Headline matching supplies a review queue; it never generates game constraints.

Club archives provide event dates, not publication timestamps. Some displayed
archive dates differ by a day from contemporaneous announcements (for example,
the Saints’ July 2020 opt-out entry). This pass retains the archive date basis
explicitly; independent validation of those dates remains a coverage-quality
limitation, especially for events close to a forecast cutoff.

Every capture retains its requested and final URL, HTTP status, collection time,
compressed raw HTML, and SHA-256. The transaction export is bound to its source
captures by a manifest. Reviewed absence additions also retain raw source hashes,
publication dates, modification dates, and the review basis. The rebuild copies
the source cache and runs with network access disabled.

## Changes in interpretation

- Rookies and market-only candidates now enter the same dated transaction replay
  as returners. Their initial team is unknown; current player metadata does not
  establish a historical team.
- Stable player and crosswalk identities restore all 723 missing names. Only
  unambiguous stable names are added; current teams and biographies are excluded.
- Shared reserve clauses are interpreted for the named player. Nick Chubb's 2024
  PUP placement no longer inherits another player's exempt-list mechanism.
- Concatenated sentences, present-tense archive verbs, signings, releases,
  contract extensions, draft rights, and explicit IR activation wording are
  recognized. Draft rights and extensions can establish a club without proving
  active status.
- A trade of picks **for** a player recognizes the acquiring club. Julio Jones's
  2021 evidence establishes Tennessee.
- COVID and exempt mechanisms are distinct from suspension. Josh Oliver's 2022
  archive includes a January 6 activation inside a January 3 record; the extracted
  activation cannot affect a forecast before January 6.
- News review expanded the ledger from 14 to 97 announcements, covering 94 player
  seasons and 88 players. Ninety-one player seasons have evidence usable at the
  actual forecast cutoff: 82 returners, six market-only players, and three rookies.
  Revisions replace the same incident rather than adding suspension lengths.

Hunt's July 17, 2019 record now has Cleveland signing evidence and the previously
repaired eight-game cap. The [March NFL announcement](https://www.nfl.com/news/nfl-suspends-browns-rb-kareem-hunt-eight-games-0ap3000001022826)
remains the source of the absence constraint. A signing records a roster action;
it does not clear a separately announced suspension.

The finished Hunt forecast retains eight games and 140.99 fitted season points,
now with observed Cleveland club evidence. Damien Williams's 2020 forecast moves
from 8.97 fitted games and 80.04 points to zero games and zero points after his
full-season opt-out is applied. These examples describe factual corrections, not
a claim of aggregate predictive improvement.

Examples of newly reviewed facts include Damien Williams's 2020 opt-out,
Antonio Gates's 2015 suspension, and Bell's reduced suspensions. Conditional
additional absence is not treated as certain: Vincent Jackson's 2010 record uses
the unconditional three-game suspension, not a possible six-game absence. Pitta's
initial 2013 season-ending outlook was excluded after locating later reporting
that reopened a return before the forecast cutoff.

## Remaining gaps and next batches

There are still 18,198 forecasts without a reviewed absence fact. This is a count
of unknown evidence status, not a count of missed suspensions or injuries.
Retirements, indefinite suspensions without a bounded absence, speculative injury
reports, and articles about other people require further review. Headlines alone
also miss relevant facts in roundups and articles with less explicit wording.

Priority follow-ups are the 224 unresolved actions and 46 same-day conflicts,
additional dated absence announcements and alternate
sources for missing club/year archives. The collection and review inventories
make these gaps enumerable and resumable.

The second review pass resolved opt-out omissions for Devin Funchess, Matt
LaCosse, Jason Vander Laan, and Da'Mari Scott, then checked the rest of the 2020
skill-position opt-out cohort. For the rolling NFL opt-out list, the ledger uses
August 7 as the conservative known-on date after the explicitly reported August 6
deadline, rather than treating the list's original July 28 publication date as
proof that every later addition was already known. The list's later modification
timestamp remains attached to the evidence.

The name review also resolved all 19 IDs absent from the main player table using
the existing crosswalk and historical market identity backfill. Conflicting name
mappings are rejected, and identity supplementation never imports a current team.

This pass does not solve historical depth-chart or contract-date coverage:
depth-chart rank remains present in 1,161 rows from 2025–2026; canonical signing
dates, money, and remaining contract terms are still unavailable. Complete-team
vacated target and carry shares remain null because team coverage is not proven
complete. Partial known-vacated measures retain their separate interpretation.

These are retrospective reconstructions from currently available historical
archives. Source publication dates and later modification dates are retained,
but a current page is not an original publication-time snapshot. A successful
integrity audit establishes consistent use of the evidence; it does not establish
complete coverage or improved predictive accuracy.

## Reproduction and validation

Collection, reviewed export, and model rebuilding are separate operations:

```sh
.venv/bin/python research/backfill_public_evidence.py \
  --output data/research/evidence_backfill_20260923_r1
.venv/bin/python research/export_transaction_backfill.py \
  --collection data/research/evidence_backfill_20260923_r1
.venv/bin/python research/import_absence_reviews.py \
  --reviews data/research/evidence_backfill_20260923_r1/approved_absence_reviews.json \
  --players data/cache/nflverse/dd7fbf2265c3529eb5ef6e211cdc5feb.parquet
.venv/bin/python research/rebuild_history.py --version A_NEW_VERSION
.venv/bin/python research/accept_evidence_backfill.py \
  --version data/research/A_NEW_VERSION \
  --baseline data/research/historical_absences_20260922_r3 \
  --collection data/research/evidence_backfill_20260923_r1
```

Rebuild versions are create-only. Raw captures are reused without silently
refreshing historical evidence. The separate article-harvest command takes the
forecast and identity paths and supports `--legacy`; imported decisions refer to
explicit URLs and stable player IDs and require previously captured source text.

Validation: 571 offline tests passed, 35 network tests excluded;
Ruff passed for the changed files; mypy passed for the five changed data/metric
modules. All 13 input and final integrity checks passed, including cutoff-date constraints,
absence caps, and transaction replay in reversed input order. Replay now covers
all candidate populations and the pending 2026 forecast. The audit reports this
scope as `all_candidate_rows_replayed`.

Final acceptance verified 4,140 source files, the rebuilt predictions and other
output hashes, unchanged outcome labels and candidate keys, and all 13 protected
live/frozen artifacts. There were zero changes to actual games, actual season
points, or outcome-completeness labels.

Acceptance caught an audit-export defect: replay rows had unspecified ordering,
and the final audit overwrote the initial replay artifact. All 21 replay fields
were verified against both retained input and finished forecast rows for all
18,289 candidates. The audit-only artifact was canonicalized before acceptance,
with its previous and replacement hashes recorded in the manifest. Final audits
now export a separate file in stable key order. Source inputs and predictions
were not rewritten. The final audit and acceptance procedures are included with
the accepted evidence collection.

The accepted artifacts are:

- [Acceptance record](../data/research/historical_backfill_20260923_r2/acceptance.json)
- [Coverage changes and forecast examples](../data/research/historical_backfill_20260923_r2/outputs/backfill_coverage_changes.json)
- [Final integrity audit](../data/research/historical_backfill_20260923_r2/outputs/data_integrity_audit_2026-09-22.json)
- [Remaining roster review queue](../data/research/historical_backfill_20260923_r2/outputs/roster_evidence_review_queue.parquet)
- [Article review status](../data/research/historical_backfill_20260923_r2/outputs/evidence_collection/absence_review_status.json)
- [Restored identity names](../data/research/historical_backfill_20260923_r2/outputs/backfilled_identity_names.parquet)
