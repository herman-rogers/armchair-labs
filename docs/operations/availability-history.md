# Historical player availability

The shared availability dataset starts in **2002** and covers every position.
It preserves transaction, roster, and injury evidence independently of a model's
candidate list. Opening Drive v2's sealed inputs and results are unchanged; using
this dataset in a subsequent experiment requires a new dataset/model version.

The reviewed snapshot is
`data/research/availability_history_2002_2026_003/`. Its integrity manifest binds
the outputs, raw captures, imported official archives, and generator source hashes.
The independent review is in
`data/research/availability_history_review_20261010_003/audit.json`.

The October 10, 2026 build contains **1,152,624 usable observations for 17,041
players**, drawn from **115,075 transaction reports** plus roster and injury
records. All 25 ESPN year requests retrieved every reported page. All **12,574
completed, scheduled team-weeks** have roster observations. Another 2,824
not-yet-usable observations are preserved separately.

The audit passed, as did 57 targeted regression tests. Earlier team-roster evidence
resolved 6,019 transaction observations with otherwise ambiguous names. The review
queue still contains 48,479 items, including unsupported actions, unidentified or
ambiguous names, and bundled actions. These are review items, not necessarily
unique missing players or erroneous reports. There are 22,994 same-time conflict
groups across the full ledger; queries expose those conflicts rather than choosing
an arbitrary row. These counts keep the dataset's remaining uncertainty visible.

## Sources and scope

| Source | Captured history | Purpose |
|---|---|---|
| nflverse weekly rosters | 2002–2026 | Dated team membership and roster status, including players with no box-score production |
| ESPN transaction reports | 2002–2026 | Dated moves throughout the calendar year; all reported pages, including recovered 2003 records |
| Preserved official club archives | 2004–2026, uneven club coverage | Additional independent transaction descriptions |
| nflverse injury reports | Files from 2009–2026 | Separate injury-report evidence; usable dates depend on source timestamps |
| nflverse player identities | Captured identity crosswalk | IDs, names, and birth dates; no latest-team, career-end, or current-status inference |

The [weekly roster documentation](https://nflreadr.nflverse.com/reference/load_rosters_weekly.html)
establishes the 2002 boundary. [Injury files](https://nflreadr.nflverse.com/reference/load_injuries.html)
begin in 2009. Season-level roster history extends further back, but does not
establish availability at a historical forecast date and is not used here.

ESPN sometimes returns 404 for a particular year/page-size combination even when
the archive exists. The collector retries the alternate ESPN API representation,
verifies page indexes, counts, years, and repeated-page fingerprints, and never
combines overlapping partial attempts. A successful download means all *reported*
pages were retrieved, not that the publisher recorded every real-world move.

## Evidence clocks and states

Every event records its source, raw checksum, raw-record ID, collection timestamp,
event date, usable timestamp, date basis, and identity-matching basis. Publication
time stays null unless actually established; it is not invented from an event date.

- Date-only transactions become usable the next day at noon UTC.
- Weekly rosters become usable after the regular-season week's final game, on the
  following day at noon UTC. They do not supply preseason status.
- Injury modification timestamps are used when present. Without one, a regular
  season/week label permits only an explicitly flagged completed-week proxy.
  Other undated injuries are withheld. Some historical modification times are
  much later than the represented week; they cannot be used at the earlier cutoff.
- Not-yet-usable observations are stored separately in `pending_events.parquet`.

The query defaults to **observed** mode: both the usable clock and collection
timestamp must precede the cutoff. Historical training must explicitly request
**reconstructed** mode, which admits retrospective records with an earlier usable
date. This distinction prevents us from claiming we possessed a historical
snapshot that we downloaded years later.

Roster status, injury reports, retirement announcements, and return-window notices
are separate dimensions. A roster signing is `ROSTERED`, not proof of health or
game-day activation. Formal reserve/retired moves are distinct from retirement
announcements. Return-window designations do not imply activation. Injuries,
releases, reserve states, and missing evidence do not imply zero future production.
The dataset imposes no hard model eligibility rule.

Same-time status or team conflicts return `UNKNOWN`, with the competing evidence
IDs. Queries retain evidence age and source/date bases. Defaults expire roster and
retirement evidence after 550 days, injury-report evidence after 14 days, and
return-window notices after 30 days; callers can choose explicit age limits.
These are query freshness policies, not medical or league-rule assertions.

Transactions match exact normalized names. When a name maps to several players,
an earlier roster record on the reporting team can resolve it only if exactly one
candidate has qualifying evidence within 365 days. Retired, released, unknown,
and conflicting prior statuses do not qualify. The supporting evidence ID is
saved, and its collection time also constrains observed-mode use. Unresolved names,
trades, ambiguous bundled actions, and embedded chronology remain in the review
queue. Historical team abbreviations map to stable franchise codes; original
spellings remain in the captured source.

## Read and query

```python
from datetime import UTC, datetime
from pathlib import Path
import polars as pl

from engine.data.availability_history.build import verify
from engine.data.availability_history.ledger import as_of

root = Path("data/research/availability_history_2002_2026_003")
verify(root)
events = pl.read_parquet(root / "events.parquet")
states = as_of(
    events,
    datetime(2021, 9, 9, 12, tzinfo=UTC),
    mode="reconstructed",
    player_ids=["00-0020531", "00-0019596"],  # Brees and Brady
)
```

Omit `player_ids` to query the entire evidence population. Supply a candidate list
to get explicit `UNKNOWN` roster rows for candidates without usable evidence.
This supports building a dated candidate universe without filtering on eventual
appearances. It does not automatically replace an existing model's population.

## Build, refresh, and audit

```bash
uv run python -m engine.data.availability_history.build \
  --output data/research/availability_history_NEW_SNAPSHOT \
  --first-year 2002 --last-year 2026

uv run python -m engine.data.availability_history.audit \
  --dataset data/research/availability_history_NEW_SNAPSHOT \
  --output data/research/availability_history_NEW_REVIEW/audit.json
```

Use a **new output directory** for fresh captures. Reusing a directory resumes its
existing captures or verifies its sealed manifest; it does not silently refresh
old receipts. The collector is callable, but no recurring job is installed.
Official archive imports refer to the checksum-verified local backfill and its raw
files, so those dependencies must accompany a copied dataset.

`coverage.csv` reports observations and distinct players by calendar year/source.
`source_coverage.json` distinguishes acquisition from normalization failures.
`team_week_coverage.csv` lists expected scheduled team-weeks, roster counts, and
whether the week had completed at build time. `review_queue.parquet` and
`conflicts.parquet` expose unresolved evidence instead of quietly guessing.

The independent audit verifies source hashes, transaction/raw-record links,
calendar clocks, prior-roster identity evidence, future-record invariance, and
the Brees/Brady cases. Team-week coverage is a useful completeness check, but does
not prove that every player or transaction is present. Historical states remain
reconstructions; accumulating new dated captures improves prospective evidence.
