# Injury archive and RB forecasting rules — September 25, 2026

The separate marimo RB notebook now uses a verified, versioned injury archive and
dated availability rules. No models were trained for this update. New inputs and
rules apply when the user next submits **Train RB models**. Existing gold releases,
saved model artifacts, and the canonical data pointer were preserved.

## What was collected

Archive: [`injury_archive_20260925_r1`](../data/research/injury_archive_20260925_r1/).
The sealed manifest is pinned in `rb_availability.py` by SHA-256:
`fe52e77b885b6a38c4ff4ebb02f5dbfbb80ae5461c154bb0277684f36154ac11`.

| Material | Coverage/count | Use |
|---|---:|---|
| Original structured injury reports | 90,752 rows, 2009–2025, all positions | Preserve original values and timestamps |
| Historical transactions | 95,630 rows | Retained context; a transaction date alone is not a publication timestamp |
| Additional official injury records | 2,855 rows, 2001–2008 | Archive only until dates/identities are established |
| Secondary early weekly reports | 168 captured pages | Archive only; not approved forecast constraints |
| NFL news index | 157,649 articles, full-calendar collection attempted for 2001–2025 | Discovery index, not 157,649 injury articles |
| RB-name/health-related news candidates | 2,228 captured bodies, all HTTP 200 with publication metadata | Broad candidates; includes college/other mentions and requires review |
| Weekly articles for timestamp matching | 33 captured sources, with January 2026 included for the end of the 2025 season | Strict corroboration of player/week/injury/status |
| Reviewed absence events | 107, up from 97 | Includes uncertain announcements and revisions; not 107 full-season injuries |

All positions are retained in the structured reports and timestamp corroborations;
the news-body discovery and notebook integration focus on RBs. The earlier news
collection covered January–August. This collection also searches September–December.
Empty early NFL index pages are recorded, not counted as complete historical coverage.

## Publication timestamps recovered through cross-referencing

The original feed has **10,934 records without an update timestamp**: 4,804 in 2009,
62 in 2010, and all 6,068 in 2025. Dated NFL reporting corroborates **2,233 distinct
original records**, including **155 RB records**:

- 2025: 2,229 records, including 151 RB records, matched against numbered NFL weekly
  articles using player name, season, week, injury, and game status.
- 2009: four manually checked RB records—James Davis, Jonathan Stewart, Pierre
  Thomas, and LaDainian Tomlinson—from the September 18 weekly report.

These are **separate dated observations**, not invented original-feed timestamps.
`corroborated_reports.parquet` retains the original record ID, publication timestamp,
modification timestamp, source URL, capture hash, match basis, and retrieval date.
The eligible timestamp is the **later of publication and modification**. Practice
participation remains unknown unless separately established; corroborating an
“Out” designation does not prove “Did Not Participate.”

The original missing values remain intact. **8,701 original undated records still
lack dated corroboration**. Additional early records also lack usable publication
timestamps; 290 have unresolved player identities. The year-by-year CSV reports
these gaps instead of treating missing coverage as evidence a player was healthy.

Sources include the [NFL weekly injury articles](https://www.nfl.com/news/nfl-week-3-injury-report-player-statuses-2025-season),
[2009 weekly reporting](https://www.nfl.com/news/mcnabb-bryant-among-fantasy-players-who-could-miss-week-2-09000d5d812b8c34),
[official older weekly tables](https://www.nfl.com/injuries/league/2008/reg1), and
[John Troan's early weekly archive](https://www.jt-sw.com/football/pro/index.nsf/Documents/2001-ir).
The NFL media archive was discovered but direct capture failed with certificate/access
errors. No paid/licensed injury archive was acquired. Public coverage is not complete.

## Rules used in the notebook

1. Keep the actual outcomes unchanged, including zero-production players. Missing
   outcome values remain unknown under the existing target-data rules.
2. Resolve evidence using what was known by each forecast cutoff. Preseason uses
   its saved cutoff; in-season forecasts use the end of the day before the first
   predicted NFL week starts. Timestamp comparisons use source UTC civil dates,
   conservatively excluding observations that fall on the following UTC date.
3. Resolve revisions per incident; ambiguous same-date revisions disable the zero
   override. Later uncertain/cleared revisions can remove earlier constraints.
4. A confirmed absence covering the **entire forecast period** forces the model
   and baseline to zero. Injury diagnosis, ordinary IR placement, or an uncertain
   recovery estimate alone does not establish a full-season absence.
5. Partial and uncertain absences enter as features. Do not multiply a predicted
   season total by another availability fraction: the model already learns totals.
6. Map team-game ordinals to calendar weeks only when dated team evidence supports
   it. The in-season implementation deliberately does not carry a preseason team
   forward to map partial suspensions. Reviewed explicit calendar-week absences
   and full-season absences can still constrain an in-season horizon.
7. Preserve raw model/baseline forecasts. Display raw and adjusted final-test
   metrics, per-player evidence and adjustments, and an exportable audit table.
   Capacity and learning curves use adjusted forecasts; capacity retains raw RMSE.
   Validation selects tree count; test outcomes do not select it.

Eight additional inputs describe known absence games/weeks, confirmed full-period
absence, uncertainty, dated/current recent reports, latest game status, and latest
practice participation. Only eligible evidence becomes inputs; the entire news
archive is not automatically converted to model features.

Reviewed historical announcements can reconstruct a knowledge date from explicit
dated team/player statements even when the modern page shows a later CMS update.
That reconstruction basis is recorded. Examples include Barkley's 2020 team
announcement and Mostert's dated September 2021 player announcement. These are
retrospective source reconstructions, **not original publication snapshots**.
Automatically matched injury reports do not receive that exception.

Ekeler's September 12, 2025 suspected Achilles injury remains uncertain; the
September 13 MRI-confirmed report supersedes it. Chubb's September 2023 absence
uses the Browns' explicit confirmation. The Cam Akers diagnosis is retained as an
uncertain full-period absence and cannot force zero.

## Three audit passes

1. **Source/data integrity:** verified 2,946 compressed source captures against
   their content hashes; checked corroborations against original player/week/status/
   injury fields and parsed publication/update metadata. Original injury rows and
   timestamps are unchanged. Early linked players have no future rookie-season
   contradictions; unresolved identities remain unlinked.
2. **Temporal and outcome replay:** replayed 20,152 RB forecast examples across
   season, next-week, next-four, and remaining-season horizons. Injected future
   evidence did not alter historical annotations. Actual targets stayed unchanged;
   partial absences did not scale forecasts. There were zero outcome disagreements
   for the full-period zero overrides across the seven stats checked. These are
   data/rule checks, not new model performance estimates.
3. **Saved forecast effects:** checked eight existing RB forecast artifacts without
   fitting a model. In the saved 2025 remaining-season forecasts, Brooks and Ekeler
   receive the zero rule. Raw forecasts are retained. This is a rule-only comparison
   against saved research artifacts, not the user's current live notebook run.

The targeted test suite passed **44 tests**. The marimo notebook passed static
validation, both new reporting cells rendered against real data without fitting,
and the training-cell wiring was checked with fixed fake estimator callbacks.

Audit outputs: [summary](../data/research/injury_archive_20260925_r1/audit_summary.json),
[coverage by season](../data/research/injury_archive_20260925_r1/coverage_by_season.csv),
[historical replay](../data/research/injury_archive_20260925_r1/temporal_replay.json),
[saved forecast comparison](../data/research/injury_archive_20260925_r1/existing_forecast_audit.json).

## Reproduction and further expansion

The collectors resume immutable URL captures while an archive is being built.
Completed releases are sealed; choose a new `OUT` version in the shared collector
before expanding or repeating collection. Run from the repository root, in order:

```sh
.venv/bin/python research/build_injury_archive.py
.venv/bin/python research/extend_early_injuries.py
.venv/bin/python research/recover_injury_dates.py
.venv/bin/python research/review_injury_evidence.py
PYTHONPATH=. .venv/bin/python research/audit_injury_archive.py
```

After reviewing a new audit, pin its final manifest hash in `rb_availability.py`.
Further work is documented by the missing coverage, unresolved identities, and
unreviewed news candidates. None of those are silently promoted to zero rules.
