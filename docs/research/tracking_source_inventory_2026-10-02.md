# Tracking source capture — October 2, 2026

Six public source collections were downloaded and verified. Kaggle competition archives remain access-gated. See the [source inventory](../architecture/data-sources.md) for provider citations and the [runbook](../operations/tracking-data.md) for repeatable commands.

## Selected local releases

| Source | Version | Gold tables | Rows across tables |
|---|---|---:|---:|
| `bdb_2019_sample` | `tracking_bdb_2019_sample_20261002_r4` | 4 | 332,022 |
| `nflverse_ftn` | `tracking_nflverse_ftn_20261002_r2` | 5 | 193,280 |
| `nflverse_ngs` | `tracking_nflverse_ngs_20261002_r2` | 6 | 27,336 |
| `nflverse_participation` | `tracking_nflverse_participation_20261002_r2` | 10 | 478,989 |
| `nflverse_players` | `tracking_nflverse_players_20261002_r2` | 1 | 24,844 |
| `nflverse_rosters` | `tracking_nflverse_rosters_20261002_r2` | 12 | 43,850 |

Totals mix table grains; they are not independent plays or training examples. The roster total includes its derived crosswalk, and the sample total includes games/players/plays metadata.

## Measured coverage

- Official tracking sample: **316,025 entity-frame rows, 177 distinct plays, one game (`2017090700`), 89 distinct player IDs**, plus football rows. Its 91 games, 14,193 plays and 1,713 players in accompanying metadata must not be reported as tracked coverage.
- **27 rows have missing coordinates**. Native out-of-field coordinates are retained and flagged. Play direction is absent in this sample, so attack-normalized coordinates remain null.
- Legacy sample player IDs have **0/89 exact matches** in the current roster GSIS IT crosswalk; all 302,282 player-frame rows remain unmapped. The 13,743 football rows are marked `not_a_player`. Matching names alone was not used to manufacture identity.
- NGS: 27,336 provider rows across passing, receiving and rushing, split into six gold tables for weekly versus week-0 season totals.
- Participation: **478,989 rows**, yearly partitions 2016–2025. No live 2026 participation archive was captured.
- FTN charting: **193,280 rows**, 2022–2026; 2026 is partial (8,065 rows).
- Player identity: 24,844 rows. Rosters: eleven yearly partitions, 2016–2026, plus a derived crosswalk.

## Verification and limits

All six selected releases were loaded through recursive raw/enriched/gold manifest verification. Source hashes, schemas, row counts, roles and coverage summaries are retained in their generated manifests. Public HTTP captures preserve GitHub asset or commit identifiers; collection time is not treated as historical publication time.

The focused tracking and existing data-pipeline suite passed **23 tests**, covering coordinate handling, missing identities, namespaces, ambiguous crosswalks, separate prediction clocks, handoff snapshots, overlapping partitions, NGS week-0 separation, tampering, safe extraction, download truncation and verified retries. Ruff passed for the added implementation and tests.

Earlier interrupted or superseded versions remain local evidence; only the versions listed above are selected. No existing analytical release was replaced. No model training, video processing, cloud upload or competition data redistribution was performed.

## Remaining work

1. Configure authorized Kaggle access and retrieve authenticated file inventories and archives. Unauthenticated API probes returned 401; this is an access limitation, not proof of archive removal.
2. Validate each full competition schema, true coverage, overlaps, and labels against real downloaded files. Current competition adapters have fixture coverage but have not ingested those full archives.
3. Establish a verified legacy ID bridge and canonical game/play joins before combining tracking with existing serving data.
4. Build archive-appropriate features and time/game-held-out evaluations, then assess incremental value over existing usage/play-by-play/NGS baselines.

## Authenticated Kaggle retry

After account setup, Kaggle CLI 2.2.4 successfully listed competition files. Download requests still returned HTTP 403 Forbidden for BDB 2020–2023 and 2026 analytics. The 2026 prediction archive downloaded successfully (107,074,944 compressed bytes). Downloading the 2025 README also returned 403. This separates successful file discovery from actual download entitlement; account/competition rules or another access restriction still needs resolution. The 2026 prediction archive was extracted and ingested; see validation results below.

| Competition | Listed files | Listed uncompressed bytes | Observation |
|---|---:|---:|---|
| BDB 2020 | 5 | 289,895,667 | Data files listed; downloads forbidden |
| BDB 2021 | 20 | 2,334,575,123 | Data files listed; downloads forbidden |
| BDB 2022 | 7 | 5,003,924,109 | Data files listed; downloads forbidden |
| BDB 2023 | 12 | 965,072,781 | Data files listed; downloads forbidden |
| BDB 2024 | 1 | 78 | README only |
| BDB 2025 | 1 | 78 | README only |
| BDB 2026-prediction | 49 | 864,820,234 | Downloaded successfully |
| BDB 2026-analytics | 37 | 863,837,579 | Data files listed; downloads forbidden |

The 2026 listings expose 2023 Weeks 1–18 input/output files, not the previously expected 2023–24 coverage. Treat the observed listing as the available archive inventory until additional files are established. Safety competition listings were also accessible but initial requests are paginated and are not a complete inventory. Raw listing responses are retained under `data/.runtime/kaggle-inventory-20261002/`.

### 2026 prediction validation

Selected release: `tracking_bdb2026_prediction_20261002_r2`. All 38 CSV files from the archive were preserved and parsed. Evaluation Python modules were not executed. The adapter was corrected to normalize `test_input.csv` as test input frames while retaining `test.csv` as source context. Output coordinates remain opt-in labels.

| Split | Phase | Entity-frame rows | Distinct games |
|---|---|---:|---:|
| test | input | 49,753 | 3 |
| train | input | 4,880,579 | 272 |
| train | output | 562,936 | 272 |

Crosswalk results for input frames: [{'mapping_status': 'ambiguous', 'len': 23678}, {'mapping_status': 'exact_provider_crosswalk', 'len': 4906654}]. Provider test rows may overlap public training games; these are not claimed as an independent holdout. The 2026 competition archive is now locally available for research; other denied downloads still require account/competition access resolution. The final focused suite passed 24 tests and Ruff passed.

### Access diagnosis

Direct SDK probes confirmed `RulesAcceptanceRequired` for BDB 2020–2025 and 2026 analytics, rather than a credential failure. The [runbook](../operations/tracking-data.md#resolving-competition-access) now includes the account action and competition-specific links. README-only 2024/2025 listings remain a separate archive-availability issue. No rules were accepted or external messages sent by the agent.

### 2020 archive after rules acceptance

The user accepted the 2020 rules and the official CLI download succeeded. Selected release: `tracking_bdb2020_20261002_r4`. The archive's `train.csv` contains **682,154 player snapshots across 31,007 plays, 688 games, and 2,570 player IDs**. Each play has 22 rows. Actual season coverage is 2017 (261,800 rows), 2018 (247,962), and 2019 (172,392), expanding the original expected 2017–18 coverage. These are handoff snapshots, not trajectories.

Provider wind fields contain mixed numeric/text values (including `SSW` in `WindSpeed`). The CSV reader now preserves these columns as strings; a regression test covers text appearing after the schema-inference window. This archive also uses legacy NFL IDs: all player rows remain unmapped against the modern roster crosswalk, and the canonical namespace is explicitly `legacy_nfl` to prevent accidental joins.

Raw training bytes, enriched provider columns (including outcome labels), and canonical gold snapshots are preserved and hash-verified. Encrypted test files and bundled competition executable code remain in the downloaded archive but were not executed, decrypted, or represented as usable observations. Download/extraction evidence is retained under `data/.runtime/tracking/tracking_bdb2020_20261002_r1/kaggle/`; the successful release was imported from those extracted bytes. Earlier failed/superseded versions were not selected. The serving catalog remains unchanged.

Focused verification: 25 tests passed, plus Ruff and recursive release verification.

### Retry of remaining competitions

Retried full downloads for BDB 2021–2025 and 2026 analytics after the user's additional access attempts. **2026 analytics succeeded** and is selected as `tracking_bdb2026_analytics_20261002_r1`, with 37 verified tables. All 36 training input/output CSV files are byte-identical (SHA-256) to the 2026 prediction archive; they add no independent tracking observations. Its additional `supplementary_data.csv` contains **18,009 rows** of source context. Raw objects deduplicate by content hash; source-specific gold partitions remain separately identifiable. Do not concatenate the two tracks as independent training data.

BDB 2021, 2022, 2023, 2024 and 2025 still returned HTTP 403. Direct SDK error bodies reconfirmed `RulesAcceptanceRequired` for each. The user reported that acceptance was unavailable for some competitions. The next step for those is Kaggle support/competition-host clarification about closed-competition research access; no request was sent. No further login or CLI reinstall is indicated.

### Public mirror acquisition result

Downloaded and ingested the SumerSports 2024 reproducibility ZIP as `tracking_bdb2024_sumersports_20261002_r1`. The verified source release contains **12,187,398 tracking rows across 12,486 plays and 136 games**, with nine weekly tracking partitions and games, players, plays and tackle tables. It includes 17,426 tackle-label rows. This is new tackling evidence from the 2022 season, not an additional season beyond that coverage.

The downloaded ZIP SHA-256 is `5813b8d000b9e3162c7146dffd2055f11a552b8398ea58a7fb439355c37625c2`. Its GitHub release metadata and download receipt are retained under `data/.runtime/tracking-mirrors-20261002/`; CSV bytes and parsed tables are sealed in existing raw/enriched/gold layers. The source registry explicitly identifies SumerSports as the republisher. Official archive byte equivalence has not been established because the original Kaggle archive is unavailable to this account. No model or script from the external repository was executed.

Validation: all source layers verified on load; duplicate frame keys and canonical schema checks passed. The focused suite passed 25 tests, and Ruff passed. StatsBomb remains inventoried, not ingested; the other missing competition archives remain unresolved.
