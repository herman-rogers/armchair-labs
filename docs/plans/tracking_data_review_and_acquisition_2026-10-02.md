# Tracking data review and acquisition plan

Reviewed October 2, 2026. We have real player-position observations and supporting play context available for research. The next task is to connect identities and plays, measure archive coverage, and test whether spatial information adds value beyond existing usage and play-by-play features. Ingestion is implemented; feature engineering, model evaluation and production promotion are proposed work.

See the [source inventory](../architecture/data-sources.md), [measured capture report](../research/tracking_source_inventory_2026-10-02.md), and [operating guide](../operations/tracking-data.md). The source catalog and manifests remain the authority for selected data versions.

## Data already acquired

| Dataset | Measured coverage | Research use and limits |
|---|---|---|
| 2026 Big Data Bowl prediction | 4,880,579 input player-frame rows and 562,936 output-label rows from selected passing plays across 272 games in 2023 Weeks 1–18. Also 49,753 public test-input rows. | Study movement, receiver/defender spacing, and movement during passes. This is selected play footage represented as coordinates, not continuous tracking of every play. Public test examples are not assumed independent of training. |
| 2026 analytics | 18,009 supplementary context rows | All 36 training tracking files are byte-identical to the prediction archive. Use its context without counting the same observations twice. |
| 2020 Big Data Bowl | 682,154 player snapshots, 31,007 rushing plays, 688 games, 2,570 player IDs; actual archive spans 2017–2019 | All 22 players at handoff. Study blocking/defensive geometry and expected rushing outcomes. One snapshot cannot reveal a route or movement sequence. |
| Official 2019 sample | 316,025 entity-frame rows, 177 plays, one 2017 game | Useful for animation and pipeline development. Larger companion metadata does not imply larger tracking coverage. |
| nflverse NGS | 27,336 provider rows across passing, rushing and receiving | Tracking-derived summaries; weekly and season totals kept separate. Not raw coordinates. |
| nflverse participation | 478,989 records, 2016–2025 | Who was on each play; historical publication delays matter for forecasting. |
| FTN charting | 193,280 records, 2022–partial 2026 | Motion, play design and other charted context; some fields are outcomes. |
| Players and rosters | 24,844 player records and roster partitions for 2016–2026 | Identity mapping and roster context. Current status/team cannot be used as historical facts. |

These releases reuse raw/enriched/gold storage and have independent source selection. The app's serving release was not replaced. No video collection or live league-wide tracking feed was obtained.

Modern tracking mostly maps through the roster GSIS IT bridge, with ambiguous mappings retained as unknown. The 2019 sample and 2020 archive use legacy IDs and still need an independently verified bridge. We preserve missing/out-of-field coordinates, retain source outcome fields in enriched evidence, and separate prediction labels from the default frame reader.

## Search for missing archives

A submission repository can contain analysis code, small metadata tables, derived features, or a full archive. These are different deliverables. Public mirrors should retain their publisher, source URL, captured revision, hash and acquisition method; they must not be labeled as an official Kaggle download or silently replace a source vintage. Repository code licenses do not automatically relicense NFL data.

| Missing source or alternative | Search result | Action and status |
|---|---|---|
| BDB 2024 tackling | [SumerSports SportsTrackingTransformer](https://github.com/SumerSports/SportsTrackingTransformer) explicitly states the host removed the original archive and provides a reproducibility copy in [data-v1.0 releases](https://github.com/SumerSports/SportsTrackingTransformer/releases/tag/data-v1.0). API inventory lists a 293,090,157-byte ZIP and a 225,322,310-byte tar.zst. | Downloaded public ZIP, retained release metadata and digest, inspected files and ingested as distinct `bdb_2024_sumersports` research source. Completion and measured counts recorded below. No claim of newly granted commercial rights or official byte equivalence. |
| BDB 2023 pass protection | [BenRossJenkins NFLBDB2023](https://github.com/BenRossJenkins/NFLBDB2023) contains games, plays, players and PFF metadata plus an author-linked [Google Drive folder](https://drive.google.com/drive/folders/1t79_ujDkUsgPUaYADOvYWAB-n1VXI2ga). | GitHub tree has no weekly tracking CSVs. Drive contents/access were not verified by the web tool. This is a lead, not a recovered archive. |
| BDB 2022 special teams | [egrace479 NFL-BDB-2022](https://github.com/egrace479/NFL-BDB-2022) documents tracking schema, source row counts and processing. | Inspected repository tree did not expose raw tracking CSV/Parquet archives. Useful implementation reference, not a download replacement. |
| BDB 2021 pass defense | [Fourth-and-Four](https://github.com/Fourth-and-Four/NFL-Big-Data-Bowl-2021), [rossdrucker](https://github.com/rossdrucker/big_data_bowl_2021) | Inspected trees did not expose raw CSV/Parquet tracking files. Source descriptions and local file references do not establish available bytes. |
| BDB 2025 pre-snap behavior | [dhdzmota submission](https://github.com/dhdzmota/NFL-Big-Data-Bowl-2025), [abfhdays blitz model](https://github.com/abfhdays/bdb25-blitz1) | First contains only a small position dictionary among inspected data files; second describes cleaned weekly data but does not expose it in the inspected tree. No complete 2025 archive verified. |
| StatsBomb American football | [Hudl StatsBomb American Football Open Data](https://github.com/hudl/amf-open-data) provides play/event data, low-frequency coordinates and high-frequency tracking links, independent of Kaggle. | Newly discovered provider-published alternative. Downloaded README, license and game index for inspection. Index contains 124 games spanning 2016–2022; full tracking files have not been ingested. Prioritize a dedicated adapter next. |

Search covered official competition pages, GitHub repository trees and release assets, author-linked storage, Hugging Face search results and academic/reproducibility projects. No full replacement for BDB 2021, 2022, 2023 or 2025 was verified in this pass. Kaggle still returns `RulesAcceptanceRequired` for those years; 2024/2025 listings expose only a README. Public mirrors change where bytes are obtained, not the underlying data-use terms. No restricted account endpoint was bypassed.

## StatsBomb follow-up

This corrects the earlier inventory: there is an American-football-specific open-data repository, separate from StatsBomb's soccer repository. The current `data/games.json` index has 18 games for 2016, 16 for 2017, 16 for 2018, 17 for 2019, 20 for 2020, 19 for 2021 and 18 for 2022. These are Tom Brady Data Biography selections, not all teams' full seasons. The README links per-game compressed JSON and season ZIPs; file payload coverage still requires validation. [Provider repository](https://github.com/hudl/amf-open-data).

Its public-data agreement is research-oriented, restricts commercial exploitation and third-party data distribution, and requires attribution for published analysis. Record these terms separately from nflverse and NFL competition data. Before integrating, inspect its native event/frame schema, coordinate conventions, sampling, inferred observations and player IDs; do not run it through a BDB parser merely because both contain x/y. [Provider agreement](https://github.com/hudl/amf-open-data/blob/main/LICENSE.pdf).

## Proposed research sequence

1. **Finish identity and game joins.** Preserve separate ID namespaces, measure ambiguous/unmapped players, and construct explicit old-style game ID to nflverse game/play bridges. Acceptance: counts, unmatched examples and collision tests per archive; no guessed identities.
2. **Build archive-specific quality reports.** Measure frames per play, missing players, event coverage, coordinate ranges, sampling intervals and overlaps. Confirm mirror files against available official schemas and provider metadata. Acceptance: every usable partition has provenance and measured coverage; raw bytes remain preserved.
3. **Start with rushing geometry.** Use handoff positions to model opportunity and expected gain, then investigate residual performance. Acceptance: game/time-held-out performance against existing situational baselines; no final yards or post-handoff evidence in input features.
4. **Study passing movement and spacing.** Use 2023 sequences and supplemental context to evaluate separation and defender reactions at defined event cutoffs. Acceptance: labels/future frames separated, provider test overlap removed, and gains measured against simple movement/context baselines.
5. **Add tackling and alternate tracking sources.** Validate the 2024 mirror and build a dedicated StatsBomb adapter. Treat differing providers as separate measurement systems until cross-source consistency is demonstrated.
6. **Evaluate fantasy relevance before serving.** Ask whether resulting player summaries persist and improve held-out workload/efficiency forecasts. Archive studies alone do not justify live features: specify how a comparable signal could be available at each prediction date and during the current season.

The main hypothesis is that spatial context can help separate opportunity from player execution—for example, favorable blocking from what a runner does with that blocking. This is a research hypothesis, not an established improvement to our models.

## Public mirror acquisition result

Downloaded and ingested the SumerSports 2024 reproducibility ZIP as `tracking_bdb2024_sumersports_20261002_r1`. The verified source release contains **12,187,398 tracking rows across 12,486 plays and 136 games**, with nine weekly tracking partitions and games, players, plays and tackle tables. It includes 17,426 tackle-label rows. This is new tackling evidence from the 2022 season, not an additional season beyond that coverage.

The downloaded ZIP SHA-256 is `5813b8d000b9e3162c7146dffd2055f11a552b8398ea58a7fb439355c37625c2`. Its GitHub release metadata and download receipt are retained under `data/.runtime/tracking-mirrors-20261002/`; CSV bytes and parsed tables are sealed in existing raw/enriched/gold layers. The source registry explicitly identifies SumerSports as the republisher. Official archive byte equivalence has not been established because the original Kaggle archive is unavailable to this account. No model or script from the external repository was executed.

Validation: all source layers verified on load; duplicate frame keys and canonical schema checks passed. The focused suite passed 25 tests, and Ruff passed. StatsBomb remains inventoried, not ingested; the other missing competition archives remain unresolved.
