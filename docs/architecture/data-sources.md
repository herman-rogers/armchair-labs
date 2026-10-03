# NFL tracking, video, and supporting data sources

Inventory researched and first ingested October 2, 2026. This covers official NFL/Kaggle archives, nflverse and its wrappers, public research datasets, video catalogs, and commercial tracking providers. It is not proof that every historical competition file remains downloadable. The [tracking runbook](../operations/tracking-data.md) describes the implemented pipeline; the [initial capture report](../research/tracking_source_inventory_2026-10-02.md) records actual local coverage.

**Latest retry:** 2020 and both 2026 tracks are now downloaded and verified. BDB 2021–2025 still return `RulesAcceptanceRequired`. All 36 analytics-track training CSVs exactly duplicate prediction-track files; only supplementary context adds new evidence. See the capture report for details.

**Earlier access update:** After account setup, authenticated file listings succeeded. BDB 2024 and 2025 still list only a README; 2026 prediction downloaded and was ingested. BDB 2020–2023 and 2026 analytics list data, but their download attempts returned HTTP 403. The 2026 listings contain 2023 Weeks 1–18. See the [authenticated retry results](../research/tracking_source_inventory_2026-10-02.md#authenticated-kaggle-retry) for measured file counts. Earlier access descriptions below record the initial capture.

## Additional public sources found

The subsequent [acquisition plan and search](../plans/tracking_data_review_and_acquisition_2026-10-02.md) recovered the 2024 tackling archive from SumerSports public GitHub releases and ingested it as a separately attributed mirror. It also identified [StatsBomb American Football Open Data](https://github.com/hudl/amf-open-data), whose provider game index lists 124 games across 2016–2022. This corrects the earlier search conclusion below: an additional provider-published NFL tracking corpus does exist. StatsBomb payloads are not yet ingested; its research-only agreement and distinct schema need their own integration.

## What exists

The NFL runs the annual **Big Data Bowl**; the 2026 edition is its eighth. The releases are selected research samples with different tasks and seasons, not an annual dump of every NFL game. There is no verified public directory containing all NFL videos plus synchronized tracking. NFL+ film access, competition clips, tracking coordinates, and publicly reported Next Gen Stats are separate products. [NFL Big Data Bowl](https://operations.nfl.com/programs-initiatives/innovation/big-data-bowl), [NFL Coaches Film](https://support.nfl.com/hc/en-us/articles/35869702764052-Will-Coaches-Film-be-available-with-NFL).

### Spatial tracking archives

Coverage below describes the provider's research task; only the downloaded sample has been measured locally. Competition year is **not** season year. Links point to official sources, not republished mirrors.

| Registry ID | Source and task | Expected coverage / limits | Local access status |
|---|---|---|---|
| `bdb_2019_sample` | [NFL Football Operations GitHub](https://github.com/nfl-football-ops/Big-Data-Bowl) | One 2017 game of frame tracking. Games/plays metadata covers much more than its coordinate file. The original 2019 competition covered a larger sample; do not equate this repository with that complete archive. | Downloaded, normalized, verified |
| `bdb_2020` | [2020 rushing prediction](https://www.kaggle.com/competitions/nfl-big-data-bowl-2020/data) | 2017–19 handoff snapshots (verified current archive); cannot reconstruct routes or acceleration histories from one snapshot | Kaggle access needed |
| `bdb_2021` | [2021 pass defense](https://www.kaggle.com/competitions/nfl-big-data-bowl-2021/data) | 2018 passing sample; selected positions, not guaranteed all 22 players | Kaggle access needed |
| `bdb_2022` | [2022 special teams](https://www.kaggle.com/competitions/nfl-big-data-bowl-2022/data) | 2018–20 special teams; not ordinary offensive snaps | Kaggle access needed |
| `bdb_2023` | [2023 pass protection](https://www.kaggle.com/competitions/nfl-big-data-bowl-2023/data) | 2021 Weeks 1–8 dropbacks, snap through pass release; scouting annotations | Kaggle access needed |
| `bdb_2024` | [2024 tackling](https://www.kaggle.com/competitions/nfl-big-data-bowl-2024/data) | 2022 Weeks 1–9 tackling sample | Kaggle access needed |
| `bdb_2025` | [2025 pre-snap behavior](https://www.kaggle.com/competitions/nfl-big-data-bowl-2025/data) | Frame types distinguish before/after snap. Actual season/week inventory remains unverified until files are retrieved. | Kaggle access needed |
| `bdb_2026_prediction` | [2026 prediction](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction/data) | 2023–24 training described by NFL; predict movement while ball is airborne. Input and output frame clocks restart independently. | Kaggle access needed |
| `bdb_2026_analytics` | [2026 analytics](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-analytics/data) | Companion analysis track; likely overlapping evidence must be checked, never counted as another independent season | Kaggle access needed |

The NFL describes the 2026 evaluation window as 2025 Weeks 14–18; this does not establish that hidden evaluation coordinates are now public. The competition data pages remain discoverable. Unauthenticated API checks for 2021, 2025 and 2026 prediction returned HTTP 401; the 2025 web listing exposed a README-only view. Neither observation proves the archive has been deleted. Credentials, account eligibility, accepted competition rules, and a successful authenticated file listing are still required to establish availability and completeness. [NFL competition overview](https://operations.nfl.com/programs-initiatives/innovation/big-data-bowl).

### Video and safety competition archives

| Source | Potential use | Limits and pipeline support |
|---|---|---|
| [NFL Impact Detection](https://www.kaggle.com/competitions/nfl-impact-detection/data) | Paired camera clips, impact labels, sensor trajectories; video/tracking alignment research | Selected safety events, not representative games. Tabular companion import only; no video decoding pipeline. |
| [NFL Helmet Assignment](https://www.kaggle.com/competitions/nfl-health-and-safety-helmet-assignment/data) | Associate helmets across camera views with player tracking identities | Selected clips and labeled images; mock/public-test data must not be mistaken for independent games. Tabular companion import only. |
| [NFL Player Contact Detection](https://www.kaggle.com/competitions/nfl-player-contact-detection/data) | Contact labels, video and tracking context | Contact-biased sample; preserve clip/frame alignment and label provenance. Tabular companion import only. |
| [NFL+ Coaches Film](https://support.nfl.com/hc/en-us/articles/35869702764052-Will-Coaches-Film-be-available-with-NFL), [NFL Pro](https://support.nfl.com/hc/en-us/articles/35869706651156-NFL-Pro-FAQ-s-and-Answers) | Human film review and searchable cutups linked to analysis | Subscriber viewing is not a verified bulk download or machine-learning redistribution entitlement. No public full-history video API or downloadable corpus established. |

The impact and helmet competition listings describe archives on the order of 3.3–3.4 GB each, but authenticated inventory must establish actual current files. Sensor frames and camera frames have different clocks (approximately 10 Hz versus 59.94 fps in the impact archive); never join merely on frame number. Highlight videos and broadcast replays are also unsuitable substitutes for continuous all-22 tracking because of cuts, replays, occlusion and incomplete field visibility.

### nflverse and complementary public data

| Source | What it adds beside existing datasets | What it is not |
|---|---|---|
| [Next Gen Stats via nflverse](https://nflreadr.nflverse.com/reference/load_nextgen_stats.html) | Passing, rushing and receiving aggregates from 2016; weekly rows and week-0 season aggregates | No player-by-frame x/y archive; thresholds omit low-volume players |
| [Participation](https://nflreadr.nflverse.com/reference/load_participation.html) | Personnel and players on plays; NGS through 2022, FTN from 2023; FTN release after postseason | Not live coordinates or automatically available before each historical game |
| [FTN charting](https://nflreadr.nflverse.com/reference/load_ftn_charting.html) | From 2022: manually charted play attributes, such as motion/play action/RPO/throw context | Not exhaustive trajectory tracking; some fields encode outcomes |
| [Rosters](https://nflreadr.nflverse.com/reference/load_rosters.html), [players](https://github.com/nflverse/nflverse-data/releases/tag/players) | GSIS and provider IDs plus historical roster context | Current team/status is not evidence of historical availability; similar ID column names need explicit namespaces |
| [nflverse data releases](https://github.com/nflverse/nflverse-data/releases) | Existing play-by-play, schedules, stats and roster inputs provide join context | A GitHub release timestamp is not the first publication time of each row |
| [SportsDataverse](https://www.sportsdataverse.org/snippets/sdv-py-nfl-pbp), [status](https://www.sportsdataverse.org/status) | Alternative loading/access and ESPN event/participant information | Its nflverse loader accesses the same upstream evidence, not another tracking corpus |
| [Cross-platform fantasy IDs](https://nflreadr.nflverse.com/reference/load_ff_playerids.html) | Useful for league/platform joins | Not proof of a historical tracking-ID bridge |

The existing repo already has nflverse loading helpers and a profile layer using season summaries. These new captures preserve weekly NGS and supporting play data as separate source releases; they do not silently replace the app's serving tables or make them model-approved features.

Generic nflverse data uses CC-BY-4.0, with source-specific exceptions. Participation and FTN charting require CC-BY-SA-4.0 and provider attribution. Preserve those distinctions in manifests and derived distributions. Competition datasets retain their own rules; public access does not establish unrestricted redistribution. [nflverse repository](https://github.com/nflverse/nflverse-data), [participation documentation](https://nflreadr.nflverse.com/reference/load_participation.html), [charting documentation](https://nflreadr.nflverse.com/reference/load_ftn_charting.html).

### Commercial and research leads checked

- [SkillCorner American football](https://skillcorner.com/sports/american-football) offers video-derived tracking and provider ID integration. Its [open-data repository](https://github.com/SkillCorner/opendata) contains soccer games, not an open NFL archive. Optical or inferred coordinates need their own quality/provenance fields.
- [Genius Sports](https://www.geniussports.com/engage/official-sports-data-api/) is a commercial route to official sports data; no open continuous NFL tracking feed was established. [NFL distribution announcement](https://www.geniussports.com/newsroom/national-football-league-taps-genius-sports-group-as-exclusive-distributor-of-official-league-data/).
- [Zebra](https://www.zebra.com/us/en/cpn/measuring-greatness.html) describes the sensor infrastructure; it is not a public download endpoint.
- [SumerSports](https://sumersports.com/the-zone/how-sumerlive-tracks-the-game/) describes tracking analysis using licensed data, not a freely downloadable corpus.
- [PFF play-data fields](https://media.pff.com/2018/08/PFF-Play-Data-Fields.pdf) describe charting products. Competition-provided annotations do not grant access to all commercial PFF history.
- Research such as [next-gen-scraPy](https://arxiv.org/abs/1906.03339), [route classification](https://arxiv.org/abs/1908.02423), and [NFL Ghosts](https://arxiv.org/abs/2406.17220) supplies methods or limited derived data; it does not establish a complete public tracking archive.

Searches across official NFL, Kaggle, nflverse, SportsDataverse, GitHub, dataset mirrors and research literature did not establish another authoritative open full-season NFL coordinate source. Mirrors may be useful discovery leads but were not used to evade source access requirements. This is a search result as of the inventory date, not a claim that no other dataset exists.

## How these sources fit our pipeline

Implemented in [tracking.py](../../src/engine/data/tracking.py) and [tracking_sources.py](../../src/engine/data/tracking_sources.py):

1. Preserve provider bytes in existing content-addressed `data/raw/objects`, with a source-scoped raw snapshot, acquisition metadata, hashes and coverage description.
2. Parse every source CSV/Parquet into enriched Parquet, preserving provider columns. This is where extra competition fields and labels remain inspectable.
3. Produce validated gold research observations: canonical tracking partitions, separate NGS weekly/season tables, source-native tabular context, and an identity crosswalk.
4. Select each source independently through `data/source_catalog.json`. The existing `data/current.json` serving selection remains independent.

The frame key is `(dataset_id, split, phase, game_id, play_id, frame_id, entity_id)`. Output labels have a separate role and are excluded by the default tracking reader. Different archives can overlap the same NFL play; dataset identity prevents accidental collisions but does **not** deduplicate overlapping games for model evaluation. Explicitly partition training/test rows and deduplicate by verified game/play identities before training.

Coordinates retain provider yards, missing values and out-of-field values. Attack-right coordinates are added only when play direction is supplied. Missing players never become footballs; missing coordinates are flagged rather than imputed. Handoff records use a snapshot phase, not a fabricated sequence. Prediction input/output clocks stay separate. Source adapters are tested with fixtures; full competition compatibility and coverage remain pending real authenticated archives.

Modern tracking IDs can join to roster `gsis_it_id`, then GSIS `gsis_id`. The **2019 sample and 2020 archive use legacy IDs**: Tom Brady is `2504211`, whereas current nflverse/rosters use GSIS IT `25511`. Its 89 observed player IDs remain unmapped. No fuzzy name matching or arithmetic ID conversion is applied. Legacy identity resolution is a distinct follow-up requiring a verified bridge.

Game IDs require an explicit schedule bridge: competition numeric game IDs correspond to old-style game IDs, whereas common nflverse play keys use season/week/team strings. Validate season, teams, and play IDs before joining; never cast one string format into the other. A full canonical game bridge and derived tracking features are not implemented in this ingestion layer.

## Research sequence and remaining access work

Start with verified frames and archive-specific exploratory quality reports. Next, retrieve authenticated BDB archives, measure actual seasons/weeks/plays/players/events, and validate their identity and game bridges. Prioritize pre-snap alignment/motion, routes and separation, defensive spacing, and rushing geometry where the relevant archive actually contains the required positions and events. Add film synchronization separately if needed.

Before using these as forecast inputs, distinguish descriptive post-play outcomes from information available at prediction time. Acquisition now does not establish historical publication availability. Keep holdouts by game and time, exclude future frames/labels, and test incremental value against existing play-by-play, usage, participation and NGS baselines. Archived tracks can support representation research; they do not by themselves support a live player-tracking feature in 2026.
