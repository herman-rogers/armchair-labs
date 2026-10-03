# Capture and use NFL tracking sources

Use the existing Python environment from the repository root. Source definitions are in [`tracking_sources.py`](../../src/engine/data/tracking_sources.py); coverage and access limitations are in the [source inventory](../architecture/data-sources.md).

```sh
.venv/bin/python -m engine.data.tracking inventory
.venv/bin/python -m engine.data.tracking status
.venv/bin/python -m engine.data.tracking ingest bdb_2019_sample --version tracking_sample_YYYYMMDD_r1
.venv/bin/python -m engine.data.tracking ingest nflverse_ngs --version tracking_ngs_YYYYMMDD_r1
```

Other directly downloadable IDs: `nflverse_rosters`, `nflverse_players`, `nflverse_participation`, `nflverse_ftn`. Replace version placeholders with a new immutable identifier for each acquisition. `--data-dir PATH` goes before the subcommand when using another data root.

Public acquisition pins the GitHub source revision/asset metadata and hashes downloaded bytes. Complete files with matching receipts can be reused after an interrupted download; partial files are not trusted. A failed normalization leaves evidence but does not select a source release. Use a fresh version after a partial enriched/gold build. Repeating a sealed successful version verifies and reselects that existing snapshot; it does not refresh provider data.

## Kaggle archives

The follow-up environment check installed Kaggle CLI 2.2.4 with `uv tool install kaggle` at `~/.local/bin/kaggle`. The initial check required authentication. After account setup, file listings succeeded, and the 2026 prediction archive downloaded successfully. BDB 2020–2023 and 2026 analytics downloads returned HTTP 403; check competition participation/rules and account download access before retrying. Run `kaggle auth login` to complete the browser sign-in, then accept the competition's rules in the account that will download it. On another machine, install the official CLI first. Do not put tokens in documentation or shell history. [Official client documentation](https://github.com/Kaggle/kaggle-cli).

```sh
uv tool install kaggle
kaggle competitions files -c nfl-big-data-bowl-2025
.venv/bin/python -m engine.data.tracking ingest bdb_2025 --version tracking_bdb2025_YYYYMMDD_r1 --kaggle
```

The CLI downloads the competition archive, extracts it with path/size checks, and imports CSV/Parquet companions. Alternatively, import an already authorized, extracted directory:

```sh
.venv/bin/python -m engine.data.tracking ingest bdb_2025 --version tracking_bdb2025_YYYYMMDD_r1 --local /absolute/path/to/extracted/archive
```

Keep one competition vintage per input directory. The importer rejects duplicate frame keys across files and metadata-only BDB packages. Local imports preserve the bytes and input origin; they cannot independently prove those files constitute the full official archive. For safety/video competitions, prefer a directory of tabular companions: `--kaggle` downloads the entire available archive, including videos, although only tabular files are ingested. There is no video decoding or sensor-camera alignment implementation yet.

CSV ingestion currently holds one source file in memory; large weekly tracking files may require substantial RAM. Reads across normalized partitions are lazy. ZIP expansion is capped at 50 GB; larger archives need a deliberate implementation change or an authorized extracted directory.

## Read verified observations

```python
from pathlib import Path
from engine.data.tracking import load_source, scan_tracking, link_tracking_players

root = Path("data")
release = load_source(root, "bdb_2019_sample")
frames = scan_tracking(release)  # prediction output labels excluded
rosters = load_source(root, "nflverse_rosters")
linked = link_tracking_players(frames, rosters)
print(linked.group_by("mapping_status").len().collect())

ngs = load_source(root, "nflverse_ngs")
weekly_passing = ngs.read("ngs_passing_weekly")
season_passing = ngs.read("ngs_passing_season")
```

Legacy sample IDs intentionally remain unmapped. Modern competition rows use the GSIS IT namespace; ambiguous provider crosswalks return a null `player_id`. The reader includes observation rows from both provider train and test partitions; filter `split` explicitly for each experiment. Use `include_labels=True` only for a deliberate evaluation/label path. Other metadata tables may also contain outcomes: exclusion of output-frame tables alone does not make a feature set leakage-safe.

Inspect `release.manifest['tables']` for table names, roles, row counts, schemas, missingness and available coverage. Original extra provider fields remain in enriched tables even when they are not canonical gold frame columns.

```sh
.venv/bin/python -m engine.data.tracking verify --version YOUR_SELECTED_VERSION
.venv/bin/pytest tests/test_tracking_sources.py tests/test_data_pipeline.py
```

Verification follows hash-bound raw → enriched → gold dependencies. Successful source ingestion atomically updates `data/source_catalog.json`; it does not publish a serving release or start model training. All generated data and the local source catalog are ignored by Git.

## Storage and sharing

Reuse the repository's raw/enriched/gold layout. No migration of existing analytical tables is necessary. The new source catalog is an auxiliary selection index, not a replacement for the application catalog. Source version identifiers can also be loaded explicitly with the existing `load_gold` API.

Existing shared-data profiles include broad raw/enriched/gold trees, so future snapshots can include these files. Review dataset terms before distributing a snapshot; no upload was performed by this work. The default shared profiles do not currently carry `data/source_catalog.json`: add that explicit selection file to a deliberate source-sharing profile or reselect verified versions locally. Copying arbitrary source Parquet files without their manifest dependencies loses verification.

## Resolving competition access

The follow-up SDK probe inspected Kaggle's error response rather than only the CLI's generic 403. BDB 2020–2025 and 2026 analytics returned `RulesAcceptanceRequired`: “You must accept this competition's rules before you'll be able to download files.” Authentication is working; each competition has a separate acceptance requirement.

Sign into the same Kaggle account used by the CLI and review/accept each required competition's rules:

- [2020 rushing](https://www.kaggle.com/competitions/nfl-big-data-bowl-2020/rules)
- [2021 passing](https://www.kaggle.com/competitions/nfl-big-data-bowl-2021/rules)
- [2022 special teams](https://www.kaggle.com/competitions/nfl-big-data-bowl-2022/rules)
- [2023 pass protection](https://www.kaggle.com/competitions/nfl-big-data-bowl-2023/rules)
- [2026 analytics](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-analytics/rules)

The CLI has no competition-rules acceptance command. If the website requests account verification, complete that flow there. Retry a small file before a full archive download. If acceptance is unavailable for an archived competition, ask [Kaggle Support](https://www.kaggle.com/contact) whether new accounts can obtain research access; no support message has been sent.

2024 and 2025 have an additional availability question: their current listings contain only a 78-byte README. Accepting [2024 rules](https://www.kaggle.com/competitions/nfl-big-data-bowl-2024/rules) and [2025 rules](https://www.kaggle.com/competitions/nfl-big-data-bowl-2025/rules) allows us to retry the README and verify its explanation; it does not guarantee the coordinate files return. [SumerSports' research repository](https://github.com/SumerSports/SportsTrackingTransformer) reports that the host removed the original 2024 dataset. No equally definitive official removal explanation for 2025 has been established. Request an authorized archive from the competition host if those files remain absent.

The [2024 rules](https://www.kaggle.com/competitions/nfl-big-data-bowl-2024/rules) specify noncommercial data use and restrictions on sharing with nonparticipants. Research access should not be treated as approval to redistribute raw files or use them in a commercial product; resolve those rights with the provider before such use.

2020 access was subsequently unlocked by user rules acceptance and its archive ingested successfully. The remaining rules-gated competitions still require their own acceptance. Current 2020 training includes 2017–19 and uses legacy player IDs; see the capture report for measured coverage.

Latest retry: 2026 analytics also unlocked and was ingested. Its 36 tracking CSVs exactly duplicate prediction-track training files, while supplementary context adds 18,009 rows. BDB 2021–2025 still explicitly require rules acceptance; if the website does not allow it, use the support route above.
