# Observation tables: generation, tables, and GCS

This is the supported workflow for producing and sharing queryable observation data.
New accepted datasets belong in `src/engine/tables/registry.yaml`. Use the existing
builders, validation, cache, and transport instead of adding another data loader or
upload path. Model outputs never enter this catalog: they live in campaign bundles
and are served by `/api/chairstats`.

## Use the data

```sh
uv sync --extra shared-data
just data-fetch                  # selects data/releases/tables/current.json
just tables list
just tables query "SELECT team, season, count(*) AS appearances FROM analytics.team_player_games GROUP BY team, season ORDER BY season DESC, team"
```

The small, shareable `data/releases/tables/current.json` pins the last successfully
published selection. Distribute that file to consumers when releasing new data.
It points at an immutable, generation-pinned GCS release; it is not a remote
“latest” lookup. Consumers stay on their chosen release until they fetch another
reference. Explicit versions work too:

```sh
just tables fetch data/releases/tables/RELEASE.json  # choose a pinned historical release
just tables fetch data/releases/tables/correlation_tables_20261003_r2.json --research  # saved study
just tables fetch data/releases/tables/current.json --offline  # use verified local cache
```

Normal SQL sessions open actual native DuckDB tables, read-only. The first query
builds a local database from the selected Parquet releases. Later queries reuse it;
a new catalog gets a new database, so existing readers stay pinned. `just tables
cache` prewarms it. The native database is disposable and is not uploaded to GCS.
Use `--parquet`, or Python `connect(native=False)`, for direct Parquet queries.
Queries do not contact GCS. Connections use UTC, four query threads and a 1 GB
DuckDB memory limit; ingestion uses one writer thread and checkpoints each table
to accommodate wide feature tables. The engine limit excludes Python allocations.

```python
from engine.tables import connect

with connect() as db:
    games = db.execute(
        "SELECT * FROM analytics.team_player_games WHERE team = ? AND season >= ?",
        ["LA", 2021],
    ).pl()
```

`just tables describe NAME` shows schema, keys, lineage, cutoff, and limitations.
`just tables query @my_query.sql --output data/outputs/result.parquet` exports a
result; CSV is also supported. Existing output paths are protected. SQL is trusted
local analyst code, not an untrusted public SQL endpoint.

## Frontend serving coverage

The frontend calls FastAPI; DuckDB runs in the backend. The frontend's TanStack
Query cache manages HTTP responses and is separate from the DuckDB query engine.
Every request that reads observations runs inside one pinned catalog scope.

| Page or data | Backend read path |
|---|---|
| ChairStats rankings, QB production, player forecasts | Campaign bundles via `/api/chairstats`; the request names the campaign |
| Team analysis | DuckDB `analytics.team_player_games`, followed by statistical calculations |
| Player profiles, player directory, rookies, college history | DuckDB observation tables via `/api/tables` |
| League ownership, teams, lineups, transactions, draft and results | Captured ESPN observations → local Parquet catalog → native DuckDB → league calculations |
| League forecasts and Team strength | ESPN observations joined with the named campaign's per-game points and team correlation tables |

The boundary is `engine.tables.application.query_scope`, enabled by the API for
table-reading routes. It pins the published query catalog for the request and fails
closed when the catalog changes mid-request. Successful observation responses carry
`X-Data-Engine: duckdb`.

League captures live under `data/league/<league-id>/<season>/data` and use the same
`tables.build`, verification, and native cache implementation. Typed tables include
`league_players`, `league_teams`, `league_schedule`, `league_lineup_entries`,
`league_transactions`, `league_draft`, `league_week_lineups` and `league_snapshot`.
Nested records are also preserved losslessly. ESPN's existing TTL, refresh lock,
and stale-if-error behavior still apply; capture failure does not silently bypass
DuckDB. Pregame logs and reviewed news use their own capture catalogs. These local
league catalogs are not part of the shared GCS analytics release.

```sh
just tables query --data-dir data/league/LEAGUE_ID/2026/data \
  "SELECT owner_team_id, count(*) FROM analytics.league_players GROUP BY owner_team_id"
```

`just data-fetch` installs the shared query catalog for SQL consumers. Running the
full API also requires its original verification/policy artifacts and local ESPN
observations; a table-only replica is sufficient for direct SQL and Team analysis.

## Generate → build → upload

```mermaid
flowchart LR
    A[Capture sources] --> B[Validate observation batch]
    B --> E[Build registered tables and dependents]
    E --> F[Atomically select table catalog]
    F --> G[Upload immutable Parquet and manifests]
    G --> H[Publish consumer catalog and handoff reference]
    H --> I[Consumer fetch and DuckDB queries]
```

On the designated publisher with source archives and GCS credentials:

```sh
just data-refresh --upload
# Same command without Just:
uv run --extra shared-data engine data refresh --upload

just data-refresh --status
just data-refresh --version weekly_unique_id --upload  # resume this run after failure
```

This captures the next completed regular-season week, builds a validated observation
batch, then builds all refreshable registered tables and uploads the resulting query
catalog. An unchanged week still checks recipes and retries any unfinished upload.
Frozen research imports remain manual. `--force` incorporates reviewed same-week
source corrections. The weekly cadence is explained in [refresh data](refresh-data.md).

For table development using already captured sources, without source acquisition
or forecast/model rebuilds:

```sh
just tables refresh team_player_games --upload  # dependencies and affected dependents
just tables refresh --due --upload             # due daily/weekly recipes
just tables refresh                            # explicitly check every recipe, including frozen imports
just tables verify
```

Both commands call `engine.tables.workflow.refresh_and_publish`; there is one
implementation of table publication and retry. `--upload` is explicit; omit it
for local development. `--store file:///absolute/path/store` exercises the same
contract offline. GCS defaults to `gs://armchair-labs-data/armchair-labs/tables`.
Use ADC/service identity or an existing gcloud login. Consumers need read access
to the objects and manifests; uploading does not make private data public.

| Stage | Implementation | Result |
|---|---|---|
| Production orchestration | `engine.data.pipeline`, `engine.data.refresh` | Capture, register, enrich, build the observation batch, refresh/upload tables |
| Enrichment | `engine.data.enrich` (`sql/*.sql`) | DuckDB recipes over preserved nflverse files: positions, league scoring, identities, college |
| Source contracts | `engine.tables.contracts` | Keys, finite quantities, per-table specs |
| Table recipes | `engine.tables.registry`, `registry.yaml`, `builders.py` | Declared keys, dependencies, cadence, code |
| Application reads | `engine.tables.application`, `engine.tables.observations` | Request-scoped pinned catalog and the observation SQL the API uses |
| League captures | `engine.tables.league` | Typed local Parquet catalogs and native DuckDB queries using the same build/cache code |
| Table building | `engine.tables.build` | Versioned Parquet, schema, null counts, code snapshots, lineage |
| Publication/retry | `engine.tables.workflow`, `transport` | Verified objects, completion manifest, consumer catalog, reference |
| Byte transfer | `engine.data.shared`, `shared_gcs` | Content-addressed objects, deduplication, checksums, pinned generations |
| Local query/cache | `engine.tables.query` | Read-only DuckDB snapshot of a selected catalog |

`engine clear-cache` removes the nflverse download cache; it does not refresh
published data.

### Publication and failure rules

- A table build validates keys, required types, row counts and finite quantities.
  Schema, null counts, exact input versions and observation cutoffs are retained.
- Source tables import as a unit. Changed dependencies rebuild their dependents;
  mixed dependency versions cannot be selected together.
- All builds finish before one atomic table-catalog switch. Failed builds leave
  the previous complete catalog selected. Unchanged fingerprints are a no-op.
- Uploads write immutable content-addressed objects first, then the completion
  manifest, then the consumer catalog. Only after success is the small local
  handoff reference updated. Upload failure retains validated local tables and
  the previous published reference. Rerun the same command to retry.
- Batch building and table publication are separate atomic steps. If table
  building fails after a batch is built, resume the run or execute
  `just tables refresh team_player_games --upload`.
- `--due` measures elapsed hours since a successful build. Scheduling does not
  create new source observations: the full data refresh captures them first.

`deploy/com.engine.data.plist` is the single production launchd template. Edit its
paths and install it on the publisher if scheduling is wanted. It runs `engine
data refresh --upload`. No scheduler is installed automatically. Equivalent cron:

```cron
15 6 * * * cd /absolute/path/fantasy && /absolute/path/uv run --extra shared-data engine data refresh --upload >> /absolute/path/data-refresh.log 2>&1
```

Advanced bootstrap/staged builds use `engine data build`, `enrich`, `tables`, and
`verify`; inspect each command's `--help`. `build` stages an immutable observation
batch; the latest accepted batch is the source for registered `kind: source` tables.
After a manual build, run the table refresh/upload command above. The normal
`data refresh` command performs these steps in order.

## Add a table

For a refreshable finding, add a SQL file next to the registry and declare a recipe:

```sql
-- src/engine/tables/sql/team_season_points.sql
SELECT team, season, count(DISTINCT game_id) AS games,
       sum(league_points) AS qb_wr_te_points
FROM analytics.team_player_games
GROUP BY team, season
```

```yaml
# Under tables: in src/engine/tables/registry.yaml
team_season_points:
  description: Observed QB/WR/TE points by team and season.
  grain: One team per season.
  primary_key: [team, season]
  kind: sql
  sql_file: sql/team_season_points.sql
  dependencies: [team_player_games]
  refresh_hours: 168
  required_columns: {team: String}
```

Then run `just tables refresh team_season_points --upload`. SQL recipes contain one
SELECT (including WITH), over declared dependencies. Python recipes specify
`builder: package.module:function`, receive `BuildContext`, and return a Polars
DataFrame. Read inputs using `context.read(name)`; declare every dependency and
helper module (`code_modules`) or data file (`inputs`) affecting the output.
`kind: source` imports a named validated source-batch table into the same catalog.
Whole-table replacements are intentional; distributed writers and incremental
partition updates are not implemented.

Keep exploratory work under `data/research/`. To retain an accepted study:

```sh
just tables archive-research data/outputs/my_study my_study_001
```

This retains original files unchanged and converts CSV tables to Parquet. Promote
one output with `kind: parquet`, `source: research/my_study_001/tables/RESULT.parquet`,
explicit keys, cutoff and limitations, and the archive manifest in `inputs`.
Omit `refresh_hours` for a frozen study. Include the archive in a distribution with
`just tables publish UNIQUE_RELEASE --research my_study_001`; query consumers need
only the registered tables, not the research profile.

## Consume from another system

Each reference includes `consumer_catalog`: its GCS URI, SHA-256, byte size and
object generation. That JSON is the language-neutral integration contract; it
lists every table's description, row count, columns, primary key, cutoff, source
reference and limitations. Each table's `parquet` entry supplies the exact object
URI, size, SHA-256 and generation. Column type strings describe the producer's
schema; Parquet carries its portable physical/logical types.

The current release and exact consumer-catalog URI are in
[`data/releases/tables/current.json`](../../data/releases/tables/current.json).
Always distribute that pinned reference; historical release references remain valid.

Systems using this repository can fetch the reference and use `engine.tables`.
Other systems can use only DuckDB and Google's storage SDK (with ADC configured):

```python
import hashlib
import json
from pathlib import Path
import duckdb
from google.cloud import storage

client = storage.Client()
reference = json.loads(Path("current.json").read_text())  # supplied by publisher

def download(spec):
    bucket, key = spec["uri"].removeprefix("gs://").split("/", 1)
    blob = client.bucket(bucket).blob(key, generation=int(spec["generation"]))
    payload = blob.download_as_bytes()
    if len(payload) != spec["size"] or hashlib.sha256(payload).hexdigest() != spec["sha256"]:
        raise ValueError("Published object failed verification")
    return payload

catalog = json.loads(download(reference["consumer_catalog"]))
table = catalog["tables"]["team_player_games"]
Path("team_player_games.parquet").write_bytes(download(table["parquet"]))
with duckdb.connect() as db:
    db.execute("CREATE TABLE team_player_games AS SELECT * FROM read_parquet('team_player_games.parquet')")
    print(db.sql("SELECT team, count(*) FROM team_player_games GROUP BY team").fetchall())
```

Retain the reference alongside any derived analysis to pin its inputs. GCS objects
are not named by table: resolve them through the catalog rather than listing a
prefix or guessing filenames. DuckDB caches are rebuilt locally, never shared as
writable database files. Research/source archives use the same underlying byte
transport but are an explicit separate profile (`just source-fetch`), not a second
way to select queryable tables. See [shared research inputs](shared-data.md).

## Inventory, storage, and Team analysis

The catalog contains the cleaned observation source tables, two explicit postgame
schedules, and the analysis tables. Local league catalogs are separate. Run
`just tables list` for the full schema inventory. Representative sources are
`players`, `nfl_player_weeks`, `current_weeks`, `nfl_schedule` and `college_annual`.

| Analytical table | Contents |
|---|---|
| `analytics.team_player_games` | 59,400 participation-backed QB/WR/TE player-games, all teams, 2013–2026 Week 3 |
| `analytics.player_pair_correlations` | 64,682 season/pooled pair results; filter scope, sample and basis |
| `analytics.rams_pair_correlations` | 100 frozen Stafford-start study pairs, with permutation/BH diagnostics |
| `analytics.rams_pair_correlations_by_season` | 214 frozen study season/pair rows |

The original study and the all-QB refreshable table use different populations;
results are not interchangeable. `starter` receivers require at least 50% offensive
snaps; `active` uses positive snaps for eligible players. Missing appearances stay
absent; verified positive-snap zero-stat games score zero. Season centering is
within each pair's shared games. Undefined correlations stay null; Fisher intervals
are exploratory and unadjusted. Pairwise covariances from different samples do not
form a coherent lineup covariance matrix. The frozen study's exact methodology
remains in `data/research/rams_correlations_20261003/originals/methodology.json`.

Team analysis queries `team_player_games` with parameterized team/year filters,
then applies its existing QB/participation and complete-case lineup statistics.
Its response includes the exact table version and observation coverage. The browser
requests analysis directly and polls it every minute, without a catalog round trip.
The catalog endpoint and optional `table_version` parameter remain available to
clients that explicitly pin a version. Missing,
corrupt or source-mismatched tables fail closed; superseded table tokens return
409. Query-only replicas need no source archives or application catalog.

The API preloads the current native database and team coverage at startup. Native
snapshot setup and coverage validation are cached per catalog; subsequent queries
use lightweight file revision checks rather than reloading all manifests, hashing
the database, or rescanning coverage. A changed publication selects a new snapshot;
changed files invalidate the cached verification. New snapshots load on first use
if published while the server is running. Failed validation is never cached.

`nfl_schedule`/`current_schedule` are forecast-safe schedules; the separate
`*_observed_schedule` tables retain realized QB identities and scores. Keep
`season_outcomes` separate from `preseason_features` when forecasting. A table is
queryable evidence, not automatic approval to use future information as a feature.

```text
data/raw/, data/enriched/             # captured bytes and reproducible build inputs
data/tables/batches/<version>/        # validated source-builder batches and quality reports
data/tables/releases/<name>/<version>/# accepted Parquet + manifest + recipe source
data/tables/catalogs/<hash>.json      # immutable consistent query selection
data/tables/current.json             # atomic local query-catalog pointer
data/cache/tables/*.duckdb            # disposable native query database
data/league/<league-id>/<season>/data/ # separate local league table catalog/cache
data/research/<run>/                 # retained studies; frozen imports by explicit recipe
data/releases/tables/<release>.json  # immutable distribution reference
data/releases/tables/current.json    # shareable reference to selected published tables
```

The table pointer selects queryable datasets; campaign bundles under
`data/campaigns/` hold model outputs and are read directly by `/api/chairstats`.
Old `data/gold/` batches remain readable as accepted batches; there is no gold
query layer. Never rewrite immutable historical manifests or move their files.

Validation: `uv run pytest tests/test_tables.py tests/test_observations.py tests/test_team_correlations.py`. The suite checks
source verification, atomic builds, dependency freshness, upload failure/retry,
consumer inventories, fresh/offline fetch, application SQL/research isolation,
league capture round trips, and Team analysis behavior.
