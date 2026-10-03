"""Capture and normalize auxiliary NFL sources in the existing release layers.

These are research observations, never a replacement for the serving catalog.
Run ``python -m engine.data.tracking --help`` for commands.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import re
import shutil
import subprocess
import tempfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from engine.data.releases import (
    atomic_json,
    digest,
    identifier,
    inside,
    load_gold,
    load_manifest,
    preserve_object,
    reference,
    write_json,
)
from engine.data.tracking_sources import SOURCES, Source, public_assets

KEY = ["dataset_id", "split", "phase", "game_id", "play_id", "frame_id", "entity_id"]
ALIASES = {
    "game_id": ["gameId", "GameId", "game_id"],
    "play_id": ["playId", "PlayId", "play_id"],
    "nfl_id": ["nflId", "NflId", "nfl_id"],
    "frame_id": ["frameId", "frame.id", "frame_id"],
    "x": ["x", "X"],
    "y": ["y", "Y"],
    "speed": ["s", "S"],
    "acceleration": ["a", "A"],
    "distance": ["dis", "Dis"],
    "orientation": ["o", "Orientation"],
    "direction": ["dir", "Dir"],
    "team": ["club", "team", "Team", "player_side"],
    "play_direction": ["playDirection", "PlayDirection", "play_direction"],
    "event": ["event"],
    "frame_type": ["frameType"],
    "time": ["time"],
}


def now() -> str:
    return datetime.now(UTC).isoformat()


def http_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers["User-Agent"] = "Armchair Labs-tracking-research/1.0"
    return session


def fetch_asset(session: requests.Session, spec: dict, root: Path) -> Path:
    """Resume only completed, hash-verified files; never trust an existing partial."""
    target = inside(root, spec["name"])
    receipt = target.with_name(target.name + ".receipt.json")
    if receipt.exists() and target.exists():
        prior = json.loads(receipt.read_text())
        if prior["spec"] == spec and prior["sha256"] == digest(target):
            return target
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            with session.get(spec["url"], stream=True, timeout=(20, 120)) as response:
                response.raise_for_status()
                if "text/html" in response.headers.get("Content-Type", ""):
                    raise ValueError("Expected a data file, received HTML")
                for chunk in response.iter_content(1024 * 1024):
                    stream.write(chunk)
            stream.flush()
            if temporary.stat().st_size != spec["size"]:
                raise ValueError(f"Truncated or changed download: {spec['name']}")
            sha = digest(temporary)
            expected = spec.get("provider_digest")
            if expected and expected.startswith("sha256:") and expected != f"sha256:{sha}":
                raise ValueError(f"Provider digest mismatch: {spec['name']}")
            temporary.replace(target)
            atomic_json(receipt, {"spec": spec, "sha256": sha, "captured_at": now()})
        finally:
            temporary.unlink(missing_ok=True)
    return target


def extract_zip(path: Path, destination: Path, max_bytes: int = 50_000_000_000):
    """Bound expansion and reject traversal, symlinks, and ambiguous member names."""
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if sum(m.file_size for m in members) > max_bytes:
            raise ValueError("Archive exceeds extraction byte limit")
        names = [m.filename for m in members]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate archive members")
        for member in members:
            if "\\" in member.filename or ".." in Path(member.filename).parts:
                raise ValueError("Unsafe archive path")
            inside(destination, member.filename)
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Archive symlink is not allowed")
        archive.extractall(destination)


def capture(
    data_dir: Path, source: Source, version: str, local: Path | None = None, *, kaggle: bool = False
) -> dict:
    """Seal raw source bytes. Retry unsealed downloads with the same version."""
    identifier(version)
    root = data_dir / "raw/snapshots" / version
    if (root / "manifest.json").exists():
        _, manifest = load_manifest(data_dir, "raw", reference(root))
        if manifest.get("source", {}).get("id") != source.id:
            raise ValueError("Version belongs to another source")
        return reference(root)
    stage = data_dir / ".runtime/tracking" / version
    stage.mkdir(parents=True, exist_ok=True)
    origin = {
        "source_id": source.id,
        "local": str(local.resolve()) if local else None,
        "kaggle": kaggle,
    }
    origin_path = stage / "origin.json"
    if origin_path.exists() and json.loads(origin_path.read_text()) != origin:
        raise ValueError("Cannot resume version with different inputs")
    atomic_json(origin_path, origin)
    specs = []
    if kaggle:
        if not source.competition:
            raise ValueError("Source is not a Kaggle competition")
        executable = shutil.which("kaggle")
        if not executable:
            raise ValueError("Install the Kaggle CLI and configure credentials; see runbook")
        local = stage / "kaggle"
        local.mkdir(exist_ok=True)
        # The official client enforces account access and competition rules.
        subprocess.run(
            [executable, "competitions", "download", "-c", source.competition, "-p", str(local)],
            check=True,
        )
        for archive in local.glob("*.zip"):
            extract_zip(archive, local / "extracted")
    if local:
        local = local.resolve()
        if not local.is_dir():
            raise ValueError("Local import must be an extracted directory")
        for path in sorted(local.rglob("*")):
            if path.is_symlink():
                raise ValueError("Do not import symlinks")
            if path.is_file() and path.suffix.lower() in {".csv", ".parquet"}:
                specs.append(
                    {
                        "name": path.relative_to(local).as_posix(),
                        "url": source.page,
                        "size": path.stat().st_size,
                        "source_revision": None,
                        "source_updated_at": None,
                        "local_path": str(path),
                    }
                )
    else:
        with http_session() as session:
            plan = stage / "discovery.json"
            if plan.exists():
                specs = json.loads(plan.read_text())
            else:
                specs = public_assets(source, session)
                atomic_json(plan, specs)
            for spec in specs:
                print(f"Download {source.id}: {spec['name']}", flush=True)
                fetch_asset(session, spec, stage / "downloads")
    if not specs:
        raise ValueError("No supported data files found; a README-only archive is not data")
    if source.adapter in {"bdb", "handoff"} and not any(
        is_spatial_file(s["name"], source) for s in specs
    ):
        raise ValueError("No spatial tracking files found; metadata alone is insufficient")
    assets, files = {}, {}
    for spec in specs:
        path = (
            Path(spec["local_path"]) if "local_path" in spec else stage / "downloads" / spec["name"]
        )
        sha = digest(path)
        obj = preserve_object(data_dir, path, sha)
        details = {k: v for k, v in spec.items() if k != "local_path"}
        receipt = stage / "downloads" / (spec["name"] + ".receipt.json")
        captured_at = json.loads(receipt.read_text())["captured_at"] if receipt.exists() else now()
        assets[spec["name"]] = {
            **details,
            "object": obj,
            "sha256": sha,
            "captured_at": captured_at,
            "publication_time_known": False,
        }
        files[obj] = sha
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "version": version,
        "layer": "raw",
        "status": "accepted",
        "kind": "auxiliary_source",
        "source": source.metadata(),
        "registered_at": now(),
        "assets": assets,
        "files": files,
        "scope": "research_observations",
        "capture_mode": "local_import" if local else "http",
    }
    atomic_json(root / "manifest.json", manifest)
    return reference(root)


def is_spatial_file(name: str, source: Source) -> bool:
    base = Path(name).name.lower()
    if source.adapter == "handoff":
        return base in {"train.csv", "test.csv"}
    return source.adapter == "bdb" and (
        base.startswith(("tracking", "input_", "output_"))
        or base == "test_input.csv"
        or bool(re.fullmatch(r"week\d+\.csv", base))
    )


def normalize_tracking(
    frame: pl.DataFrame, dataset: str, name: str, *, handoff: bool = False
) -> pl.DataFrame:
    """Preserve native coordinates; add attack-right coordinates only if known.

    A null NFL ID is a ball only with a provider ball marker. Unknown players do
    not silently become footballs. Output frames have their own clock/phase.
    """

    def value(key, dtype=pl.String):
        col = next((c for c in ALIASES[key] if c in frame.columns), None)
        expr = pl.col(col) if col else pl.lit(None)
        if dtype == pl.String:
            return expr.cast(pl.String).str.replace(r"\.0$", "").alias(key)
        return expr.cast(dtype, strict=True).alias(key)

    required = ["game_id", "play_id", "nfl_id", "x", "y"]
    if not handoff:
        required.append("frame_id")
    for key in required:
        if not any(c in frame.columns for c in ALIASES[key]):
            raise ValueError(f"Missing tracking field {key}: {name}")
    base = Path(name).name.lower()
    split = "test" if "test" in Path(name).parts or base.startswith("test") else "train"
    phase = (
        "handoff"
        if handoff
        else "output"
        if base.startswith("output_")
        else ("input" if base.startswith("input_") or base == "test_input.csv" else "observed")
    )
    numeric = {"x", "y", "speed", "acceleration", "distance", "orientation", "direction"}
    result = frame.select(
        [
            value(k, pl.Float64 if k in numeric else pl.Int64 if k == "frame_id" else pl.String)
            for k in ALIASES
        ]
    ).with_columns(
        pl.lit(dataset).alias("dataset_id"),
        pl.lit("legacy_nfl" if dataset in {"bdb_2019_sample", "bdb_2020"} else "gsis_it").alias(
            "id_namespace"
        ),
        pl.lit(name).alias("source_file"),
        pl.lit(phase).alias("phase"),
        pl.lit(split).alias("split"),
    )
    if handoff:
        result = result.with_columns(pl.lit(0, dtype=pl.Int64).alias("frame_id"))
    marker = pl.col("team").str.to_lowercase().is_in(["ball", "football"]).fill_null(False)
    result = result.with_columns(
        pl.when(pl.col("nfl_id").is_not_null())
        .then(pl.concat_str([pl.lit("nfl:"), "nfl_id"]))
        .when(marker)
        .then(pl.lit("football"))
        .otherwise(None)
        .alias("entity_id"),
        pl.when(marker).then(pl.lit("ball")).otherwise(pl.lit("player")).alias("entity_type"),
        pl.col("play_direction").str.to_lowercase(),
    )
    if result.select(pl.any_horizontal(pl.col(KEY).is_null()).any()).item():
        raise ValueError(f"Unknown tracking identity or null key: {name}")
    if result.select(pl.struct(KEY).is_duplicated().any()).item():
        raise ValueError(f"Duplicate tracking keys: {name}")
    for key in numeric:
        if result.filter(pl.col(key).is_not_null() & ~pl.col(key).is_finite()).height:
            raise ValueError(f"Nonfinite {key}: {name}")
    left, right = pl.col("play_direction") == "left", pl.col("play_direction") == "right"
    return result.with_columns(
        (pl.col("x").is_not_null() & pl.col("y").is_not_null()).alias("coordinates_available"),
        pl.when(left)
        .then(120 - pl.col("x"))
        .when(right)
        .then(pl.col("x"))
        .otherwise(None)
        .alias("x_attack"),
        pl.when(left)
        .then(160 / 3 - pl.col("y"))
        .when(right)
        .then(pl.col("y"))
        .otherwise(None)
        .alias("y_attack"),
        (~pl.col("x").is_between(0, 120) | ~pl.col("y").is_between(0, 160 / 3)).alias(
            "outside_field"
        ),
    )


def read_source(path: Path, name: str) -> pl.DataFrame:
    if name.endswith(".parquet"):
        return pl.read_parquet(path)
    # Historical weather fields mix numbers, ranges and text (including provider
    # entry errors). Preserve them as evidence rather than coercing them to null.
    columns = pl.scan_csv(path, infer_schema=False).collect_schema().names()
    overrides = {c: pl.String for c in ("WindSpeed", "WindDirection") if c in columns}
    return pl.read_csv(
        path,
        infer_schema_length=10000,
        null_values=["NA", "N/A", ""],
        schema_overrides=overrides,
    )


def table_name(name: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_]", "_", str(Path(name).with_suffix("")))
    return clean.lower()


def table_spec(root: Path, name: str, frame: pl.DataFrame, *, role: str) -> dict:
    path = root / f"{name}.parquet"
    frame.write_parquet(path, compression="zstd")
    coverage = {}
    for key in ("season", "week", "phase", "game_id"):
        if key in frame.columns:
            vals = frame[key].drop_nulls().unique().sort()
            coverage[key] = {"distinct": len(vals), "min": vals.min(), "max": vals.max()}
    return {
        "path": path.name,
        "sha256": digest(path),
        "rows": frame.height,
        "schema": {k: str(v) for k, v in frame.schema.items()},
        "null_counts": frame.null_count().to_dicts()[0],
        "coverage": coverage,
        "role": role,
    }


def player_crosswalk(rosters: pl.DataFrame) -> pl.DataFrame:
    """Modern competition IDs use GSIS IT; the 2019 sample and 2020 archive use legacy NFL IDs."""
    pairs = (
        rosters.select(
            pl.col("gsis_it_id").cast(pl.String).str.replace(r"\.0$", "").alias("nfl_id"),
            pl.col("gsis_id").cast(pl.String).alias("player_id"),
        )
        .drop_nulls()
        .unique()
    )
    return (
        pairs.group_by("nfl_id")
        .agg(
            pl.col("player_id").n_unique().alias("candidate_count"),
            pl.col("player_id").first(),
        )
        .with_columns(
            pl.when(pl.col("candidate_count") == 1)
            .then(pl.col("player_id"))
            .otherwise(None)
            .alias("player_id"),
            pl.when(pl.col("candidate_count") == 1)
            .then(pl.lit("exact_provider_crosswalk"))
            .otherwise(pl.lit("ambiguous"))
            .alias("mapping_status"),
        )
        .sort("nfl_id")
    )


def scan_tracking(release, *, include_labels: bool = False) -> pl.LazyFrame:
    """Verified, lazy multi-partition read. Prediction output labels are opt-in."""
    roles = {"tracking_observation"}
    if include_labels:
        roles.add("tracking_labels")
    scans = [
        pl.scan_parquet(release.path(name))
        for name, spec in release.manifest["tables"].items()
        if spec["role"] in roles
    ]
    if not scans:
        raise ValueError("Release has no normalized frame/snapshot tables")
    return pl.concat(scans, how="diagonal_relaxed")


def link_tracking_players(tracking: pl.LazyFrame, roster_release) -> pl.LazyFrame:
    """Join only stable provider identity, never current teams/status as historical facts."""
    ids = roster_release.read("tracking_player_ids").with_columns(
        pl.lit("gsis_it").alias("id_namespace")
    )
    return tracking.join(
        ids.lazy(), on=["id_namespace", "nfl_id"], how="left", validate="m:1"
    ).with_columns(
        pl.when(pl.col("entity_type") == "ball")
        .then(pl.lit("not_a_player"))
        .otherwise(pl.col("mapping_status").fill_null("unmapped"))
        .alias("mapping_status")
    )


def build(data_dir: Path, raw_ref: dict) -> dict:
    """One verified table batch per auxiliary source capture."""
    _, raw = load_manifest(data_dir, "raw", raw_ref)
    version = raw["version"]
    source = Source(**raw["source"])
    legacy = data_dir / "gold/releases" / version
    if (legacy / "manifest.json").exists():
        return load_gold(data_dir, version).ref
    gold_root = data_dir / "tables/batches" / version
    if (gold_root / "manifest.json").exists():
        return load_gold(data_dir, version).ref
    enriched_root = data_dir / "enriched/releases" / version
    if enriched_root.exists() or gold_root.exists():
        raise ValueError("Partial build exists; use a new version (raw bytes deduplicate)")
    enriched_root.mkdir(parents=True)
    gold_root.mkdir(parents=True)
    enriched_tables, gold_tables = {}, {}
    tracking_paths = []
    for name, asset in raw["assets"].items():
        frame = read_source(inside(data_dir, asset["object"]), name)
        key = table_name(name)
        if key in enriched_tables:
            raise ValueError(f"Colliding table names: {name}")
        enriched_tables[key] = table_spec(enriched_root, key, frame, role="provider_observation")
        if is_spatial_file(name, source):
            normalized = normalize_tracking(
                frame, source.id, name, handoff=source.adapter == "handoff"
            )
            role = (
                "tracking_labels"
                if Path(name).name.startswith("output_")
                else "tracking_observation"
            )
            gold_tables[key] = table_spec(gold_root, key, normalized, role=role)
            tracking_paths.append(gold_root / gold_tables[key]["path"])
        elif source.adapter == "ngs":
            required = ["season", "season_type", "week", "player_gsis_id"]
            if not set(required) <= set(frame.columns):
                raise ValueError(f"Missing NGS keys: {name}")
            if frame.select(pl.any_horizontal(pl.col(required).is_null()).any()).item():
                raise ValueError(f"Null NGS key: {name}")
            if frame.select(pl.struct(required).is_duplicated().any()).item():
                raise ValueError(f"Duplicate NGS key: {name}")
            for suffix, part in [
                ("weekly", frame.filter(pl.col("week") > 0)),
                ("season", frame.filter(pl.col("week") == 0)),
            ]:
                part = part.rename({"player_gsis_id": "player_id"})
                gold_tables[f"{key}_{suffix}"] = table_spec(
                    gold_root, f"{key}_{suffix}", part, role=f"tracking_derived_{suffix}"
                )
        else:
            primary = None
            if source.id == "nflverse_participation":
                primary = ["nflverse_game_id", "play_id"]
            elif source.id == "nflverse_ftn":
                primary = ["nflverse_game_id", "nflverse_play_id"]
            if primary:
                if not set(primary) <= set(frame.columns):
                    raise ValueError(f"Missing play keys: {name}")
                if frame.select(pl.any_horizontal(pl.col(primary).is_null()).any()).item():
                    raise ValueError(f"Null play key: {name}")
                if frame.select(pl.struct(primary).is_duplicated().any()).item():
                    raise ValueError(f"Duplicate play key: {name}")
            role = (
                "context_with_possible_outcomes"
                if source.adapter in {"bdb", "handoff"}
                else source.kind
            )
            gold_tables[key] = table_spec(gold_root, key, frame, role=role)
        print(f"Normalize {source.id}: {name} ({frame.height:,} rows)", flush=True)
    if tracking_paths:
        keys = pl.concat([pl.scan_parquet(p).select(KEY) for p in tracking_paths])
        duplicate = keys.group_by(KEY).len().filter(pl.col("len") > 1).limit(1).collect()
        if duplicate.height:
            raise ValueError("Overlapping tracking partitions; select one source vintage")
    if source.id == "nflverse_rosters":
        rosters = pl.concat(
            [
                pl.read_parquet(
                    gold_root / spec["path"], columns=["gsis_it_id", "gsis_id"]
                ).with_columns(pl.col("gsis_it_id").cast(pl.String))
                for spec in gold_tables.values()
            ],
            how="vertical_relaxed",
        )
        gold_tables["tracking_player_ids"] = table_spec(
            gold_root, "tracking_player_ids", player_crosswalk(rosters), role="identity_crosswalk"
        )
    for layer, root, tables, dependency in [
        ("enriched", enriched_root, enriched_tables, raw_ref),
        ("tables", gold_root, gold_tables, None),
    ]:
        quality = {
            "checks": {
                "nonempty_table_inventory": bool(tables),
                "source_bytes_verified": True,
                "schema_checks_passed": True,
            },
            "limits": [
                "Source-scoped observations only; not approved forecast inputs",
                "Actual game dates are not historical publication timestamps",
                "Auxiliary metadata may cover games without tracking",
                "Coordinates outside field retained and flagged",
            ],
        }
        write_json(root / "quality.json", quality)
        files = {s["path"]: s["sha256"] for s in tables.values()}
        files["quality.json"] = digest(root / "quality.json")
        implementation = Path(__file__)
        shutil.copyfile(implementation, root / "implementation.py")
        files["implementation.py"] = digest(root / "implementation.py")
        registry = Path(__file__).with_name("tracking_sources.py")
        shutil.copyfile(registry, root / "tracking_sources.py")
        files["tracking_sources.py"] = digest(root / "tracking_sources.py")
        manifest = {
            "schema_version": 1,
            "version": version,
            "layer": layer,
            "status": "accepted",
            "quality_passed": True,
            "kind": "auxiliary_source",
            "scope": "research_observations",
            "source": source.metadata(),
            "created_at": now(),
            "input": dependency or reference(enriched_root),
            "files": files,
            "tables": tables,
            "publication_time_known": False,
            "environment": {"polars": pl.__version__},
        }
        atomic_json(root / "manifest.json", manifest)
    return load_gold(data_dir, version).ref


def register(data_dir: Path, source: Source, gold_ref: dict):
    """Select a source package independently of the application serving release."""
    release = load_gold(data_dir, gold_ref["version"])
    if release.ref != gold_ref or release.manifest["source"]["id"] != source.id:
        raise ValueError("Source identity mismatch")
    path = data_dir / "source_catalog.json"
    catalog = (
        json.loads(path.read_text()) if path.exists() else {"schema_version": 1, "sources": {}}
    )
    catalog["sources"][source.id] = gold_ref
    catalog["updated_at"] = now()
    atomic_json(path, catalog)


def load_source(data_dir: Path, source_id: str):
    catalog = json.loads((data_dir / "source_catalog.json").read_text())
    ref = catalog["sources"][source_id]
    release = load_gold(data_dir, ref["version"])
    if release.ref != ref or release.manifest["source"]["id"] != source_id:
        raise ValueError("Source catalog identity mismatch")
    return release


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("inventory")
    commands.add_parser("status")
    ingest = commands.add_parser("ingest")
    ingest.add_argument("source", choices=sorted(SOURCES))
    ingest.add_argument("--version", required=True)
    mode = ingest.add_mutually_exclusive_group()
    mode.add_argument("--local", type=Path)
    mode.add_argument("--kaggle", action="store_true")
    verify = commands.add_parser("verify")
    verify.add_argument("--version", required=True)
    args = parser.parse_args()
    if args.command == "inventory":
        print(json.dumps([s.metadata() for s in SOURCES.values()], indent=2))
    elif args.command == "status":
        path = args.data_dir / "source_catalog.json"
        print(path.read_text() if path.exists() else '{"sources": {}}')
    elif args.command == "verify":
        release = load_gold(args.data_dir, args.version)
        print(json.dumps({"verified": release.ref, "tables": len(release.manifest["tables"])}))
    else:
        runtime = args.data_dir / ".runtime"
        runtime.mkdir(parents=True, exist_ok=True)
        with (runtime / "source_ingest.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            source = SOURCES[args.source]
            raw_ref = capture(args.data_dir, source, args.version, args.local, kaggle=args.kaggle)
            gold_ref = build(args.data_dir, raw_ref)
            register(args.data_dir, source, gold_ref)
            print(json.dumps(gold_ref))


if __name__ == "__main__":
    main()
