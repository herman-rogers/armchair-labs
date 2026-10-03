"""Table-specific bundles use the existing verified, generation-pinned GCS transport."""

from __future__ import annotations

import json
from pathlib import Path

from engine.data import shared
from engine.data.releases import atomic_json
from engine.tables.research import verify_archive
from engine.tables.storage import (
    content_hash,
    current,
    file_spec,
    load_catalog,
    load_table,
    publish_catalog,
)

DEFAULT_STORE = "gs://armchair-labs-data/armchair-labs/tables"


def select_published(data_dir: Path, reference: dict) -> None:
    """Advance the local handoff reference only for the still-selected catalog.

    This small file is safe to distribute/commit. Consumers pin its immutable
    release; there is deliberately no mutable global GCS latest pointer.
    """
    with shared.lock(data_dir / ".runtime/table-refresh"):
        if content_hash(current(data_dir)) == reference["table_catalog"]["sha256"]:
            atomic_json(data_dir / "releases/tables/current.json", reference)


def publish(
    data_dir: Path,
    release: str,
    *,
    store_uri: str = DEFAULT_STORE,
    cache: Path | None = None,
    research_runs: list[str] | None = None,
    expected_catalog: str | None = None,
) -> dict:
    data_dir = data_dir.resolve()
    if data_dir.name != "data":
        raise ValueError("Transport requires a data directory named 'data' under the project root")
    shared.identifier(release)
    cache = cache or shared.default_cache()
    stage = data_dir / ".shared/tables" / release
    snapshot_path = stage / "manifest.json"
    with shared.lock(data_dir / ".runtime/table-refresh"):
        catalog = current(data_dir)
        if expected_catalog is not None and content_hash(catalog) != expected_catalog:
            raise ValueError("Table catalog changed between build and upload; retry the workflow")
        if not catalog["tables"]:
            raise ValueError("No tables published; run tables refresh first")
        catalog_ref = json.loads((data_dir / "tables/current.json").read_text())
        includes = [f"data/{catalog_ref['path']}"]
        inventory = {}
        for name, ref in catalog["tables"].items():
            root, table = load_table(data_dir, name, ref)
            includes.append(root.relative_to(data_dir.parent).as_posix())
            inventory[name] = {
                "sql_name": f"analytics.{name}",
                "version": table["version"],
                "rows": table["rows"],
                "columns": table["columns"],
                "primary_key": table["recipe"]["primary_key"],
                "description": table["recipe"]["description"],
                "limitations": table["recipe"].get("limitations", []),
                "observation_cutoff": table["observation_cutoff"],
                "source_release": table.get("source_release"),
                "dependencies": table["dependencies"],
                "product": table["recipe"]["parameters"].get("product"),
                "parquet": table["files"]["data.parquet"],
            }
        profiles = {"tables": {"include": sorted(includes)}}
        if research_runs:
            research_paths = []
            for run in research_runs:
                shared.identifier(run)
                root = data_dir / "research" / run
                verify_archive(root)
                research_paths.append(f"data/research/{run}")
            profiles["research"] = {"include": sorted(research_paths)}
        config = {"schema_version": 1, "profiles": profiles}
        if snapshot_path.exists():
            manifest = json.loads(snapshot_path.read_text())
            if manifest["profiles"] != profiles:
                raise ValueError("Release ID belongs to a different snapshot; use a new release ID")
        else:
            manifest = shared.snapshot(data_dir.parent, config, release, cache, snapshot_path)
    store = shared.make_store(store_uri)
    result = shared.publish(manifest, store, cache, stage / "published.json")
    published = json.loads((stage / "published.json").read_text())
    for table in inventory.values():
        parquet = table["parquet"]
        parquet.update(
            uri=f"{store_uri.rstrip('/')}/{shared.object_key(parquet['sha256'])}",
            generation=published["object_generations"][parquet["sha256"]],
        )
    # A language-neutral, small inventory is the public integration contract.
    # Publish it only after the complete table bundle is remotely available.
    consumer = {
        "schema_version": 1,
        "release": release,
        "table_catalog": catalog_ref,
        "tables": inventory,
    }
    consumer_path = stage / "consumer.json"
    shared.write_new(consumer_path, shared.json_bytes(consumer))
    key = f"releases/{release}/catalog.json"
    spec = file_spec(consumer_path)
    generation = store.put(key, consumer_path, spec)
    reference = {
        **result,
        "store": store_uri,
        "table_catalog": catalog_ref,
        "consumer_catalog": {
            **spec,
            "key": key,
            "uri": f"{store_uri.rstrip('/')}/{key}",
            "generation": generation,
        },
    }
    destination = data_dir / "releases/tables" / f"{release}.json"
    shared.write_new(destination, shared.json_bytes(reference))
    select_published(data_dir, reference)
    return {"reference": str(destination), **reference}


def fetch(
    data_dir: Path,
    reference_path: Path,
    *,
    cache: Path | None = None,
    offline: bool = False,
    include_research: bool = False,
) -> dict:
    data_dir = data_dir.resolve()
    if data_dir.name != "data":
        raise ValueError("Transport requires a data directory named 'data' under the project root")
    cache = cache or shared.default_cache()
    reference = json.loads(reference_path.read_text())
    manifest = shared.read_release(reference, cache, offline=offline)
    ref = reference["table_catalog"]
    # Bind the activation pointer to bytes in the generation-pinned transport manifest.
    files = shared.selected(manifest, "tables")
    if files.get(f"data/{ref['path']}", {}).get("sha256") != ref["sha256"]:
        raise ValueError("Table catalog is not bound to the remote release")
    if include_research and "research" not in manifest["profiles"]:
        raise ValueError("This release contains no research profile")
    store = None if offline else shared.make_store(reference["store"])
    with shared.lock(data_dir / ".runtime/table-refresh"):
        shared.fetch(manifest, data_dir.parent, cache, "tables", store)
        if include_research:
            shared.fetch(manifest, data_dir.parent, cache, "research", store)
        catalog = load_catalog(data_dir, ref)
        publish_catalog(data_dir, catalog)
    select_published(data_dir, reference)
    return {"release": reference["release"], "tables": sorted(catalog["tables"])}
