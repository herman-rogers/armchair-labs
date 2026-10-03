"""The single table build → optional GCS publication orchestration path.

Called by both engine tables refresh and the source/product refresh. An upload
failure leaves validated local tables intact; repeating the command retries the
same immutable release even when no table needs rebuilding.
"""

import json
from pathlib import Path

from engine.tables import transport
from engine.tables.build import refresh
from engine.tables.registry import DEFAULT_REGISTRY


def refresh_and_publish(
    data_dir: Path,
    *,
    registry_path: Path = DEFAULT_REGISTRY,
    names: list[str] | None = None,
    due: bool = False,
    force: bool = False,
    upload: bool = False,
    store: str = transport.DEFAULT_STORE,
) -> dict:
    result = refresh(data_dir, registry_path=registry_path, names=names, due=due, force=force)
    if upload:
        # Store identity prevents the same catalog's publication to a test store
        # from colliding with its later publication to GCS.
        from engine.tables.storage import content_hash

        release = f"tables_{result['catalog'][:24]}_{content_hash({'store': store})[:8]}"
        existing = data_dir / "releases/tables" / f"{release}.json"
        if existing.exists():
            reference = json.loads(existing.read_text())
            if (
                reference["store"] != store
                or reference["table_catalog"]["sha256"] != result["catalog"]
                or "consumer_catalog" not in reference
            ):
                raise ValueError("Existing publication reference does not match this build")
            transport.select_published(data_dir, reference)
            result["upload"] = {"status": "already_published", "reference": str(existing)}
        else:
            result["upload"] = transport.publish(
                data_dir, release, store_uri=store, expected_catalog=result["catalog"]
            )
    return result
