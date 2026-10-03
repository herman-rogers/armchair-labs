"""Preserve current injury observations in the existing immutable raw store.

ESPN statuses, captured reporting, and reviewed interpretations retain separate
provenance. This collector does not change historical forecasts or the gold catalog.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from collections import Counter
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl
import requests

from engine.data.historical_evidence import article_content
from engine.data.releases import (
    digest,
    identifier,
    load_manifest,
    preserve_object,
    reference,
    write_json,
)


def capture(data: Path, version: str, snapshot_path: Path, review_path: Path) -> Path:
    root = data / "raw/snapshots" / identifier(version)
    if root.exists():
        raise FileExistsError(f"Use a new version; raw captures are immutable: {root}")
    snapshot_bytes = snapshot_path.read_bytes()
    snapshot = json.loads(snapshot_bytes)
    observed_at = datetime.fromisoformat(snapshot["captured_at"])
    if observed_at.utcoffset() is None:
        raise ValueError("League observation timestamp must include its timezone")
    review_bytes = review_path.read_bytes()
    review = json.loads(review_bytes)
    if review.get("schema_version") != 1:
        raise ValueError("Unsupported injury review schema")
    sources = review["sources"]
    urls = [s["url"] for s in sources]
    if len(urls) != len(set(urls)) or any(not u.startswith("https://") for u in urls):
        raise ValueError("Sources require unique HTTPS URLs")
    for source in sources:
        published = date.fromisoformat(source["published_on"])
        for event in source.get("observations", []):
            if date.fromisoformat(event["known_on"]) < published:
                raise ValueError("An observation cannot be known before its source")
    assets = {}
    with tempfile.TemporaryDirectory(prefix="engine-injuries-") as temporary:
        stage = Path(temporary)

        def add(name, path, kind, **metadata):
            expected = digest(path)
            assets[name] = {
                "object": preserve_object(data, path, expected),
                "sha256": expected,
                "bytes": path.stat().st_size,
                "kind": kind,
                **metadata,
            }

        saved = stage / "league_snapshot.json"
        saved.write_bytes(snapshot_bytes)
        add(
            "injuries/league_snapshot.json",
            saved,
            "provider_snapshot",
            observed_at=snapshot["captured_at"],
        )
        saved = stage / "review.json"
        saved.write_bytes(review_bytes)
        add("injuries/review.json", saved, "reviewed_annotation")
        statuses = [
            {
                "espn_id": p["espn_id"],
                "player_name": p["player_display_name"],
                "position": p["position"],
                "team": p.get("espn_team"),
                "injury_status": p.get("injury_status"),
                "season": snapshot["season"],
                "week": snapshot["week"],
                "observed_at": snapshot["captured_at"],
                "source_published_at": None,
                "source_asset": "injuries/league_snapshot.json",
            }
            for p in snapshot["players"]
        ]
        if len({p["espn_id"] for p in statuses}) != len(statuses):
            raise ValueError("Duplicate ESPN identities in injury snapshot")
        parsed = stage / "espn_statuses.parquet"
        pl.DataFrame(statuses).write_parquet(parsed)
        add("injuries/espn_statuses.parquet", parsed, "parsed_provider_observation")
        observations = []
        for index, source in enumerate(sources):
            response = requests.get(
                source["url"], timeout=30, headers={"User-Agent": "Armchair Labs/1.0"}
            )
            response.raise_for_status()
            if not response.content:
                raise ValueError(f"Empty injury source: {source['url']}")
            retrieved_at = datetime.now(UTC).isoformat()
            parsed_article = article_content(response.content.decode("utf-8", errors="replace"))
            page = stage / f"{index}.html"
            page.write_bytes(response.content)
            name = f"injuries/sources/{index}.html"
            add(
                name,
                page,
                "captured_web_evidence",
                requested_url=source["url"],
                final_url=response.url,
                status_code=response.status_code,
                retrieved_at=retrieved_at,
                published_at=parsed_article["published_at"],
                modified_at=parsed_article["modified_at"],
                reviewed_published_on=source["published_on"],
                content_type=response.headers.get("Content-Type"),
            )
            observations.extend(
                {
                    **event,
                    "source_url": source["url"],
                    "source_asset": name,
                    "source_sha256": assets[name]["sha256"],
                    "retrieved_at": retrieved_at,
                    "source_published_on": source["published_on"],
                }
                for event in source.get("observations", [])
            )
        parsed = stage / "observations.json"
        write_json(parsed, {"schema_version": 1, "observations": observations})
        add("injuries/observations.json", parsed, "reviewed_annotation")
        manifest = {
            "schema_version": 1,
            "layer": "raw",
            "version": version,
            "status": "accepted",
            "kind": "current_injury_capture",
            "registered_at": datetime.now(UTC).isoformat(),
            "current_observations": {
                "season": snapshot["season"],
                "week": snapshot["week"],
                "saved_at": snapshot["captured_at"],
            },
            "assets": assets,
            "files": {a["object"]: a["sha256"] for a in assets.values()},
            "source_kinds": dict(Counter(a["kind"] for a in assets.values())),
            "status_counts": dict(Counter(p["injury_status"] for p in statuses)),
            "player_status_rows": len(statuses),
            "reviewed_observations": len(observations),
            "limitations": [
                "ESPN tags cover only the captured league player pool, not every NFL player.",
                "ACTIVE/NORMAL is a provider tag, not proof of health or a starting role.",
                "News interpretations are reviewed annotations, not raw medical facts.",
                "Reported expectations, team announcements, observation and capture dates "
                "remain distinct.",
                "Raw registration does not update gold or retroactively adjust saved forecasts.",
            ],
        }
        root.mkdir(parents=True, exist_ok=False)
        write_json(root / "manifest.json", manifest)
    load_manifest(data, "raw", reference(root))
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--league-snapshot", type=Path, default=Path("data/outputs/league_snapshot.json")
    )
    args = parser.parse_args()
    root = capture(args.data_dir, args.version, args.league_snapshot, args.review)
    manifest = json.loads((root / "manifest.json").read_text())
    print(
        json.dumps(
            {
                "snapshot": str(root),
                "players": manifest["player_status_rows"],
                "reviewed_observations": manifest["reviewed_observations"],
            }
        )
    )


if __name__ == "__main__":
    main()
