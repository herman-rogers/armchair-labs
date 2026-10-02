"""Import explicit reviewed absence decisions from immutable source captures."""

import argparse
import json
import re
from pathlib import Path

import polars as pl

from patron.data.historical_evidence import EvidenceCapture, article_content
from patron.metrics.availability import validate_absences
from patron.metrics.experimental import _normalized_name, _transaction_aliases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--players", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    path = args.data_dir / "static/historical_absences.json"
    ledger = json.loads(path.read_text())
    players = pl.read_parquet(args.players)
    names = dict(players.select("gsis_id", "display_name").iter_rows())
    aliases = _transaction_aliases(players)
    capture = EvidenceCapture(args.data_dir / "cache/historical_evidence/v1")
    for decision in json.loads(args.reviews.read_text()):
        url = decision["source_url"]
        # Import is offline: collection and source review must have happened first.
        from patron.data.historical_evidence import sha256

        if not (capture.root / f"{sha256(url.encode())}.json").exists():
            raise ValueError(f"Uncaptured review source: {url}")
        record, html = capture.fetch(url)
        article = article_content(html)
        if record["status_code"] != 200 or not article["text"]:
            raise ValueError(f"Unusable absence source: {url}")
        if article["published_at"][:10] != decision["source_published_on"]:
            raise ValueError(f"Publication date disagrees with reviewed source: {url}")
        pid, season, kind = decision["player_id"], decision["season"], decision["kind"]
        name = names[pid]
        body = f" {_normalized_name(re.sub(r'[’\x27]s\b', '', article['text']))} "
        if not any(f" {alias} " in body for alias in aliases[pid]):
            raise ValueError(f"Reviewed player identity is absent from source: {name}")
        existing = [
            e
            for e in ledger["events"]
            if (e["player_id"], e["season"], e["kind"]) == (pid, season, kind)
        ]
        incidents = {e["absence_id"] for e in existing}
        if len(incidents) > 1:
            raise ValueError("Review must identify which incident is revised")
        incident = next(iter(incidents), f"backfill-{pid}-{season}-{kind}")
        evidence_id = f"{incident}-{decision['known_on']}"
        if any(
            e["evidence_id"] == evidence_id
            or (e["source_url"] == url and e["player_id"] == pid and e["season"] == season)
            for e in ledger["events"]
        ):
            continue
        ledger["events"].append(
            {
                "evidence_id": evidence_id,
                "absence_id": incident,
                "player_id": pid,
                "player_name": name,
                "season": season,
                "kind": kind,
                "source_published_on": decision["source_published_on"],
                "known_on": decision["known_on"],
                "verified_on": record["retrieved_at"][:10],
                "unavailable_games": list(range(1, decision["games"] + 1)),
                "source_url": url,
                "source_sha256": record["sha256"],
                "source_capture_file": record["capture_file"],
                "source_modified_at": article["modified_at"],
                "review_basis": decision["review_basis"],
                "summary": f"{name}: first {decision['games']} regular-season team games "
                f"unavailable under the dated {kind} announcement.",
            }
        )
    validate_absences(ledger["events"])
    ledger["scope"] = (
        "Reviewed dated absence announcements; unreviewed news candidates remain separate. "
        "Coverage is not exhaustive."
    )
    path.write_text(json.dumps(ledger, indent=2) + "\n")
    print(f"Validated {len(ledger['events'])} absence events")


if __name__ == "__main__":
    main()
