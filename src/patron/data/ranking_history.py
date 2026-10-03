"""Historical ranks from the verified publication chain, never unpublished experiments."""

from pathlib import Path

from patron.data.frames import read_frame
from patron.data.nextgen import eligible, load_analysis
from patron.data.releases import current_catalog, digest, identifier
from patron.data.verification import read_json, verified


def ranking_history(data: Path, season: int, through_week: int, horizon: str) -> list[dict]:
    catalog = current_catalog(data)
    if not catalog:
        return []

    def load():
        snapshots = {}
        seen = set()
        pointer = catalog.get("previous_catalog_sha256")
        # Newest publication for each completed-week cutoff wins. Same-week recipe
        # revisions are not week-to-week movement.
        while pointer:
            if pointer in seen:
                raise ValueError("Cyclic catalog history")
            seen.add(pointer)
            path = data / "catalog_history" / (identifier(pointer) + ".json")
            if digest(path) != pointer:
                raise ValueError("Historical catalog changed")
            previous = read_json(path)
            ref = previous.get("products", {}).get("analysis")
            if ref:
                root = data / "research" / identifier(ref["version"])
                if digest(root / "manifest.json") != ref["manifest_sha256"]:
                    raise ValueError("Historical ranking manifest changed")
                manifest = read_json(root / "manifest.json")
                if manifest.get("ranking_schema_version") == 1:
                    report_path = root / "ranking_report.json"
                    if digest(report_path) != manifest["files"]["ranking_report.json"]:
                        raise ValueError("Historical ranking report changed")
                    report = read_json(report_path)
                    week = report["through_week"]
                    if (
                        report["season"] == season
                        and 0 <= week < through_week
                        and week not in snapshots
                    ):
                        # Verify the complete historical release before showing its ranks.
                        root, _ = load_analysis(data, ref)
                        registry = {r["id"]: r for r in read_json(root / "registry.json")}
                        incidents = read_json(root / "incidents.json")
                        rows = read_frame(root / "rankings.parquet").to_dicts()
                        ranks = []
                        for row in rows:
                            if row["horizon"] != horizon:
                                continue
                            allowed, _ = eligible(
                                registry.get(row["entry_id"], {}),
                                use="ranking",
                                target="league_points",
                                horizon=horizon,
                                position=row["position"],
                                population=row["population"],
                                incidents=incidents,
                            )
                            if allowed:
                                ranks.append(
                                    {
                                        k: row.get(k)
                                        for k in (
                                            "player_id",
                                            "overall_rank",
                                            "position_rank",
                                            "prediction",
                                            "scheduled_games",
                                        )
                                    }
                                )
                        snapshots[week] = dict(
                            through_week=week, version=ref["version"], rankings=ranks
                        )
            pointer = previous.get("previous_catalog_sha256")
        return [snapshots[w] for w in sorted(snapshots)]

    return verified(
        (
            "ranking-history",
            str(data.absolute()),
            catalog.get("previous_catalog_sha256"),
            season,
            through_week,
            horizon,
        ),
        load,
    )
