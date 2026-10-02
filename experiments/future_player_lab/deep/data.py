"""Verified raw histories plus the full admissible statistical inventory."""

import json

import numpy as np
import polars as pl

from ..data import build_panel, digest, load_inputs, numeric, validate_candidates

KEYS = ["player_id", "position", "year", "origin", "horizon"]


def verified_files(root, directory, filenames, pinned):
    base = root / directory
    manifest_path = base / "manifest.json"
    if digest(manifest_path) != pinned[str(manifest_path.relative_to(root))]:
        raise ValueError("Source manifest changed")
    manifest = json.loads(manifest_path.read_text())
    hashes = {str(manifest_path.relative_to(root)): digest(manifest_path)}
    entries = manifest.get("files", manifest.get("artifacts", {}))
    for name in filenames:
        path = base / name
        expected = entries[name]
        if isinstance(expected, dict):
            expected = expected["sha256"]
        if digest(path) != expected:
            raise ValueError(f"Source file changed: {path}")
        hashes[str(path.relative_to(root))] = expected
    return hashes


def inventory_columns(registry, frame):
    # Input validity only. Never use screened gains, full-data aliases or coverage.
    rows = [
        r
        for r in registry
        if r["status"] in {"admitted_research_only", "constant", "unavailable_in_gold"}
    ]
    rows = [
        r
        for r in rows
        if r["stat"] in frame
        and (frame.schema[r["stat"]].is_numeric() or frame.schema[r["stat"]] == pl.Boolean)
    ]
    if any(r["stat"].startswith(("actual_", "fitted_", "adaptive_", "week1_proxy_")) for r in rows):
        raise ValueError("Unsafe predictor")
    return rows


def prepare(root, config, run):
    tables, hashes = load_inputs(root, config)
    inv = root / config["inventory_root"]
    hashes.update(
        verified_files(
            root,
            config["inventory_root"],
            ["features.parquet", "registry.json", "specs.json"],
            config["pinned_manifests"],
        )
    )
    source = pl.read_parquet(inv / "features.parquet")
    validate_candidates(source)
    registry = json.loads((inv / "registry.json").read_text())
    admitted = inventory_columns(registry, source)
    cols = [r["stat"] for r in admitted]
    panel, raw, targets, baseline, names, groups = build_panel(tables, config)
    indexed = source.select(
        "player_id",
        pl.col("forecast_season").alias("year"),
        pl.col("forecast_cutoff_date").alias("inventory_cutoff"),
        "actual_season_points",
        *cols,
    ).with_row_index("inventory_row")
    attached = panel.join(
        indexed.select(
            "player_id", "year", "inventory_row", "inventory_cutoff", "actual_season_points"
        ),
        on=["player_id", "year"],
        how="left",
        validate="m:1",
    )
    if attached["inventory_row"].null_count():
        raise ValueError("Inventory does not cover lab candidates")
    if not (attached["forecast_cutoff_date"] == attached["inventory_cutoff"].cast(pl.String)).all():
        raise ValueError("Preseason inventory cutoff mismatch")
    season = panel["horizon"].to_numpy() == "season"
    if not np.allclose(
        targets[season, 0], attached["actual_season_points"].to_numpy()[season], rtol=0, atol=1e-7
    ):
        raise ValueError("Season labels disagree with inventory")
    inventory = numeric(indexed, cols)[attached["inventory_row"].to_numpy()]
    full = np.column_stack([raw, inventory]).astype(np.float32)
    names += ["inventory:" + c for c in cols]
    groups += ["inventory:" + r["family"] for r in admitted]
    raw_indices = list(range(raw.shape[1]))
    own = [raw.shape[1] + i for i, r in enumerate(admitted) if not r["market_input"]]
    market = [raw.shape[1] + i for i, r in enumerate(admitted) if r["market_input"]]
    basic = json.loads((inv / "specs.json").read_text())["basic"]
    views = {
        "raw": raw_indices,
        "inventory": own,
        "all": raw_indices + own,
        "market": list(range(full.shape[1])),
        "summary": [i for i in raw_indices if not names[i].startswith("sequence:")],
        "basic": [raw.shape[1] + cols.index(c) for c in basic],
    }
    # In-season raw data are appended to compact/inventory views, keeping the
    # preseason inventory dated while providing observations through the origin.
    for view in ("inventory", "basic"):
        views[view] += raw_indices
        views[view] = sorted(set(views[view]))
    panel.write_parquet(run / "panel.parquet")
    np.save(run / "x.npy", full)
    np.save(run / "targets.npy", targets)
    np.save(run / "baseline.npy", baseline * panel["exposure"].to_numpy()[:, None])
    meta = {
        "names": names,
        "groups": groups,
        "views": views,
        "rows": panel.height,
        "columns": full.shape[1],
        "inventory_columns": len(cols),
        "market_columns": len(market),
        "unique_players": panel["player_id"].n_unique(),
        "years": sorted(panel["year"].unique().to_list()),
        "retrospective_inventory_vintage": True,
        "unknown_labels": {str(i): int(np.isnan(targets[:, i]).sum()) for i in range(5)},
    }
    (run / "features.json").write_text(json.dumps(meta, indent=2))
    return hashes, meta


def published_benchmark(root, config):
    directory = config["published_root"]
    hashes = verified_files(
        root,
        directory,
        ["ranking_predictions.parquet", "qb_variations/decisions.json"],
        config["pinned_manifests"],
    )
    decisions = json.loads((root / directory / "qb_variations/decisions.json").read_text())
    approved = {(d["target"], d["horizon"]): d["approved"] for d in decisions}
    if (
        approved.get(("league_points", "next4")) is not True
        or approved.get(("league_points", "rest_of_season")) is not False
    ):
        raise ValueError("Published scope changed; review benchmark adapter")
    frame = pl.read_parquet(root / directory / "ranking_predictions.parquet")
    # Published September 24 policy: QB next4 integration, reference elsewhere.
    # Pin the decision source as well as the exact saved predictions.
    return (
        frame.select(
            "player_id",
            "position",
            pl.col("season").alias("year"),
            pl.col("through_week").alias("origin"),
            pl.col("horizon").replace({"next4": "next_four", "rest_of_season": "remaining"}),
            pl.col("end_week").alias("published_end"),
            pl.col("actual").alias("published_actual"),
            pl.when((pl.col("position") == "QB") & (pl.col("horizon") == "next4"))
            .then(pl.col("policy"))
            .otherwise(pl.col("reference"))
            .alias("published"),
            pl.col("reference").alias("published_reference"),
        ),
        hashes,
        decisions,
    )
