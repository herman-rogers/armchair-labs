"""Add exact-ID drafted rookies to frozen v2 panels, preserving every existing value."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import polars as pl

from ..forecasting.data import add_supplements, history_inputs, snapshot_inputs
from ..gpu_lab.run import digest, json_write
from . import data
from .availability import data as availability

ROOT = data.ROOT
V1 = availability.V1_RUN
V2 = ROOT / "experiments/future_player_lab/runs/opening_drive_v2_001"
RUN = ROOT / "experiments/future_player_lab/runs/opening_drive_v2_update_20261010_001"
CACHE = RUN / "inputs_candidate"
CURRENT = RUN / "current_candidates"
SNAPSHOT = ROOT / "data/raw/snapshots/weekly_20261008_w4_r3/manifest.json"
KEY = ["year", "origin", "player_id"]


@lru_cache(maxsize=1)
def admission_tables():
    base = V1 / "inputs/history"
    receipt = json.loads((base / "manifest.json").read_text())
    return {
        name: availability.checked_parquet(
            base, f"{name}.parquet", receipt["files"][f"{name}.parquet"]
        )
        for name in ["combine", "mapping", "identity"]
    }


@lru_cache(maxsize=1)
def history_tables():
    return history_inputs(V1 / "inputs/history")


@lru_cache(maxsize=1)
def current_tables():
    tables, _ = snapshot_inputs(history_tables(), SNAPSHOT, 2026, 2)
    tables, _ = add_supplements(tables, availability.SUPPLEMENTS, 2026, 2)
    return tables


def contract(position, horizon):
    return dict(
        position=position,
        horizon=horizon,
        admission_source_sha256=digest(Path(data.__file__)),
        revision_source_sha256=digest(Path(__file__)),
        dependency_sha256={str(p.relative_to(ROOT)): digest(p) for p in data.dependency_paths()},
        history_manifest_sha256=digest(V1 / "inputs/history/manifest.json"),
        availability_manifest_sha256=digest(availability.SOURCE_DIR / "manifest.json"),
        availability_layer_sha256=digest(Path(availability.layer.__file__)),
        policy="Append only absent exact-ID drafted rookies; preserve every frozen old value",
    )


def missing_rookies(panel, position, year):
    entrants = data.dated_entrants(admission_tables(), year)
    entrants = entrants.filter((pl.col("draft_year") == year) & (pl.col("pos") == position))
    return (
        entrants.select("player_id")
        .unique()
        .join(panel.filter(pl.col("year") == year).select("player_id"), on="player_id", how="anti")[
            "player_id"
        ]
        .to_list()
    )


def only_candidates(tables, ids):
    """Player filtering is an optimization; complete team/schedule context is retained."""
    result = dict(tables)
    for name in ["box", "player", "college"]:
        result[name] = tables[name].filter(pl.col("player_id").is_in(ids))
    pfr = tables["mapping"].filter(pl.col("player_id").is_in(ids))["pfr_id"]
    result["combine"] = tables["combine"].filter(pl.col("pfr_id").is_in(pfr.implode()))
    return result


def append_preserving(base, additions):
    """Align dtypes and prove existing rows survive unchanged, including nulls."""
    if not additions.height:
        return base.clone()
    missing = set(base.columns) - set(additions.columns)
    if missing:
        raise ValueError(f"Added candidate schema is incomplete: {sorted(missing)}")
    addition = additions.select(
        [pl.col(c).cast(dtype, strict=True) for c, dtype in base.schema.items()]
    )
    if base.select(KEY).join(addition.select(KEY), on=KEY, how="inner").height:
        raise ValueError("Revision attempts to replace an existing candidate")
    result = pl.concat([base, addition], how="vertical").sort(KEY)
    old = result.join(base.select(KEY), on=KEY, how="semi").sort(KEY)
    if not old.equals(base.sort(KEY)):
        raise ValueError("Candidate revision changed an existing feature or outcome")
    if result.select(KEY).is_duplicated().any():
        raise ValueError("Duplicate revised candidate")
    return result


def load_dataset(position, horizon, cache=CACHE):
    """Historical corrected panel and unchanged v2 feature order; checksum-bound cache."""
    model_id = f"{position.lower()}_{horizon}"
    directory = Path(cache) / model_id
    v1dir = V1 / "inputs" / model_id
    v2dir = V2 / "inputs" / model_id
    v1meta = json.loads((v1dir / "dataset.json").read_text())
    v2meta = json.loads((v2dir / "dataset.json").read_text())
    receipt = contract(position, horizon) | dict(
        parent_v1_sha256=digest(v1dir / "dataset.json"),
        parent_v2_sha256=digest(v2dir / "dataset.json"),
    )
    if (directory / "dataset.json").exists():
        saved = json.loads((directory / "dataset.json").read_text())
        if saved["contract"] != receipt:
            raise ValueError("Revision data contract changed")
        return availability.checked_parquet(
            directory, "panel.parquet", saved["panel_sha256"]
        ), saved["feature_names"]
    base = availability.checked_parquet(v1dir, "panel.parquet", v1meta["panel_sha256"])
    old = availability.checked_parquet(v2dir, "panel.parquet", v2meta["panel_sha256"])
    pieces = []
    counts = []
    origin = 0 if horizon == "season" else 2
    for year in sorted(base["year"].unique()):
        ids = missing_rookies(base, position, year)
        counts.append(dict(year=year, added=len(ids)))
        if ids:
            tables = only_candidates(history_tables(), ids)
            new = data.feature_rows(tables, position, year, origin).filter(
                pl.col("player_id").is_in(ids)
            )
            new = data.attach_targets(new, tables, position, year, origin)
            pieces.append(new)
    additions = pl.concat(pieces, how="diagonal_relaxed") if pieces else base.head(0)
    corrected = append_preserving(base, additions)
    sources, _ = availability.load_sources()
    audit = availability.augment(corrected, sources)
    panel = audit.filter(pl.col("availability_eligible"))
    preserved = panel.join(old.select(KEY), on=KEY, how="semi").select(old.columns).sort(KEY)
    if not preserved.equals(old.sort(KEY)):
        raise ValueError("Existing v2 historical rows changed")
    names = v2meta["feature_names"]
    if set(names) - set(panel.columns):
        raise ValueError("Historical revision feature contract differs")
    directory.mkdir(parents=True, exist_ok=False)
    panel.write_parquet(directory / "panel.parquet")
    audit.select(
        *KEY,
        "player_display_name",
        "team",
        *[c for c in audit.columns if c.startswith("availability")],
    ).write_parquet(directory / "candidate_audit.parquet")
    additions.write_parquet(directory / "added_candidates.parquet")
    json_write(
        directory / "dataset.json",
        dict(
            contract=receipt,
            feature_names=names,
            rows=panel.height,
            columns=len(names),
            parent_rows=old.height,
            added_before_availability=additions.height,
            added_eligible=panel.height - old.height,
            added_by_year=counts,
            existing_rows_exactly_preserved=True,
            panel_sha256=digest(directory / "panel.parquet"),
            candidate_audit_sha256=digest(directory / "candidate_audit.parquet"),
            added_candidates_sha256=digest(directory / "added_candidates.parquet"),
        ),
    )
    return panel, names


def current_panel(position, horizon):
    """Return (panel, feature_names, receipt) for the same saved 2026 origin."""
    model_id = f"{position.lower()}_{horizon}"
    directory = CURRENT / model_id
    campaign = ROOT / "data/campaigns/opening_drive_v2"
    manifest = json.loads((campaign / "manifest.json").read_text())
    model = next(m for m in manifest["models"] if m["model_id"] == model_id)
    bound = model["artifacts"]["forecast_features"]
    receipt = contract(position, horizon) | dict(
        parent_campaign_sha256=digest(campaign / "manifest.json"),
        parent_features_sha256=bound["sha256"],
        snapshot_sha256=digest(SNAPSHOT),
        supplements_sha256=digest(availability.SUPPLEMENTS / "manifest.json"),
    )
    if (directory / "manifest.json").exists():
        saved = json.loads((directory / "manifest.json").read_text())
        if saved["contract"] != receipt:
            raise ValueError("Current candidate revision contract changed")
        panel = availability.checked_parquet(
            directory, "features.parquet", saved["feature_parquet_sha256"]
        )
        return panel, saved["feature_names"], saved
    old = availability.checked_parquet(campaign, bound["path"], bound["sha256"])
    feature_ref = model["artifacts"]["feature_contract"]
    if digest(campaign / feature_ref["path"]) != feature_ref["sha256"]:
        raise ValueError("Published feature names checksum mismatch")
    names = json.loads((campaign / feature_ref["path"]).read_text())
    ids = missing_rookies(old, position, 2026)
    if ids:
        tables = only_candidates(current_tables(), ids)
        additions = data.feature_rows(tables, position, 2026, model["origin_week"]).filter(
            pl.col("player_id").is_in(ids)
        )
        sources, _ = availability.load_sources()
        audit = availability.augment(additions, sources)
        eligible = audit.filter(pl.col("availability_eligible"))
    else:
        audit = old.head(0)
        eligible = old.head(0)
    panel = append_preserving(old, eligible)
    directory.mkdir(parents=True, exist_ok=False)
    panel.write_parquet(directory / "features.parquet")
    audit.write_parquet(directory / "added_candidate_audit.parquet")
    saved = dict(
        contract=receipt,
        feature_names=names,
        rows=panel.height,
        parent_rows=old.height,
        added_eligible=eligible.height,
        existing_rows_exactly_preserved=True,
        added_candidates=audit.select(
            "player_id",
            "player_display_name",
            "team",
            "availability_status",
            "availability_eligible",
        ).to_dicts(),
        feature_parquet_sha256=digest(directory / "features.parquet"),
        added_candidate_audit_sha256=digest(directory / "added_candidate_audit.parquet"),
        limitation=(
            "Retrospective immutable draft facts; current capture can lack rookie weekly records. "
            "Missing records remain unavailable, not proof of zero real-world appearances."
        ),
    )
    json_write(directory / "manifest.json", saved)
    return panel, names, saved
