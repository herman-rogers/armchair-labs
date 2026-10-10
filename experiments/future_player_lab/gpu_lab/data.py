"""Broad, cutoff-aware RB observations; no previously screened feature matrices."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from ..notebooks import raw_distribution_data as raw

STATS = tuple(raw.STATS)
RAW_MANIFEST_SHA256 = "f4595619cd7c261c006d31fd6b5cfd871bdd9fbc01daffa2e533766f1203613f"
DEFAULT_CACHE = raw.ROOT / "experiments/future_player_lab/runs/gpu_raw_inputs_001"
KEYS = ["player_id", "year", "origin"]


@dataclass
class Dataset:
    panel: pl.DataFrame
    x: np.ndarray
    names: list[str]
    groups: list[str]
    targets: np.ndarray
    exposure: np.ndarray
    baseline: np.ndarray
    audit: pl.DataFrame
    provenance: dict
    views: dict[str, list[int]]


def source_family(name: str) -> str:
    """Source-based partitions are descriptive, never chosen using outcome scores."""
    for marker, family in (
        ("pbp_", "plays"),
        ("team_prior__", "team_box"),
        ("team_current__", "team_box"),
        ("box__", "box"),
        ("snaps__", "snaps"),
        ("ngs__", "ngs"),
        ("depth__", "depth"),
        ("injuries__", "injuries"),
        ("rosters__", "rosters"),
        ("college__", "college"),
        ("combine__", "combine"),
    ):
        if marker in name:
            return family
    return "history_totals" if name.startswith("history_") else "calendar_demographics"


def validate_features(names: list[str]) -> None:
    prohibited = {"year", "target_records", "market_implied", "market_spread"}
    unsafe = [
        name
        for name in names
        if name in prohibited
        or name.startswith(("y_", "actual_", "fitted_", "adaptive_"))
        or name.endswith("_id")
    ]
    if unsafe:
        raise ValueError(f"Outcomes, identifiers, or future market data in features: {unsafe}")
    if len(names) != len(set(names)):
        raise ValueError("Duplicate feature names")


def add_history_views(panel: pl.DataFrame, observations: pl.DataFrame) -> pl.DataFrame:
    """All numeric source channels, with windows fixed independently of outcomes."""
    columns = raw.numeric(observations, ["player_id"])
    chunks = []
    for (year, origin), block in panel.partition_by(["year", "origin"], as_dict=True).items():
        known = raw.cutoff_player_history(observations, year, origin).sort(
            "player_id", "season", "week"
        )
        prior2 = known.filter(pl.col("season").is_between(year - 2, year - 1))
        recent8 = known.group_by("player_id", maintain_order=True).tail(8)
        for prefix, history in (("prior2", prior2), ("career", known), ("recent8", recent8)):
            block = block.join(
                raw.aggregate_history(history, "player_id", prefix),
                on="player_id",
                how="left",
                validate="m:1",
            )
        last = known.group_by("player_id").agg(
            *[pl.col(c).last().alias(f"last_observed__{c}") for c in columns],
            ((year - pl.col("season").last()) * 32 + origin - pl.col("week").last()).alias(
                "last_observed__calendar_recency"
            ),
        )
        chunks.append(block.join(last, on="player_id", how="left", validate="m:1"))
    return pl.concat(chunks, how="diagonal_relaxed").sort(KEYS)


def prior_rate_baseline(panel: pl.DataFrame, box: pl.DataFrame) -> np.ndarray:
    """Calendar-week persistence; recorded unknowns remain unknown, absence is zero."""
    parts = []
    for (year, origin), block in panel.partition_by(["year", "origin"], as_dict=True).items():
        history = box.filter(
            (pl.col("season") == (year if origin else year - 1))
            & ((pl.col("week") <= origin) if origin else pl.lit(True))
        )
        totals = history.group_by("player_id").agg(
            pl.len().alias("baseline_records"),
            *[
                pl.when(pl.col(stat).null_count() == 0)
                .then(pl.col(stat).sum())
                .otherwise(None)
                .alias(f"baseline_{stat}")
                for stat in STATS
            ],
        )
        attached = block.select(*KEYS, "weeks", "history_weeks").join(
            totals, on="player_id", how="left", validate="m:1"
        )
        parts.append(
            attached.select(
                *KEYS,
                *[
                    (
                        pl.when(pl.col("baseline_records").is_null())
                        .then(0)
                        .otherwise(pl.col(f"baseline_{stat}"))
                        / pl.col("history_weeks")
                        * pl.col("weeks")
                    ).alias(stat)
                    for stat in STATS
                ],
            )
        )
    joined = panel.select(KEYS).join(
        pl.concat(parts), on=KEYS, how="left", validate="1:1", maintain_order="left"
    )
    return joined.select(STATS).to_numpy().astype(np.float64)


def _contract(horizon: str, origins: tuple[int, ...], first_year: int | None) -> dict:
    if horizon not in {"season", "remaining", "week"}:
        raise ValueError(f"Unsupported horizon: {horizon}")
    if not origins or any(origin < 1 or origin > 17 for origin in origins):
        raise ValueError("Origins must be completed regular-season weeks from 1 to 17")
    if raw.digest(raw.SNAPSHOT) != RAW_MANIFEST_SHA256:
        raise ValueError("Pinned canonical raw snapshot manifest changed")
    return {
        "schema_version": 1,
        "position": "RB",
        "horizon": horizon,
        "origins": list(origins),
        "first_year": first_year,
        "last_complete_season": raw.LAST_COMPLETE_SEASON,
        "raw_manifest_sha256": RAW_MANIFEST_SHA256,
        "source_sha256": {
            str(p.relative_to(raw.ROOT)): raw.digest(p)
            for p in (Path(__file__), Path(raw.__file__))
        },
        "extended_history": ["prior2", "career", "recent8", "last_observed"],
        "target_unknown_policy": "Per-stat NaN mask; no whole-row outcome filtering",
        "baseline_unknown_policy": "Recorded null stays NaN; absent historical row is zero",
        "preselected_features_used": False,
    }


def _read_cache(directory: Path, contract: dict) -> Dataset:
    meta = json.loads((directory / "dataset.json").read_text())
    if meta["contract"] != contract:
        raise ValueError("Dataset contract changed")
    for name, expected in meta["files"].items():
        if raw.digest(directory / name) != expected:
            raise ValueError(f"Cached transformed raw input changed: {name}")
    panel = pl.read_parquet(directory / "panel.parquet")
    names, groups = meta["names"], meta["groups"]
    validate_features(names)
    x = panel.select(names).to_numpy().astype(np.float32)
    x[~np.isfinite(x)] = np.nan
    return Dataset(
        panel=panel,
        x=x,
        names=names,
        groups=groups,
        targets=panel.select([f"y_{stat}" for stat in STATS]).to_numpy().astype(np.float64),
        exposure=panel["weeks"].to_numpy().astype(np.float64),
        baseline=np.load(directory / "baseline.npy", allow_pickle=False),
        audit=pl.read_parquet(directory / "audit.parquet"),
        provenance={**meta, "cache_directory": str(directory)},
        views={
            "all": list(range(len(names))),
            **{
                family: [i for i, group in enumerate(groups) if group == family]
                for family in sorted(set(groups))
            },
        },
    )


def load_dataset(
    horizon: str,
    origins: tuple[int, ...] = (2,),
    cache_dir: Path | str = DEFAULT_CACHE,
    *,
    first_year: int | None = None,
    progress=print,
) -> Dataset:
    """Build once from verified raw bytes, then reuse a sealed transformed-input cache."""
    origins = tuple(sorted(set(origins)))
    contract = _contract(horizon, origins, first_year)
    key = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()[:16]
    parent = Path(cache_dir)
    directory = parent / f"{horizon}_{key}"
    if directory.exists():
        return _read_cache(directory, contract)
    started = time.monotonic()
    progress(f"Building unselected raw RB/{horizon} panel")
    panel, names, audit = raw.make_panel("RB", horizon, origins, first_year, drop_unknown=False)
    originals = set(panel.columns)
    observations = raw.load_raw()
    panel = add_history_views(panel, observations["player"])
    names += [name for name in panel.columns if name not in originals]
    validate_features(names)
    if panel.select(KEYS).is_duplicated().any():
        raise ValueError("Duplicate RB forecast key")
    baseline = prior_rate_baseline(panel, observations["box"])
    groups = [source_family(name) for name in names]
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{horizon}_", dir=parent))
    try:
        panel.write_parquet(temporary / "panel.parquet")
        audit.write_parquet(temporary / "audit.parquet")
        raw.catalog().write_parquet(temporary / "raw_catalog.parquet")
        np.save(temporary / "baseline.npy", baseline)
        meta = {
            "contract": contract,
            "names": names,
            "groups": groups,
            "rows": panel.height,
            "columns": len(names),
            "years": sorted(panel["year"].unique().to_list()),
            "unique_players": panel["player_id"].n_unique(),
            "unknown_targets": {stat: panel[f"y_{stat}"].null_count() for stat in STATS},
            "unknown_baselines": {
                stat: int(np.isnan(baseline[:, i]).sum()) for i, stat in enumerate(STATS)
            },
            "build_seconds": time.monotonic() - started,
            "notes": [
                "Historical provider vintages are retrospective, not archived as-of snapshots.",
                "Candidates use past NFL observations and exactly linked drafted entrants.",
                "Undrafted or unlinked newcomers can be absent before their first NFL observation.",
                "Team is last observed, which can miss offseason or in-season moves.",
                "Recent8 is the last eight observed weekly rows across season boundaries.",
                "Calendar recency uses a fixed 32-slot season index, not elapsed real weeks.",
                "Career/prior2 means use recorded observations; missing remains missing.",
                "Future closing lines and all future production are excluded from predictors.",
                "Current-season 2023+ participation is masked: FTN publishes after postseason.",
                "No 2026 outcomes and no prepared or previously screened feature matrices used.",
                "Auxiliary FTN and BDB tracking releases are outside this canonical snapshot.",
            ],
            "files": {p.name: raw.digest(p) for p in sorted(temporary.iterdir())},
        }
        (temporary / "dataset.json").write_text(json.dumps(meta, indent=2))
        temporary.rename(directory)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    progress(
        f"Built {panel.height:,} RB rows × {len(names):,} raw-derived features in "
        f"{time.monotonic() - started:.1f}s: {directory}"
    )
    return _read_cache(directory, contract)
