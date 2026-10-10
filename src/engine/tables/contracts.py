"""Table batch contracts: key uniqueness, finite quantities, and per-table specs."""

from __future__ import annotations

import polars as pl


def unique(frame: pl.DataFrame, keys: list[str], name: str) -> None:
    if not set(keys) <= set(frame.columns):
        raise ValueError(f"{name}: missing key columns {keys}")
    if frame.select(pl.any_horizontal(pl.col(keys).is_null()).any()).item():
        raise ValueError(f"{name}: null primary key")
    if frame.select(keys).is_duplicated().any():
        raise ValueError(f"{name}: duplicate primary key {keys}")


def clean_floats(frame: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, int]]:
    """Undefined source ratios are unknown, never zero, NaN, or infinity in accepted tables."""
    columns = [c for c, dtype in frame.schema.items() if dtype in (pl.Float32, pl.Float64)]
    if not columns:
        return frame, {}
    counts = frame.select(
        (pl.col(c).is_not_null() & ~pl.col(c).is_finite()).sum().alias(c) for c in columns
    ).row(0, named=True)
    changed = {c: n for c, n in counts.items() if n}
    return frame.with_columns(
        pl.when(pl.col(c).is_finite()).then(pl.col(c)).otherwise(None).alias(c) for c in changed
    ), changed


def table_spec(frame: pl.DataFrame, keys: list[str], role: str, description: str) -> dict:
    unique(frame, keys, description)
    # Counts are diagnostics, never an instruction to impute missing values.
    return {
        "rows": frame.height,
        "primary_key": keys,
        "role": role,
        "description": description,
        "schema": {c: str(t) for c, t in frame.schema.items()},
        "null_counts": frame.null_count().row(0, named=True),
    }
