"""Stable full-result ordering before slicing server-side pages."""

import polars as pl
from fastapi import HTTPException


def sorted_page(frame, *, sort, allowed, default, tie, offset, limit):
    if offset < 0:
        raise HTTPException(422, "Offset must be nonnegative")
    token = sort or default
    field = token.removeprefix("-")
    if field not in allowed or field not in frame.columns:
        raise HTTPException(422, "Unknown sort column")
    ordering = pl.col(field)
    if isinstance(frame.schema[field], pl.List):
        ordering = ordering.list.join(", ")
    keys = [ordering, *[pl.col(k) for k in tie if k != field]]
    return frame.sort(
        keys, descending=[token.startswith("-"), *[False] * (len(keys) - 1)], nulls_last=True
    ).slice(offset, limit)
