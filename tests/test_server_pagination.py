import polars as pl
import pytest
from fastapi import HTTPException

from engine.api.pagination import sorted_page


def test_global_sort_is_stable_and_null_last_before_pagination():
    frame = pl.DataFrame({"id": ["d", "b", "c", "a"], "score": [None, 2, 4, 2]})

    def page(offset, sort="-score"):
        return sorted_page(
            frame, sort=sort, allowed={"score"}, default="score", tie=["id"], offset=offset, limit=2
        )["id"].to_list()

    assert page(0) == ["c", "a"]
    assert page(2) == ["b", "d"]
    assert page(0, "score") == ["a", "b"]
    with pytest.raises(HTTPException):
        page(0, "private_column")
    with pytest.raises(HTTPException):
        page(-1)
