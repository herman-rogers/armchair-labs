"""Exact draft identity and additive-population invariants for the v2 revision."""

import polars as pl
import pytest
from experiments.future_player_lab.campaigns.data import dated_entrants
from experiments.future_player_lab.campaigns.revision_data import append_preserving


def tables():
    return {
        "combine": pl.DataFrame(
            {
                "pfr_id": ["exact"],
                "season": [2026],
                "pos": ["QB"],
                "player_name": ["Same Name"],
                "draft_year": [None],
                "draft_round": [None],
                "draft_ovr": [None],
                "draft_team": [None],
            },
            schema_overrides={
                "draft_year": pl.Float64,
                "draft_round": pl.Float64,
                "draft_ovr": pl.Float64,
                "draft_team": pl.String,
            },
        ),
        "mapping": pl.DataFrame({"pfr_id": ["exact"], "player_id": ["rookie"]}),
        "identity": pl.DataFrame(
            {
                "gsis_id": ["rookie"],
                "pfr_id": ["exact"],
                "position": ["QB"],
                "draft_year": [2026],
                "draft_round": [1],
                "draft_pick": [13],
                "draft_team": ["LA"],
                "latest_team": ["FUTURE_TEAM"],
                "status": ["RET"],
                "last_season": [2040],
            }
        ),
    }


def test_missing_draft_fields_recovered_only_from_exact_dated_event():
    source = tables()
    original = source["combine"].clone()
    result = dated_entrants(source, 2026).row(0, named=True)
    assert result["draft_year"] == 2026
    assert result["draft_ovr"] == 13
    assert result["draft_round"] == 1
    assert result["draft_team"] == "LA"
    assert source["combine"].equals(original)
    assert dated_entrants(source, 2025)["draft_year"][0] is None
    changed = dict(source)
    changed["identity"] = source["identity"].with_columns(
        pl.lit("OTHER").alias("latest_team"),
        pl.lit("ACT").alias("status"),
        pl.lit(2099).alias("last_season"),
    )
    assert dated_entrants(changed, 2026).equals(dated_entrants(source, 2026))


@pytest.mark.parametrize(
    "column,value",
    [
        ("gsis_id", "other"),
        ("pfr_id", "different"),
        ("position", "WR"),
        ("draft_year", 2027),
        ("draft_round", 0),
        ("draft_pick", 0),
        ("draft_team", None),
    ],
)
def test_no_name_matching_future_event_or_invalid_event_fallback(column, value):
    source = tables()
    source["identity"] = source["identity"].with_columns(
        pl.lit(value, dtype=source["identity"].schema[column]).alias(column)
    )
    assert dated_entrants(source, 2026)["draft_year"][0] is None


def test_ambiguous_exact_identity_is_withheld():
    source = tables()
    extra = source["identity"].with_columns(pl.lit(99, pl.Int64).alias("draft_pick"))
    source["identity"] = pl.concat([source["identity"], extra])
    assert dated_entrants(source, 2026)["draft_year"][0] is None


def test_ambiguous_crosswalk_is_withheld():
    source = tables()
    source["mapping"] = pl.concat(
        [source["mapping"], pl.DataFrame({"pfr_id": ["exact"], "player_id": ["different"]})]
    )
    assert dated_entrants(source, 2026).is_empty()


def test_conflicting_known_combine_fact_blocks_fallback():
    source = tables()
    source["combine"] = source["combine"].with_columns(pl.lit(12.0).alias("draft_ovr"))
    assert dated_entrants(source, 2026)["draft_year"][0] is None


def test_past_known_combine_event_is_not_rewritten():
    source = tables()
    source["combine"] = source["combine"].with_columns(
        pl.lit(2024.0).alias("draft_year"),
        pl.lit(2024).alias("season"),
        pl.lit("ATL").alias("draft_team"),
    )
    row = dated_entrants(source, 2026).row(0, named=True)
    assert row["draft_year"] == 2024 and row["draft_team"] == "ATL"


def test_append_preserves_nulls_outcomes_and_values_exactly():
    base = pl.DataFrame(
        {
            "year": [2025],
            "origin": [2],
            "player_id": ["old"],
            "feature": [None],
            "y_points": [123.456],
        },
        schema_overrides={"feature": pl.Float64},
    )
    addition = pl.DataFrame(
        {"year": [2025], "origin": [2], "player_id": ["new"], "feature": [4.0], "y_points": [0.0]}
    )
    revised = append_preserving(base, addition)
    assert revised.filter(pl.col("player_id") == "old").equals(base)
    assert revised.height == 2
    with pytest.raises(ValueError, match="existing"):
        append_preserving(base, base)
    with pytest.raises(ValueError, match="incomplete"):
        append_preserving(base, addition.drop("feature"))
