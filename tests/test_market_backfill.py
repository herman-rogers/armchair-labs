import polars as pl

from patron.data.market_backfill import (
    extend_crosswalk,
    extend_rankings,
    normalize_offense_pages,
)
from patron.metrics.enrichment import build_market_rankings


def _offense_archive() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "page_type": ["redraft-offense"] * 4 + ["redraft-qb"],
            "ecr_type": ["rp", "rp", "ro", "wo", "rp"],
            "id": ["1", "2", "1", "1", "3"],
            "pos": ["RB", "QB", "RB", "RB", "QB"],
            "ecr": [1.1, 1.5, 1.0, 9.0, 2.0],
            "sd": [0.2, 0.4, 0.1, 1.0, 0.3],
            "scrape_date": ["2020-08-13"] * 4 + ["2021-08-27"],
        }
    )


def test_offense_pages_normalize_to_canonical_page_types() -> None:
    shim = normalize_offense_pages(_offense_archive())

    assert shim.height == 3  # weekly (`wo`) rows and non-offense pages are dropped
    assert set(shim["page_type"].to_list()) == {"redraft-rb", "redraft-qb", "redraft-overall"}
    overall = shim.filter(pl.col("page_type") == "redraft-overall")
    assert overall["ecr"].to_list() == [1.0]


def test_extend_rankings_keeps_archive_and_appends_shim_and_backfill() -> None:
    backfill = pl.DataFrame(
        {
            "page_type": ["redraft-wr"],
            "id": ["-27656"],
            "gsis_id": ["00-0027656"],
            "player": ["Old Receiver"],
            "pos": ["WR"],
            "team": ["STL"],
            "ecr": [3.0],
            "sd": [None],
            "scrape_date": ["2012-08-29"],
        }
    ).with_columns(pl.col("sd").cast(pl.Float64))

    combined = extend_rankings(_offense_archive(), backfill=backfill)

    assert combined.filter(pl.col("page_type") == "redraft-wr").height == 1
    assert combined.filter(pl.col("page_type") == "redraft-offense").height == 4
    assert combined.filter(pl.col("page_type") == "redraft-rb").height == 1


def test_extend_crosswalk_appends_only_missing_identity_pairs() -> None:
    crosswalk = pl.DataFrame(
        {"fantasypros_id": [10], "gsis_id": ["00-0000010"], "name": ["Known Player"]}
    )
    backfill = pl.DataFrame(
        {
            "page_type": ["redraft-wr", "redraft-rb", "redraft-te"],
            "id": ["-27656", "10", None],
            "gsis_id": ["00-0027656", "00-0000010", None],
            "player": ["Old Receiver", "Known Player", "Ghost"],
            "pos": ["WR", "RB", "TE"],
            "team": ["STL", "KC", None],
            "ecr": [3.0, 1.0, 9.0],
            "sd": [None, None, None],
            "scrape_date": ["2012-08-29"] * 3,
        }
    ).with_columns(pl.col("sd").cast(pl.Float64))

    extended = extend_crosswalk(crosswalk, backfill=backfill)

    assert extended.height == 2  # known id kept once, null id dropped, synthetic added
    added = extended.filter(pl.col("fantasypros_id") == "-27656")
    assert added["gsis_id"].to_list() == ["00-0027656"]


def test_market_rankings_take_latest_snapshot_per_page_not_per_season() -> None:
    # Wayback captures carry different dates per page; a July WR page must survive
    # an August QB page in the same season.
    rankings = pl.DataFrame(
        {
            "page_type": ["redraft-qb", "redraft-wr", "redraft-wr"],
            "id": ["1", "2", "2"],
            "pos": ["QB", "WR", "WR"],
            "ecr": [1.0, 2.0, 3.0],
            "sd": [0.1, 0.2, 0.2],
            "scrape_date": ["2011-08-23", "2011-07-08", "2011-07-01"],
        }
    )
    crosswalk = pl.DataFrame({"fantasypros_id": [1, 2], "gsis_id": ["qb", "wr"]})

    rows = {r["player_id"]: r for r in build_market_rankings(rankings, crosswalk).to_dicts()}

    assert rows["qb"]["market_snapshot"] == "2011-08-23"
    assert rows["wr"]["market_ecr"] == 2.0
    assert rows["wr"]["market_snapshot"] == "2011-07-08"
