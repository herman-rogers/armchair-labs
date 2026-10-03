"""Regression gates for point-in-time data, immutable sources, and atomic publication."""

import json
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from engine.data.catalog import publish_catalog
from engine.data.gold import clean_floats, split_forecasts, unique, validate_forecasts
from engine.data.releases import (
    digest,
    load_gold,
    preserve_object,
    product_reference,
    reference,
    write_json,
)


def input_rows():
    return pl.DataFrame(
        {
            "player_id": ["hunt", "pending", "late-only"],
            "player_display_name": ["Kareem Hunt", "Pending Player", "Later Addition"],
            "forecast_season": [2019, 2026, 2019],
            "forecast_cutoff_date": [date(2019, 7, 17), date(2026, 8, 29), date(2019, 7, 17)],
            "source_season": [2018, 2025, 2018],
            "season": [2018, 2025, 2018],
            "team": ["KC", "CLE", None],
            "cutoff_preseason_team": ["CLE", "CLE", None],
            "player_population": ["returner", "returner", "market_only"],
            "market_snapshot": ["2019-07-23", "2026-08-29", "2019-07-23"],
            "market_ecr": [40.0, 10.0, 100.0],
            "market_overall_ecr": [80.0, 20.0, None],
            "availability_latest_known_on": [date(2019, 3, 15), None, None],
            "known_absence_games": [8, None, None],
            "known_available_games_cap": [8.0, None, None],
            "actual_games": [8.0, 0.0, 0.0],
            "actual_season_points": [90.0, 0.0, 0.0],
            "actual_matched": [True, False, False],
            "outcome_complete": [True, False, True],
            "fitted_season_points": [140.0, 200.0, None],
            "week1_proxy_team": ["CLE", "CLE", None],
            "contract_apy_cap_pct_proxy": [2.0, 3.0, None],
            "rich_s0_points_mean": [10.0, 20.0, None],
            "rich_s0_xfp_mean": [9.0, 19.0, None],
            "rich_future_points": [999.0, 999.0, 999.0],
        }
    )


def test_gold_separates_labels_predictions_and_late_evidence_without_inventing_facts():
    features, labels, report = split_forecasts(input_rows())
    assert features.height == labels.height == 2
    assert report["late_positional_market_rows_quarantined"] == 2
    assert report["quarantined_candidate_rows"] == 1
    hunt = features.filter(pl.col("player_id") == "hunt").row(0, named=True)
    assert hunt["source_team"] == "KC"
    assert hunt["cutoff_preseason_team"] == "CLE"
    assert hunt["known_available_games_cap"] == 8
    assert hunt["market_ecr"] is None
    assert hunt["market_overall_ecr"] == 80
    assert hunt["rich_s0_points_mean"] == 10
    for name in (
        "actual_games",
        "fitted_season_points",
        "week1_proxy_team",
        "contract_apy_cap_pct_proxy",
        "rich_s0_xfp_mean",
        "rich_future_points",
    ):
        assert name not in features.columns
    pending = labels.filter(pl.col("player_id") == "pending").row(0, named=True)
    assert pending["actual_games"] is None
    assert pending["actual_season_points"] is None
    assert pending["actual_matched"] is None
    assert pending["outcome_complete"] is False


@pytest.mark.parametrize(
    "change,match",
    [
        ({"availability_latest_known_on": date(2030, 1, 1)}, "after forecast cutoff"),
        ({"source_season": 2030}, "source season"),
        ({"known_available_games_cap": 16.0}, "absence and available-game"),
    ],
)
def test_gold_rejects_future_facts_and_inconsistent_absence_caps(change, match):
    frame = input_rows().with_columns(pl.lit(value).alias(name) for name, value in change.items())
    with pytest.raises(ValueError, match=match):
        split_forecasts(frame)


def test_duplicate_and_null_keys_are_not_silently_deduplicated():
    frame = input_rows()
    with pytest.raises(ValueError, match="duplicate"):
        split_forecasts(pl.concat([frame, frame.head(1)]))
    with pytest.raises(ValueError, match="null primary key"):
        unique(pl.DataFrame({"player_id": [None]}), ["player_id"], "players")
    features, labels, _ = split_forecasts(frame)
    with pytest.raises(ValueError, match="populations differ"):
        validate_forecasts(features, labels.head(1))


def test_undefined_ratios_are_null_not_zero_or_nonfinite():
    frame = pl.DataFrame({"share": [0.0, float("nan"), float("inf"), None]})
    clean, counts = clean_floats(frame)
    assert clean["share"].to_list() == [0.0, None, None, None]
    assert counts == {"share": 2}


def test_preserved_objects_are_deduplicated_copies_and_reject_tampering(tmp_path):
    source = tmp_path / "source.csv"
    source.write_bytes(b"original bytes\n")
    expected = digest(source)
    relative = preserve_object(tmp_path, source, expected)
    assert preserve_object(tmp_path, source, expected) == relative
    assert source.stat().st_ino != (tmp_path / relative).stat().st_ino
    source.write_bytes(b"new capture\n")
    assert (tmp_path / relative).read_bytes() == b"original bytes\n"
    with pytest.raises(ValueError, match="Input changed"):
        preserve_object(tmp_path, source, expected)
    (tmp_path / relative).write_bytes(b"tampered bytes\n")
    source.write_bytes(b"original bytes\n")
    with pytest.raises(ValueError, match="Raw object was altered"):
        preserve_object(tmp_path, source, expected)


def release_chain(data: Path):
    raw = data / "raw/snapshots/raw1"
    raw.mkdir(parents=True)
    source = data / "raw/source.txt"
    source.write_text("raw")
    write_json(
        raw / "manifest.json",
        {
            "version": "raw1",
            "layer": "raw",
            "schema_version": 1,
            "status": "accepted",
            "files": {"raw/source.txt": digest(source)},
        },
    )
    previous = raw
    for layer in ("enriched", "gold"):
        root = data / layer / "releases/v1"
        root.mkdir(parents=True)
        (root / "table.parquet").write_text("fixture")
        write_json(root / "quality.json", {"checks": {"fixture": True}})
        write_json(
            root / "manifest.json",
            {
                "version": "v1",
                "layer": layer,
                "schema_version": 1,
                "status": "accepted",
                "files": {
                    "table.parquet": digest(root / "table.parquet"),
                    "quality.json": digest(root / "quality.json"),
                },
                "input": reference(previous),
                "quality_passed": True,
                "history": {"version": "history"},
                "current_observations": {"season": 2026},
            },
        )
        previous = root
    return previous, source


@pytest.mark.parametrize("artifact", ["raw", "enriched", "gold"])
def test_reader_verifies_all_three_layers(tmp_path, artifact):
    gold, raw_source = release_chain(tmp_path)
    assert load_gold(tmp_path, "v1").root == gold
    target = {
        "raw": raw_source,
        "enriched": tmp_path / "enriched/releases/v1/table.parquet",
        "gold": gold / "table.parquet",
    }[artifact]
    target.write_text("changed")
    with pytest.raises(ValueError, match="artifact changed"):
        load_gold(tmp_path, "v1")


def test_table_batch_reader_preserves_transitive_verification(tmp_path):
    from engine.data.releases import load_tables

    legacy, raw_source = release_chain(tmp_path)
    batch = tmp_path / "tables/batches/v1"
    batch.parent.mkdir(parents=True)
    legacy.rename(batch)
    manifest = json.loads((batch / "manifest.json").read_text())
    manifest["layer"] = "tables"
    write_json(batch / "manifest.json", manifest)
    assert load_tables(tmp_path, "v1").root == batch
    # Historical product readers also resolve newly built batches.
    assert load_gold(tmp_path, "v1").ref == reference(batch)
    write_json(
        tmp_path / "current.json",
        {"schema_version": 1, "table_release": reference(batch), "products": {}},
    )
    assert load_tables(tmp_path).ref == reference(batch)
    raw_source.write_text("changed")
    with pytest.raises(ValueError, match="artifact changed"):
        load_tables(tmp_path, "v1")


def test_failed_publication_leaves_current_untouched_and_never_falls_back(tmp_path):
    gold, _ = release_chain(tmp_path)
    current = tmp_path / "current.json"
    write_json(current, {"schema_version": 1, "gold": reference(gold), "products": {}})
    before = current.read_bytes()
    product = tmp_path / "research/wrong"
    product.mkdir(parents=True)
    write_json(
        product / "manifest.json", {"history": {"version": "history"}, "gold": {"version": "older"}}
    )
    with pytest.raises(ValueError, match="Mixed data releases"):
        publish_catalog(
            tmp_path, "v1", {"college": "wrong", "profiles": "wrong", "outlook_2026": "wrong"}
        )
    assert current.read_bytes() == before
    legacy = tmp_path / "legacy.json"
    write_json(legacy, {"version": "wrong"})
    with pytest.raises(KeyError):
        product_reference(tmp_path, "profiles", legacy)
    current.write_text("broken")
    with pytest.raises(json.JSONDecodeError):
        product_reference(tmp_path, "profiles", legacy)


def test_frontend_catalog_token_rejects_stale_and_mid_request_releases(monkeypatch):
    import asyncio
    from types import SimpleNamespace

    from fastapi.responses import JSONResponse

    from engine.api import app as api

    catalog = {"gold": {"version": "v1"}, "published_at": "today"}
    monkeypatch.setattr(api, "current_catalog", lambda _: catalog)

    async def handler(_):
        return JSONResponse({"ok": True})

    def request(token):
        return asyncio.run(
            api.pin_data_catalog(SimpleNamespace(headers={"X-Data-Catalog": token}), handler)
        )

    response = request("old@today")
    assert response.status_code == 409
    assert response.headers["X-Data-Catalog-Stale"] == "true"
    assert request("v1@today").status_code == 200
    revisions = iter([catalog, {"gold": {"version": "v2"}, "published_at": "later"}])
    monkeypatch.setattr(api, "current_catalog", lambda _: next(revisions))
    response = request("v1@today")
    assert response.status_code == 409
    assert response.headers["X-Data-Catalog-Stale"] == "true"
