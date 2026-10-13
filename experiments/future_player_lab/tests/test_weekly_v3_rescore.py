import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from experiments.future_player_lab.campaigns import rescore as driver
from experiments.future_player_lab.campaigns.weekly_v3 import data, modeling, rescore

from engine.chairstats import CampaignStore
from engine.chairstats.store import publish_campaign


def live_frame(week=6):
    return pd.DataFrame(
        {
            "player_id": ["active", "bye"],
            "player_name": ["Active", "Bye"],
            "position": ["QB", "QB"],
            "season": [2026, 2026],
            "week": [week, week],
            "team": ["BUF", "KC"],
            "opponent": ["NE", None],
            "scheduled_game": [1, 0],
            "is_bye": [0, 1],
            "target_points": [np.nan, np.nan],
            "v2_exact_available": [1.0, 1.0],
            "recent4_points_mean": [12.0, 15.0],
            "player_history_mean": [10.0, 10.0],
            "player_history_count": [20.0, 20.0],
        }
    )


@pytest.fixture
def sealed(tmp_path):
    src = tmp_path / "input"
    src.mkdir()
    registry = tmp_path / "registry"
    mean = {
        "method": "recent",
        "features": ["recent4_points_mean", "player_history_mean", "player_history_count"],
    }
    cal = pd.DataFrame(
        {
            "season": [2024, 2024, 2025, 2025],
            "prediction": [3.0, 5.0, 7.0, 9.0],
            "target_points": [1.0, 6.0, 5.0, 13.0],
        }
    )
    spread = modeling.fit_distribution(cal, ["player_history_mean"], "pooled", 2026)
    (src / "mean.pkl").write_bytes(pickle.dumps(mean))
    (src / "spread.pkl").write_bytes(pickle.dumps(spread))
    (src / "run.json").write_text(json.dumps({"packages": {}}))
    (src / "metrics.json").write_text("{}")
    (src / "old.json").write_text(
        json.dumps(
            [
                dict(
                    player_id="old",
                    position="QB",
                    season=2026,
                    origin_week=0,
                    predictions={"points": 1.0},
                    rank=1,
                )
            ]
        )
    )
    publish_campaign(
        registry,
        dict(campaign_id="opening_drive_v3", name="V3 test"),
        [
            dict(
                model_id="qb_weekly",
                position="QB",
                horizon="weekly",
                origin_week=0,
                targets=["points"],
                training_through_season=2025,
                metrics=src / "metrics.json",
                forecasts=[dict(season=2026, origin_week=0, path=src / "old.json")],
                artifacts=dict(
                    mean_model=src / "mean.pkl",
                    distribution_model=src / "spread.pkl",
                    run_manifest=src / "run.json",
                    source_data=Path(data.__file__),
                    source_modeling=Path(modeling.__file__),
                ),
            )
        ],
    )
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text(
        json.dumps(
            dict(
                status="accepted",
                current_observations=dict(
                    season=2026, through_week=5, saved_at="2026-10-13T12:00:00Z"
                ),
            )
        )
    )
    return registry, snapshot, tmp_path / "archive"


def fake_builder(snapshot, sources, registry, output_dir):
    frame = live_frame()
    path = output_dir / "panel.parquet"
    frame.to_parquet(path, index=False)
    receipt = {"panel_sha256": data.digest(path), "features": list(frame.columns)}
    (output_dir / "dataset.json").write_text(json.dumps(receipt))
    return frame, receipt


def test_refresh_appends_archives_preserves_every_existing_artifact_and_is_idempotent(
    sealed, monkeypatch
):
    registry, snapshot, archive = sealed
    bundle = registry / "opening_drive_v3"
    before = {str(p.relative_to(bundle)): p.read_bytes() for p in bundle.rglob("*") if p.is_file()}
    monkeypatch.setattr(rescore, "build_live_panel", fake_builder)
    store = CampaignStore(registry)
    result = rescore.rescore_weekly(
        store,
        store.campaign("opening_drive_v3"),
        season=2026,
        origin=5,
        snapshot=snapshot,
        sources=None,
        registry=registry,
        archive=archive,
    )
    assert result["status"] == "rescored"
    for name, content in before.items():
        if name not in ("manifest.json", "manifest.sha256"):
            assert (bundle / name).read_bytes() == content
    assert any(p.read_bytes() == before["manifest.json"] for p in archive.rglob("manifest.json"))
    store = CampaignStore(registry)
    store.check("opening_drive_v3")
    manifest = store.campaign("opening_drive_v3")
    forecast = next(f for f in manifest["models"][0]["forecasts"] if f["origin_week"] == 5)
    assert forecast["feature_matrix_artifact"] == "forecast_matrix_2026_5"
    rows = store.forecast("opening_drive_v3", "qb_weekly", 2026, 5)["rows"]
    assert {r["player_id"] for r in rows} == {"active", "bye"}
    bye = next(r for r in rows if r["player_id"] == "bye")
    assert bye["predictions"]["points"] == bye["distribution"]["std"] == 0
    assert bye["distribution"]["samples"] == [0] * 256
    preserved = (bundle / "manifest.json").read_bytes()
    monkeypatch.setattr(rescore, "build_live_panel", lambda *a: pytest.fail("Duplicate rebuild"))
    assert (
        rescore.rescore_weekly(
            store,
            manifest,
            season=2026,
            origin=5,
            snapshot=snapshot,
            sources=None,
            registry=registry,
            archive=archive,
        )["status"]
        == "current"
    )
    assert (bundle / "manifest.json").read_bytes() == preserved


def test_future_origin_or_new_season_fails_before_build(sealed, monkeypatch):
    registry, snapshot, archive = sealed
    monkeypatch.setattr(rescore, "build_live_panel", lambda *a: pytest.fail("Unexpected build"))
    store = CampaignStore(registry)
    for season, origin in [(2026, 6), (2027, 5)]:
        with pytest.raises(ValueError):
            rescore.rescore_weekly(
                store,
                store.campaign("opening_drive_v3"),
                season=season,
                origin=origin,
                snapshot=snapshot,
                sources=None,
                registry=registry,
                archive=archive,
            )


def test_observed_outcome_or_duplicate_identity_cannot_publish(sealed, monkeypatch):
    registry, snapshot, archive = sealed
    store = CampaignStore(registry)
    before = (registry / "opening_drive_v3/manifest.json").read_bytes()
    for defect in ("observed", "duplicate"):

        def broken(*args, defect=defect):
            frame, receipt = fake_builder(*args)
            if defect == "observed":
                frame.loc[0, "target_points"] = 4.0
            else:
                frame.player_id = "same"
            return frame, receipt

        monkeypatch.setattr(rescore, "build_live_panel", broken)
        with pytest.raises(ValueError, match="observed|Duplicate"):
            rescore.rescore_weekly(
                store,
                store.campaign("opening_drive_v3"),
                season=2026,
                origin=5,
                snapshot=snapshot,
                sources=None,
                registry=registry,
                archive=archive,
            )
        assert (registry / "opening_drive_v3/manifest.json").read_bytes() == before


def test_runtime_source_overrides_restore_on_builder_error(tmp_path, monkeypatch):
    original = data.SNAPSHOT, data.AVAILABILITY, data.CAMPAIGN

    def broken(**kwargs):
        assert tmp_path / "new.json" == data.SNAPSHOT
        assert tmp_path / "sources" == data.AVAILABILITY
        raise RuntimeError("source failed")

    monkeypatch.setattr(data, "build_panel", broken)
    with pytest.raises(RuntimeError, match="source failed"):
        rescore.build_live_panel(tmp_path / "new.json", tmp_path / "sources", tmp_path, tmp_path)
    assert original == (data.SNAPSHOT, data.AVAILABILITY, data.CAMPAIGN)


def test_driver_refreshes_v3_when_v2_is_already_current(sealed, monkeypatch):
    registry, snapshot, archive = sealed
    fixture_root = snapshot.parent
    current = fixture_root / "v2_current.json"
    current.write_text(
        json.dumps(
            [
                dict(
                    player_id="v2",
                    position="QB",
                    season=2026,
                    origin_week=5,
                    predictions={"points": 50.0},
                    rank=1,
                )
            ]
        )
    )
    publish_campaign(
        registry,
        dict(campaign_id=driver.LATEST, name="V2 current"),
        [
            dict(
                model_id="qb_remaining",
                position="QB",
                horizon="remaining",
                origin_week=2,
                targets=["points"],
                training_through_season=2025,
                metrics=fixture_root / "input/metrics.json",
                forecasts=[dict(season=2026, origin_week=5, rolling_origin=True, path=current)],
                artifacts={"weights": fixture_root / "input/mean.pkl"},
            )
        ],
    )
    monkeypatch.setattr(driver, "consolidate_opening_drive", lambda *a: None)
    monkeypatch.setattr(driver, "latest_supplements", lambda *a: Path("supplements"))
    monkeypatch.setattr(driver, "refresh_availability_sources", lambda *a: Path("sources"))
    monkeypatch.setattr(
        driver, "rescore_campaign", lambda *a, **kw: pytest.fail("V2 already current")
    )
    calls = []
    monkeypatch.setattr(
        rescore, "rescore_weekly", lambda *a, **kw: calls.append(kw) or {"status": "ok"}
    )
    assert (
        driver.main(
            [
                "--registry",
                str(registry),
                "--archive",
                str(archive),
                "--snapshot",
                str(snapshot),
                "--due",
            ]
        )
        == 0
    )
    assert len(calls) == 1 and calls[0]["origin"] == 5
