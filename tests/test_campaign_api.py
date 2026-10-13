"""Campaign isolation, immutable publication, cutoff contracts, and corruption checks."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from engine.api import chairstats_routes
from engine.chairstats import CampaignStore, publish_campaign, replace_campaign


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    metrics = tmp_path / "metrics.json"
    metrics.write_text(json.dumps({"rmse": 42.0, "holdout_seasons": [2021, 2022]}))
    forecasts = tmp_path / "forecast.json"
    rows = [
        {
            "player_id": pid,
            "player_name": name,
            "position": "QB",
            "season": 2026,
            "origin_week": 2,
            "predictions": {"passing_yards": yards},
        }
        for pid, name, yards in [("p1", "Alpha", 3000.0), ("p2", "Beta", 2000.0)]
    ]
    forecasts.write_text(json.dumps({"rows": rows}))
    weights = tmp_path / "model.joblib"
    weights.write_bytes(b"not executable: API must never deserialize this")
    root = tmp_path / "campaigns"
    model = {
        "model_id": "qb_remaining",
        "position": "QB",
        "horizon": "remaining",
        "origin_week": 2,
        "targets": ["passing_yards"],
        "training_through_season": 2025,
        "metrics": metrics,
        "forecasts": [{"season": 2026, "origin_week": 2, "path": forecasts}],
        "artifacts": {"model": weights},
    }
    publish_campaign(root, {"campaign_id": "opening_drive_v2", "name": "Opening Drive"}, [model])
    monkeypatch.setattr(
        chairstats_routes, "get_settings", lambda: SimpleNamespace(data_dir=tmp_path)
    )
    app = FastAPI()
    app.include_router(chairstats_routes.router)
    return root, model, rows, TestClient(app)


BASE = "/api/chairstats/campaigns/opening_drive_v2"


def _reseal(path):
    (path.parent / "manifest.sha256").write_text(hashlib.sha256(path.read_bytes()).hexdigest())


def test_list_metadata_metrics_and_player_query(bundle):
    root, model, rows, client = bundle
    assert client.get("/api/chairstats/campaigns").json()["campaigns"][0]["name"] == "Opening Drive"
    detail = client.get(BASE)
    assert detail.headers["X-Artifact-Disposition"] == "research_preview"
    token = detail.headers["X-Campaign-Manifest"]
    assert client.get(f"{BASE}/models").json()["models"][0]["targets"] == ["passing_yards"]
    assert client.get(f"{BASE}/models/qb_remaining/metrics").json()["metrics"]["rmse"] == 42
    result = client.get(
        f"{BASE}/models/qb_remaining/forecasts",
        params={"season": 2026, "origin_week": 2, "player_id": "p2"},
        headers={"X-Campaign-Manifest": token},
    )
    assert result.status_code == 200
    assert result.json()["rows"] == [rows[1]]
    assert result.json()["total"] == 1
    assert CampaignStore(root).check("opening_drive_v2")["verified_artifacts"] == 3


def test_snapshot_and_pagination_are_explicit(bundle):
    _, _, rows, client = bundle
    path = f"{BASE}/models/qb_remaining/forecasts"
    assert client.get(path).status_code == 422
    assert client.get(path, params={"season": 2026, "origin_week": 4}).status_code == 404
    response = client.get(path, params={"season": 2026, "origin_week": 2, "offset": 1, "limit": 1})
    assert response.json()["total"] == 2
    assert response.json()["rows"] == rows[1:]
    assert (
        client.get(path, params={"season": 2026, "origin_week": 2, "limit": 1001}).status_code
        == 422
    )


def test_campaigns_never_fall_back_to_another_campaign(bundle):
    root, model, _, client = bundle
    other = {**model, "model_id": "qb_other"}
    publish_campaign(root, {"campaign_id": "second_drive_v1", "name": "Second Drive"}, [other])
    assert client.get(f"{BASE}/models/qb_other").status_code == 404
    assert client.get("/api/chairstats/campaigns/missing/models/qb_remaining").status_code == 404
    assert client.get(BASE, headers={"X-Campaign-Manifest": "old"}).status_code == 409


def test_forecast_corruption_invalidates_cached_response_immediately(bundle):
    root, _, _, client = bundle
    path = f"{BASE}/models/qb_remaining/forecasts?season=2026&origin_week=2"
    assert client.get(path).status_code == 200
    file = root / "opening_drive_v2/qb_remaining/forecasts/2026_2.json"
    file.write_text(file.read_text().replace("3000.0", "9999.0"))
    assert client.get(path).status_code == 409


def test_manifest_corruption_and_escape_are_rejected(bundle):
    root, _, _, client = bundle
    assert client.get(BASE).status_code == 200
    path = root / "opening_drive_v2/manifest.json"
    manifest = json.loads(path.read_text())
    manifest["models"][0]["metrics"]["path"] = "../metrics.json"
    path.write_text(json.dumps(manifest))
    assert client.get(BASE).status_code == 409
    _reseal(path)
    assert client.get(BASE).status_code == 409
    with pytest.raises(ValueError):
        CampaignStore(root).campaign("../opening_drive_v2")


def test_symlink_escape_rejected_after_cached_success(bundle, tmp_path):
    root, _, _, client = bundle
    path = f"{BASE}/models/qb_remaining/metrics"
    assert client.get(path).status_code == 200
    metrics = root / "opening_drive_v2/qb_remaining/metrics.json"
    metrics.unlink()
    metrics.symlink_to(tmp_path / "metrics.json")
    assert client.get(path).status_code == 409


def test_publish_is_create_only_and_source_cleanup_is_safe(bundle):
    root, model, _, client = bundle
    with pytest.raises(FileExistsError):
        publish_campaign(root, {"campaign_id": "opening_drive_v2", "name": "Changed"}, [model])
    model["metrics"].unlink()
    model["forecasts"][0]["path"].unlink()
    model["artifacts"]["model"].unlink()
    assert client.get(f"{BASE}/models/qb_remaining/metrics").status_code == 200
    assert CampaignStore(root).check("opening_drive_v2")["status"] == "verified"


@pytest.mark.parametrize("change", ["target", "position", "duplicate", "nonfinite", "training"])
def test_invalid_contract_never_publishes(bundle, change):
    root, model, rows, _ = bundle
    if change == "target":
        rows[0]["predictions"] = {"rushing_yards": 10}
    elif change == "position":
        rows[0]["position"] = "WR"
    elif change == "duplicate":
        rows.append(rows[0])
    elif change == "nonfinite":
        rows[0]["predictions"]["passing_yards"] = float("nan")
    else:
        model["training_through_season"] = 2026
    model["forecasts"][0]["path"].write_text(json.dumps(rows))
    with pytest.raises(ValueError):
        publish_campaign(root, {"campaign_id": "invalid", "name": "Invalid"}, [model])
    assert not (root / "invalid").exists()


def test_binary_artifacts_are_hashed_but_not_loaded(bundle):
    root, _, _, client = bundle
    file = root / "opening_drive_v2/qb_remaining/artifacts/model.joblib"
    file.write_bytes(b"corrupt weight")
    # Saved JSON remains inspectable; full check is the explicit weight audit.
    assert client.get(f"{BASE}/models/qb_remaining/metrics").status_code == 200
    with pytest.raises(ValueError, match="Saved model artifact checksum"):
        CampaignStore(root).check("opening_drive_v2")


def test_forecast_receipt_passes_through_without_duplicate_rows(bundle, monkeypatch):
    root, model, rows, client = bundle
    receipt = {
        "campaign_id": "receipt_campaign",
        "model_id": "qb_remaining",
        "season": 2026,
        "origin_week": 2,
        "horizon": "remaining",
        "forecast_kind": "reconstructed_cutoff_from_later_capture",
        "scoring": {"passing_yards": 0.04},
        "generated_at": "2026-10-09T00:00:00Z",
        "training_through_season": 2025,
        "snapshot": {"manifest_sha256": "a" * 64},
        "supplements": {"sources": ["retained_ngs"]},
        "notes": ["Not an originally issued forecast."],
    }
    model["forecasts"][0]["path"].write_text(json.dumps({**receipt, "rows": rows}))
    publish_campaign(root, {"campaign_id": "receipt_campaign", "name": "Receipt"}, [model])
    monkeypatch.setattr(chairstats_routes, "store", lambda: CampaignStore(root))
    response = client.get(
        "/api/chairstats/campaigns/receipt_campaign/models/qb_remaining/forecasts",
        params={"season": 2026, "origin_week": 2, "limit": 1},
    )
    assert response.status_code == 200
    assert response.json()["metadata"] == receipt
    assert response.json()["rows"] == rows[:1]


@pytest.mark.parametrize(
    "key,value",
    [
        ("campaign_id", "wrong"),
        ("origin_week", 4),
        ("model_id", "other"),
        ("training_through_season", 2026),
    ],
)
def test_inconsistent_receipt_never_publishes(bundle, key, value):
    root, model, rows, _ = bundle
    model["forecasts"][0]["path"].write_text(json.dumps({key: value, "rows": rows}))
    with pytest.raises(ValueError, match="receipt"):
        publish_campaign(root, {"campaign_id": "invalid_receipt", "name": "Invalid"}, [model])
    assert not (root / "invalid_receipt").exists()


def replacement_fixture(bundle):
    root, model, rows, client = bundle
    candidate_root = root.parent / "replacement_candidates"
    archive_root = root.parent / "campaign_revisions"
    old = CampaignStore(root).campaign("opening_drive_v2")["manifest_sha256"]
    rows[0]["predictions"]["passing_yards"] = 4000.0
    model["forecasts"][0]["path"].write_text(json.dumps(rows))
    publish_campaign(
        candidate_root, {"campaign_id": "opening_drive_v2", "name": "Revised"}, [model]
    )
    return root, candidate_root, archive_root, old, client


def test_replace_campaign_preserves_archive_and_invalidates_old_api_pin(bundle):
    root, candidate_root, archive_root, old, client = replacement_fixture(bundle)
    original = (root / "opening_drive_v2/manifest.json").read_bytes()
    # Prime the existing verification cache before swapping the directory.
    assert client.get(BASE, headers={"X-Campaign-Manifest": old}).status_code == 200
    result = replace_campaign(root, candidate_root, "opening_drive_v2", old, archive_root)
    assert result["status"] == "activated"
    assert result["campaign_id"] == "opening_drive_v2"
    assert (archive_root / "opening_drive_v2" / old / "manifest.json").read_bytes() == original
    assert (candidate_root / "opening_drive_v2/manifest.json").exists()
    assert client.get(BASE, headers={"X-Campaign-Manifest": old}).status_code == 409
    assert (
        client.get(BASE, headers={"X-Campaign-Manifest": result["new_manifest_sha256"]}).status_code
        == 200
    )
    listed = client.get("/api/chairstats/campaigns").json()["campaigns"]
    assert [row["campaign_id"] for row in listed] == ["opening_drive_v2"]
    assert listed[0]["name"] == "Revised"
    assert CampaignStore(root).check("opening_drive_v2")["status"] == "verified"
    receipt = json.loads(Path(result["receipt_path"]).read_text())
    assert receipt["old_manifest_sha256"] == old
    assert not list(root.glob(".campaign-replace-*"))


@pytest.mark.parametrize("failure", ["stale_pin", "tampered_candidate", "archive_inside_registry"])
def test_replace_rejects_invalid_input_without_changing_active(bundle, failure):
    root, candidates, archive, old, _ = replacement_fixture(bundle)
    expected = old
    if failure == "stale_pin":
        expected = "0" * 64
    elif failure == "tampered_candidate":
        (candidates / "opening_drive_v2/qb_remaining/metrics.json").write_text("{}")
    else:
        archive = root / "old_revisions"
    with pytest.raises(ValueError):
        replace_campaign(root, candidates, "opening_drive_v2", expected, archive)
    assert CampaignStore(root).check("opening_drive_v2")["manifest_sha256"] == old
    assert not list(root.glob(".campaign-replace-*"))


def test_replace_rolls_back_if_activation_receipt_fails(bundle, monkeypatch):
    from engine.chairstats import store

    root, candidates, archive, old, _ = replacement_fixture(bundle)
    write = store._revision_receipt

    def fail_after_exchange(path, receipt):
        if receipt["status"] == "activated":
            assert CampaignStore(root).campaign("opening_drive_v2")["manifest_sha256"] != old
            raise OSError("simulated receipt failure")
        write(path, receipt)

    monkeypatch.setattr(store, "_revision_receipt", fail_after_exchange)
    with pytest.raises(OSError, match="simulated receipt failure"):
        replace_campaign(root, candidates, "opening_drive_v2", old, archive)
    assert CampaignStore(root).check("opening_drive_v2")["manifest_sha256"] == old
    receipt = json.loads(next((archive / "opening_drive_v2").glob("revision_*.json")).read_text())
    assert receipt["status"] == "rolled_back"
    assert (archive / "opening_drive_v2" / old / "manifest.json").exists()
    assert not list(root.glob(".campaign-replace-*"))


def test_failed_atomic_exchange_keeps_active_and_candidate(bundle, monkeypatch):
    from engine.chairstats import store

    root, candidates, archive, old, _ = replacement_fixture(bundle)

    def unsupported(*args):
        raise OSError("rename exchange unavailable")

    monkeypatch.setattr(store, "_exchange_directories", unsupported)
    with pytest.raises(OSError, match="exchange unavailable"):
        replace_campaign(root, candidates, "opening_drive_v2", old, archive)
    assert CampaignStore(root).check("opening_drive_v2")["manifest_sha256"] == old
    assert CampaignStore(candidates).check("opening_drive_v2")["status"] == "verified"
    receipt = json.loads(next((archive / "opening_drive_v2").glob("revision_*.json")).read_text())
    assert receipt["status"] == "aborted"
