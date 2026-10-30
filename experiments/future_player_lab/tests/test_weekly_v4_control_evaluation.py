import json
from datetime import UTC, datetime

import pandas as pd
from experiments.future_player_lab.campaigns.weekly_v4 import control_evaluation as controls
from experiments.future_player_lab.campaigns.weekly_v31.artifacts import digest, seal, verify, write


def fixture(tmp_path, monkeypatch, capture_hash="base", weekly_mean=5):
    source = tmp_path / "forecasts/2026_08/early_week"
    (source / "baseline").mkdir(parents=True)
    pd.DataFrame(
        [dict(player_id="p", season=2026, week=8, position="QB", prediction=5, samples=[4, 6])]
    ).to_parquet(source / "baseline/players.parquet", index=False)
    projection = {h: [dict(player_id="p")] for h in ("weekly", "remaining", "season")}
    write(source / "projections.json", projection)
    seal(
        source,
        dict(
            status="prospective",
            season=2026,
            week=8,
            forecast_horizon="early_week",
            base_capture_sha256="base",
            evidence_cutoff="2026-10-25T10:00:00+00:00",
            generated_at="2026-10-26T10:00:00+00:00",
            deadline="2026-10-30T00:00:00+00:00",
        ),
    )
    registry = tmp_path / "registry"
    root = registry / "opening_drive_v32"
    root.mkdir(parents=True)
    models = []
    for horizon in projection:
        name = f"qb_{horizon}"
        row = dict(
            player_id="p",
            position="QB",
            season=2026,
            origin_week=7,
            predictions=dict(points=weekly_mean if horizon == "weekly" else 50),
        )
        if horizon == "weekly":
            row["distribution"] = dict(samples=[4, 6])
        path = root / f"{name}.json"
        write(
            path,
            dict(
                campaign_id="opening_drive_v32",
                season=2026,
                origin_week=7,
                capture_hashes={"source": capture_hash},
                rows=[row],
            ),
        )
        models.append(
            dict(
                model_id=name,
                forecasts=[dict(season=2026, origin_week=7, path=path.name, sha256=digest(path))],
            )
        )
    write(root / "manifest.json", dict(models=models))

    class Store:
        def __init__(self, _):
            pass

        def campaign(self, campaign):
            assert campaign == "opening_drive_v32"
            return dict(models=models, manifest_sha256=digest(root / "manifest.json"))

        def forecast(self, campaign, model_id, season, origin):
            assert (campaign, season, origin) == ("opening_drive_v32", 2026, 7)
            value = json.loads((root / f"{model_id}.json").read_text())
            return dict(rows=value.pop("rows"), metadata=value)

    monkeypatch.setattr(controls, "CampaignStore", Store)
    monkeypatch.setattr(controls, "now", lambda: datetime(2026, 10, 27, tzinfo=UTC))
    return source, registry, tmp_path / "log"


def test_control_copies_baseline_identity_and_is_immutable(tmp_path, monkeypatch):
    source, registry, log = fixture(tmp_path, monkeypatch)
    before = digest(source / "manifest.json")
    result = controls.capture_controls(source, registry, log)
    assert result["status"] == "captured"
    from pathlib import Path

    destination = Path(result["path"])
    receipt = verify(destination)
    assert receipt["source_campaign"] == "opening_drive_v32"
    assert receipt["paired_capture_sha256"] == before
    assert digest(source / "manifest.json") == before
    raw = json.loads((destination / "sources/qb_remaining.json").read_text())
    assert raw["campaign_id"] == "opening_drive_v32"
    monkeypatch.setattr(controls, "now", lambda: datetime(2026, 11, 1, tzinfo=UTC))
    assert controls.capture_controls(source, registry, log)["status"] == "current"
    assert verify(destination) == receipt


def test_control_never_backfills_after_deadline(tmp_path, monkeypatch):
    source, registry, log = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(controls, "now", lambda: datetime(2026, 11, 1, tzinfo=UTC))
    result = controls.capture_controls(source, registry, log)
    assert result["reason"] == "missed_prospective_deadline"
    assert not (log / "controls").exists()
    assert (log / "control_status/2026_08/early_week/status.json").exists()


def test_control_requires_same_capture_and_unchanged_weekly_forecast(tmp_path, monkeypatch):
    first = tmp_path / "wrong_capture"
    first.mkdir()
    source, registry, log = fixture(first, monkeypatch, capture_hash="other")
    assert controls.capture_controls(source, registry, log)["reason"] == "baseline_capture_mismatch"
    second = tmp_path / "replaced_weekly"
    second.mkdir()
    source, registry, log = fixture(second, monkeypatch, weekly_mean=9)
    assert (
        controls.capture_controls(source, registry, log)["reason"]
        == "baseline_weekly_forecast_replaced"
    )
    assert not (log / "controls").exists()
