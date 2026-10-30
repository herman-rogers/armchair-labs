import json
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest
from experiments.future_player_lab.campaigns.weekly_v31.artifacts import seal, verify
from experiments.future_player_lab.campaigns.weekly_v32 import capture


def payload(status="Out"):
    teams = [dict(id=str(i), injuries=[]) for i in range(32)]
    teams[0]["injuries"] = [
        dict(
            status=status,
            date="2026-10-01T00:00Z",
            athlete=dict(
                id="123",
                displayName="Example",
                team={"abbreviation": "ABC"},
                position={"abbreviation": "RB"},
            ),
        )
    ]
    return dict(season={"year": 2026}, timestamp="2026-10-30T12:00Z", injuries=teams)


class Response:
    def __init__(self, document):
        self.document = document
        self.content = json.dumps(document).encode()

    def raise_for_status(self):
        pass

    def json(self):
        return self.document


@pytest.fixture
def provider(tmp_path, monkeypatch):
    history = tmp_path / "history"
    history.mkdir()
    pd.DataFrame(dict(espn_id=["123"], gsis_id=["player"])).to_parquet(
        history / "identity.parquet", index=False
    )
    monkeypatch.setattr(capture, "HISTORY", history)
    monkeypatch.setattr(capture, "ROOT", tmp_path / "captures")
    state = dict(document=payload(), clock=datetime(2026, 10, 30, 12, tzinfo=UTC))

    def tick():
        state["clock"] += timedelta(minutes=30)
        return state["clock"]

    monkeypatch.setattr(capture, "now", tick)
    monkeypatch.setattr(capture.requests, "get", lambda *a, **k: Response(state["document"]))
    return state


def test_unchanged_capture_records_only_a_summary(provider):
    first = capture.capture(9)
    second = capture.capture(9)
    assert {p.name for p in first.iterdir()} >= {"response.json.gz", "observations.parquet"}
    assert {p.name for p in second.iterdir()} == {
        "summary.json",
        "manifest.json",
        "manifest.sha256",
    }
    summary = json.loads((second / "summary.json").read_text())
    assert summary["observations_from"] == first.name
    assert summary["observed_at"] > json.loads((first / "summary.json").read_text())["observed_at"]
    assert verify(second)["status"] == "unchanged_research_observation"
    # A third unchanged pass still points at the full capture, not the marker.
    third = capture.capture(9)
    assert json.loads((third / "summary.json").read_text())["observations_from"] == first.name
    # Identity is stored once, outside the per-capture directories.
    assert len(list((first.parent / capture.IDENTITY_STORE).iterdir())) == 1
    assert not (first / "identity.parquet").exists()


def test_changed_observations_write_a_full_capture(provider):
    first = capture.capture(9)
    provider["document"] = payload("Questionable")
    second = capture.capture(9)
    assert (second / "observations.parquet").exists()
    assert "observations_from" not in json.loads((second / "summary.json").read_text())
    assert capture.read_response(second) == provider["document"]
    # A different forecast week context is also a change.
    assert (capture.capture(10) / "observations.parquet").exists()
    assert verify(first)["status"] == "captured_research_observation"
    store = first.parent / capture.IDENTITY_STORE
    assert len(list(store.iterdir())) == 1


def test_legacy_uncompressed_capture_is_readable_and_deduplicated(provider):
    root = capture.ROOT
    legacy = root / "20261030T110000Z"
    legacy.mkdir(parents=True)
    document = provider["document"]
    identity = pd.read_parquet(capture.HISTORY / "identity.parquet")
    frame = capture.parse(document, identity, "2026-10-30T11:00:00+00:00", 9)
    (legacy / "response.json").write_bytes(json.dumps(document).encode())
    identity.to_parquet(legacy / "identity.parquet", index=False)
    frame.to_parquet(legacy / "observations.parquet", index=False)
    (legacy / "summary.json").write_text(json.dumps(dict(observed_at="2026-10-30T11:00Z")))
    seal(legacy, dict(status="captured_research_observation"))
    assert capture.read_response(legacy) == document
    marker = capture.capture(9)
    assert json.loads((marker / "summary.json").read_text())["observations_from"] == legacy.name
