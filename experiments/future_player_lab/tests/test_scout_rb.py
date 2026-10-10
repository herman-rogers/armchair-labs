"""Verify the legacy adapter preserves saved values and rejects incompatible inputs."""

import hashlib
import json
import zipfile

import numpy as np
import polars as pl
import pytest
from experiments.future_player_lab.campaigns.scout_rb import (
    TARGETS,
    extract_archive,
    make_archive,
    normalized_rows,
)
from experiments.future_player_lab.gpu_lab.run import digest


def forecast_input():
    rows = []
    for i in (1, 2):
        rows.append(
            {
                "player_id": str(i),
                "player_display_name": f"Player {i}",
                "position": "RB",
                "team": "SEA",
                "year": 2026,
                "origin": 2,
                "rank": i,
                "ppr_ensemble_top3": 120.0 / i,
                "ppr_per_scheduled_game": 8.0 / i,
                "scheduled_games": 15,
                "weeks": 16,
                "candidate_basis": "prior_history",
            }
        )
    predictions = {target: [2.0, 1.0] for target in TARGETS}
    predictions["points"] = [60.0, 120.0]
    return pl.DataFrame(rows), {"player_ids": ["2", "1"], "predictions": predictions}


def test_exact_player_alignment_preserves_saved_points_and_rank():
    frame, predicted = forecast_input()
    predicted["predictions"]["points"][0] += 1e-9
    rows, difference = normalized_rows(frame, predicted)
    assert [r["player_id"] for r in rows] == ["1", "2"]
    assert rows[0]["predictions"]["carries"] == 1
    assert rows[1]["predictions"]["carries"] == 2
    assert rows[1]["predictions"]["points"] == 60.0
    assert rows[1]["points_per_scheduled_game"] == 4.0
    assert 0 < difference < 2e-9
    assert rows[1]["component_points"] != rows[1]["predictions"]["points"]


@pytest.mark.parametrize("change", ["duplicate", "unknown", "missing", "nonfinite", "points"])
def test_invalid_inference_rejected(change):
    frame, predicted = forecast_input()
    if change == "duplicate":
        predicted["player_ids"] = ["1", "1"]
    elif change == "unknown":
        predicted["player_ids"] = ["3", "1"]
    elif change == "missing":
        del predicted["predictions"]["carries"]
    elif change == "nonfinite":
        predicted["predictions"]["carries"][0] = np.nan
    else:
        predicted["predictions"]["points"][0] += 1
    with pytest.raises((ValueError, AssertionError)):
        normalized_rows(frame, predicted)


def test_original_cutoff_must_match():
    frame, predicted = forecast_input()
    frame = frame.with_columns(pl.lit(3).alias("origin"))
    with pytest.raises(ValueError, match="cutoff"):
        normalized_rows(frame, predicted)


def test_archive_exact_bytes_and_external_checksum(tmp_path):
    source = tmp_path / "weights.joblib"
    source.write_bytes(b"exact original bytes, not deserialized")
    archive = tmp_path / "library.zip"
    make_archive(archive, {"final_models/weights.joblib": source})
    destination = tmp_path / "extracted"
    extract_archive(archive, destination, digest(archive))
    assert (destination / "final_models/weights.joblib").read_bytes() == source.read_bytes()
    with pytest.raises(ValueError, match="archive checksum"):
        extract_archive(archive, tmp_path / "rejected", "0" * 64)


@pytest.mark.parametrize("name,tamper", [("../outside", False), ("model", True)])
def test_archive_rejects_unsafe_path_and_internal_tamper(tmp_path, name, tamper):
    content = b"original"
    archive = tmp_path / "library.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(name, b"changed" if tamper else content)
        bundle.writestr(
            "archive_receipt.json",
            json.dumps({"files": {name: hashlib.sha256(content).hexdigest()}}),
        )
    with pytest.raises(ValueError, match="Unsafe|member checksum"):
        extract_archive(archive, tmp_path / "extracted", digest(archive))
    assert not (tmp_path / "outside").exists()
