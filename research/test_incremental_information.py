"""Guard the estimands, common samples, chronology and provenance of the audit."""

import json

import polars as pl
import pytest
from data_integrity_audit import digest
from incremental_information import (
    load_release,
    point_comparisons,
    require_unique,
    role_comparisons,
    season_summary,
)


def test_equal_season_weighting_and_residual_decomposition():
    # The large season worsens by one; the small season improves by three.
    frame = pl.DataFrame(
        {
            "season": [2020] * 9 + [2021],
            "actual": [0.0] * 9 + [3.0],
            "baseline": [0.0] * 10,
            "candidate": [1.0] * 9 + [3.0],
        }
    )
    result = point_comparisons(frame, "actual", "candidate", ["baseline"])[0]
    assert result["mae_improvement"]["mean"] == pytest.approx(1)
    assert result["pooled_mae"]["baseline"] == pytest.approx(0.3)
    assert result["pooled_mae"]["candidate"] == pytest.approx(0.9)
    assert result["mse_improvement"]["mean"] == pytest.approx(4)
    assert result["residual_alignment"] - result["adjustment_cost"] == pytest.approx(4)
    assert result["mae_improvement"]["leave_one_season_out_mean_range"] == [-1, 3]


def test_all_baselines_use_identical_finite_rows():
    frame = pl.DataFrame(
        {
            "season": [2020] * 4,
            "actual": [1.0] * 4,
            "candidate": [1.0, 1.0, float("inf"), 1.0],
            "weak": [0.0, 0.0, 0.0, 0.0],
            "strong": [0.5, None, 0.5, float("nan")],
        }
    )
    results = point_comparisons(frame, "actual", "candidate", ["weak", "strong"])
    assert [r["common_rows"] for r in results] == [1, 1]
    assert [r["excluded_nonfinite_rows"] for r in results] == [3, 3]
    assert results[0]["mae_improvement"]["season_bootstrap_95"] is None


def test_future_outcomes_refused():
    frame = pl.DataFrame({"season": [2026], "actual": [1.0], "c": [0.0], "b": [0.0]})
    with pytest.raises(ValueError, match="2026"):
        point_comparisons(frame, "actual", "c", ["b"])


@pytest.mark.parametrize("ids", [["a", "a"], ["a", None]])
def test_invalid_observation_keys_refused(ids):
    with pytest.raises(ValueError, match="observation key"):
        require_unique(pl.DataFrame({"player_id": ids}), ["player_id"])


def selections():
    return pl.DataFrame(
        [
            {
                "cohort": "all",
                "policy": policy,
                "season": season,
                "week": 3,
                "player_id": p,
                "next4_points": points,
            }
            for season in (2020, 2021)
            for policy in ("trailing_points", "recent_snaps", "usage_ridge", "role_ridge")
            for p, points in (
                [("a", 10.0), ("b", 20.0)]
                + ([("d", 60.0)] if policy == "role_ridge" else [("c", 30.0)])
            )
        ]
    )


def test_selection_gain_is_per_pick_and_overlap_uses_player_ids():
    for row in role_comparisons(selections()):
        assert row["picks"] == 6
        assert row["changed_picks"] == 2
        assert row["unchanged_selection_fraction"] == pytest.approx(2 / 3)
        assert row["gain_per_selection"]["mean"] == pytest.approx(10)


def test_missing_policy_window_fails_instead_of_silently_changing_sample():
    frame = selections().filter(~((pl.col("policy") == "usage_ridge") & (pl.col("season") == 2021)))
    with pytest.raises(ValueError, match="identical decision windows"):
        role_comparisons(frame)


def test_conflicting_selection_outcomes_refused():
    frame = selections().with_columns(
        pl.when((pl.col("policy") == "role_ridge") & (pl.col("player_id") == "a"))
        .then(100.0)
        .otherwise(pl.col("next4_points"))
        .alias("next4_points")
    )
    with pytest.raises(ValueError, match="disagree"):
        role_comparisons(frame)


@pytest.mark.parametrize("change", ["artifact", "history", "incomplete"])
def test_release_provenance_fails_closed(tmp_path, change):
    source = tmp_path / "source.py"
    source.write_text("source")
    output = tmp_path / "predictions.parquet"
    output.write_bytes(b"frozen")
    manifest = {
        "status": "complete",
        "protected_artifacts_unchanged": True,
        "history": {"version": "accepted"},
        "source_sha256": {source.name: digest(source)},
        "output_sha256": {output.name: digest(output)},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    load_release(tmp_path, manifest["history"], {output.name}, {})
    if change == "artifact":
        output.write_bytes(b"changed")
    elif change == "history":
        manifest["history"] = {"version": "different"}
    else:
        manifest["status"] = "building"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load_release(tmp_path, {"version": "accepted"}, {output.name}, {})


def test_uncertainty_is_reproducible_and_not_reported_for_one_season():
    assert season_summary([-1, 0, 3]) == season_summary([-1, 0, 3])
    assert season_summary([5])["season_bootstrap_95"] is None
