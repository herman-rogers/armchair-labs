"""External evaluation must not reward future information or incomplete joins."""

import copy
import json

import polars as pl
import pytest
from typer.testing import CliRunner

from engine.chairstats import publish_campaign
from engine.cli_campaigns import app
from engine.evaluation.benchmarks import (
    SKILL_SCORING,
    compatibility,
    digest,
    evaluate,
    evaluate_all,
    load_source,
    publish_source,
    validate_source,
)
from engine.evaluation.sources import ecr_source, espn_rows, id_map


def external_row(pid="p1", value=10, **changes):
    return {
        "season": 2025,
        "origin_week": 0,
        "end_week": 18,
        "horizon": "season",
        "position": "RB",
        "player_id": pid,
        "target": "points",
        "prediction": value,
        "unit": "fantasy_points",
        "scoring": SKILL_SCORING,
        "vintage": "archived",
        "available_at": "2025-09-01T23:59:59+00:00",
        "issued_at": None,
        "source_url": "https://example.org/archived-projection",
        **changes,
    }


def source(rows=None, **changes):
    return {
        "schema_version": 1,
        "source_id": "external",
        "name": "Test comparator",
        "kind": "projection",
        "limitations": ["Fixture; no real forecast claim"],
        "rows": rows if rows is not None else [external_row()],
        **changes,
    }


@pytest.fixture
def campaign(tmp_path):
    rows = []
    for pid, pred, actual in [("p1", 9.0, 10.0), ("p2", 22.0, 20.0), ("p3", 0.0, 0.0)]:
        rows.append(
            {
                "player_id": pid,
                "year": 2025,
                "origin": 0,
                "target": "points",
                "prediction": pred,
                "actual": actual,
                "recipe": "frozen",
            }
        )
        rows.append({**rows[-1], "prediction": actual, "recipe": "tempting_perfect_challenger"})
    predictions = tmp_path / "predictions.parquet"
    pl.DataFrame(rows).write_parquet(predictions)
    metrics = tmp_path / "metrics.json"
    metrics.write_text("{}")
    root = tmp_path / "campaigns"
    publish_campaign(
        root,
        {"campaign_id": "test_campaign", "name": "Test"},
        [
            {
                "model_id": "rb_season",
                "position": "RB",
                "horizon": "season",
                "origin_week": 0,
                "targets": ["points"],
                "training_through_season": 2025,
                "metrics": metrics,
                "forecasts": [],
                "artifacts": {"evaluation_predictions": predictions},
                "benchmark_evaluation": {
                    "artifact": "evaluation_predictions",
                    "recipe": "frozen",
                    "scoring": SKILL_SCORING,
                    "cutoffs": [
                        {
                            "season": 2025,
                            "origin_week": 0,
                            "end_week": 18,
                            "as_of": "2025-09-04T12:00:00+00:00",
                        }
                    ],
                },
            }
        ],
    )
    return root


def run(tmp_path, campaign, data):
    path = publish_source(tmp_path / "external", data)
    return evaluate(campaign, "test_campaign", [path], tmp_path / "report")


def test_matched_errors_frozen_policy_coverage_and_read_only(campaign, tmp_path):
    before = {p: digest(p.read_bytes()) for p in campaign.rglob("*") if p.is_file()}
    result = run(
        tmp_path,
        campaign,
        source([external_row("p1", 14), external_row("p2", 25), external_row("outside", 50)]),
    )
    pooled = next(
        c for c in result["comparisons"] if c["season"] is None and c["cohort"] == "matched"
    )
    assert pooled["model"]["mae"] == 1.5
    assert pooled["external"]["mae"] == 4.5
    assert pooled["mae_gain"] == 3
    assert pooled["equal_season_mae_gain_ci95"] is None
    coverage = result["coverage"][0]
    assert coverage["model_rows"] == 3
    assert coverage["matched_rows"] == 2
    assert coverage["model_top_k_missing"] == ["p3"]
    assert coverage["external_only_ids"] == ["outside"]
    assert before == {p: digest(p.read_bytes()) for p in campaign.rglob("*") if p.is_file()}
    assert (tmp_path / "report/index.html").is_file()
    assert (tmp_path / "report/paired_rows.csv").is_file()
    with pytest.raises(FileExistsError):
        evaluate(
            campaign, "test_campaign", [tmp_path / "external/source.json"], tmp_path / "report"
        )


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"available_at": "2025-09-05T00:00:00+00:00"}, "external_after_cutoff"),
        (
            {
                "issued_at": "2025-09-05T00:00:00+00:00",
                "available_at": None,
                "vintage": "reconstructed",
            },
            "external_after_cutoff",
        ),
        ({"available_at": "2025-08-01T00:00:00+00:00"}, "external_too_old"),
        ({"origin_week": 2, "horizon": "remaining"}, "incompatible_origin_week"),
        ({"end_week": 17}, "incompatible_end_week"),
        ({"scoring": "half_ppr"}, "incompatible_scoring_or_unit"),
    ],
)
def test_mismatches_never_scored(campaign, tmp_path, changes, reason):
    result = run(tmp_path, campaign, source([external_row(**changes)]))
    assert result["status"] == "no_comparable_rows"
    assert result["comparisons"] == []
    assert result["coverage"][0]["excluded"][reason] == 1


def test_reconstructed_is_separate_not_certified(campaign, tmp_path):
    result = run(
        tmp_path, campaign, source([external_row(vintage="reconstructed", available_at=None)])
    )
    assert all(
        c["comparison_class"] == "reconstructed_external_vintage" for c in result["comparisons"]
    )


def test_rankings_have_no_fabricated_point_errors(campaign, tmp_path):
    result = run(
        tmp_path,
        campaign,
        source(
            [
                external_row("p1", 1, unit="rank", scoring="standard_ppr"),
                external_row("p2", 2, unit="rank", scoring="standard_ppr"),
            ],
            kind="ranking",
        ),
    )
    annual = next(c for c in result["comparisons"] if c["season"] == 2025)
    assert "mae_gain" not in annual and "model" not in annual
    assert annual["external_spearman"] == pytest.approx(-1)
    assert annual["model_spearman"] == pytest.approx(1)
    assert annual["comparison_class"] == "ranking_scoring_difference"


@pytest.mark.parametrize("change", ["duplicate", "nan", "naive_time", "missing_scoring", "unit"])
def test_invalid_external_contracts_fail(change):
    s = source()
    if change == "duplicate":
        s["rows"].append(copy.deepcopy(s["rows"][0]))
    elif change == "nan":
        s["rows"][0]["prediction"] = float("nan")
    elif change == "naive_time":
        s["rows"][0]["available_at"] = "2025-09-01"
    elif change == "missing_scoring":
        del s["rows"][0]["scoring"]
    else:
        s["rows"][0]["unit"] = "yards"
    with pytest.raises(ValueError):
        validate_source(s)


def test_source_and_raw_checksums(tmp_path):
    path = publish_source(tmp_path / "s", source(), {"raw/data.json": b"{}"})
    assert load_source(path)["source_id"] == "external"
    (path.parent / "raw/data.json").write_text("changed")
    with pytest.raises(ValueError, match="checksum"):
        load_source(path)
    path.write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        load_source(path)


def test_campaign_artifact_corruption_fails(campaign, tmp_path):
    p = next(campaign.rglob("evaluation_predictions.parquet"))
    p.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        run(tmp_path, campaign, source())


def test_unknown_campaign_does_not_infer_recipe(campaign, tmp_path):
    manifest = campaign / "test_campaign/manifest.json"
    d = json.loads(manifest.read_text())
    del d["models"][0]["benchmark_evaluation"]
    manifest.write_text(json.dumps(d))
    manifest.with_suffix(".sha256").write_text(digest(manifest.read_bytes()))
    result = run(tmp_path, campaign, source())
    assert result["status"] == "no_comparable_rows"
    assert result["models"][0]["reason"] == "No benchmark_evaluation adapter declared"


def test_espn_projection_only_exact_ids_complete_scoring():
    # ESPN's projected catches use stat 53 (REC), not legacy stat 41 (RECS).
    stats = {"23": 10, "24": 50, "25": 1, "53": 4, "42": 30, "43": 0, "58": 5}
    entry = {
        "seasonId": 2025,
        "statSourceId": 1,
        "statSplitTypeId": 0,
        "scoringPeriodId": 0,
        "stats": stats,
    }
    players = [
        {
            "id": 42,
            "defaultPositionId": 2,
            "stats": [
                entry,
                {**entry, "stats": {}},
                {**entry, "statSourceId": 0, "stats": {"24": 9999}},
            ],
        }
    ]
    rows, audit = espn_rows(players, 2025, {"42": "p1"})
    assert next(r for r in rows if r["target"] == "points")["prediction"] == 18
    assert all(r["vintage"] == "reconstructed" and r["issued_at"] is None for r in rows)
    del stats["25"]
    rows, audit = espn_rows(players, 2025, {"42": "p1"})
    assert "points" not in {r["target"] for r in rows}
    assert audit["incomplete_points_components"] == 1
    assert espn_rows(players, 2025, {})[0] == []


def test_ambiguous_provider_ids_are_withheld():
    frame = pl.DataFrame({"espn_id": ["1", "1", "2"], "gsis_id": ["a", "b", "c"]})
    assert id_map(frame, "espn_id") == ({"2": "c"}, ["1"])


def test_ecr_chooses_snapshot_not_latest_per_player(tmp_path):
    rows = [
        {
            "fp_page": "/nfl/rankings/ppr-rb-cheatsheets.php",
            "ecr_type": "rp",
            "pos": "RB",
            "scrape_date": date,
            "id": pid,
            "ecr": rank,
        }
        for date, pid, rank in [
            ("2025-08-29", "1", 1),
            ("2025-08-29", "2", 2),
            ("2025-09-03", "1", 2),
            ("2025-09-04", "1", 99),
        ]
    ]
    archive, crosswalk, calendar = [
        tmp_path / n for n in ("ecr.parquet", "ids.parquet", "calendar.json")
    ]
    pl.DataFrame(rows).write_parquet(archive)
    pl.DataFrame({"fantasypros_id": ["1", "2"], "gsis_id": ["a", "b"]}).write_parquet(crosswalk)
    calendar.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "season": 2025,
                        "origin_week": 0,
                        "end_week": 18,
                        "as_of": "2025-09-04T12:00:00+00:00",
                    }
                ]
            }
        )
    )
    p = ecr_source(archive, crosswalk, calendar, tmp_path / "source", [2025])
    rows = load_source(p)["rows"]
    assert len(rows) == 1 and rows[0]["prediction"] == 2
    assert rows[0]["player_id"] == "a"


def test_cli_runs_evaluation(campaign, tmp_path):
    path = publish_source(tmp_path / "external", source())
    result = CliRunner().invoke(
        app,
        [
            "benchmark",
            "test_campaign",
            "--source",
            str(path),
            "--output",
            str(tmp_path / "result"),
            "--data-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "result/index.html").exists()


def test_postcutoff_rank_does_not_gain_access_by_scoring_difference():
    row = external_row(
        unit="rank", scoring="standard_ppr", available_at="2025-09-06T00:00:00+00:00"
    )
    model = {**external_row(), "as_of": "2025-09-04T12:00:00+00:00"}
    assert compatibility(model, row, "ranking", 14) == (None, "external_after_cutoff")


def test_all_campaigns_links_reports_and_refuses_registry_output(campaign, tmp_path):
    path = publish_source(tmp_path / "external", source())
    result = evaluate_all(campaign, [path], tmp_path / "review")
    assert result["campaigns"][0]["status"] == "evaluated"
    assert result["campaigns"][0]["report"] == "test_campaign/index.html"
    assert (tmp_path / "review/test_campaign/report.json").exists()
    with pytest.raises(ValueError, match="outside"):
        evaluate_all(campaign, [path], campaign / "review")


def test_csv_handles_late_nonnull_vintage(campaign, tmp_path):
    # Real imports put hundreds of reconstructed rows before archived timestamps.
    paths = [
        publish_source(
            tmp_path / f"source{i}",
            source(
                [external_row(vintage="reconstructed", available_at=None)],
                source_id=f"source{i}",
            ),
        )
        for i in range(101)
    ]
    paths.append(publish_source(tmp_path / "archived", source()))
    evaluate(campaign, "test_campaign", paths, tmp_path / "report")
    rows = pl.read_csv(tmp_path / "report/paired_rows.csv", infer_schema_length=None)
    assert rows.height == 102
    assert rows["external_available_at"].drop_nulls().to_list() == ["2025-09-01T23:59:59+00:00"]
