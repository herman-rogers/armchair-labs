import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from engine.data import refresh
from engine.data.releases import atomic_json
from engine.data.weekly import capture
from engine.metrics.rookies import completed_cutoff


def observations(week):
    return dict(season=2026, through_week=week, saved_at="2026-10-02T12:00:00+00:00")


def test_weekly_product_stages_after_delivery_without_early_activation():
    plan = refresh.stages("run", "history", "college", Path("capture.json"), "evidence")
    assert [s[0] for s in plan][-2:] == ["delivery", "weekly_model"]
    command = plan[-1][1]
    assert command[command.index("--analysis") + 1] == "run_delivery"
    assert "--publish" not in command


def test_weekly_activation_checks_completed_manifest(tmp_path, monkeypatch):
    from engine.data.releases import digest, reference

    monkeypatch.setattr(refresh, "DATA", tmp_path)
    root = tmp_path / "research/run_weekly"
    atomic_json(root / "manifest.json", {"version": "run_weekly"})
    manifest = root / "manifest.json"
    plan = {"completed": {"weekly_model": {str(manifest): digest(manifest)}}}
    refresh.activate_weekly("run", plan)
    assert json.loads((tmp_path / "weekly/current.json").read_text()) == reference(root)
    atomic_json(manifest, {"version": "changed"})
    with pytest.raises(ValueError, match="weekly model changed"):
        refresh.activate_weekly("run", plan)


def test_full_season_cutoff_requires_consecutive_final_games_and_team_stats():
    schedule = pl.DataFrame(
        [
            dict(
                season=2026,
                game_type="REG",
                week=w,
                home_team="A",
                away_team="B",
                home_score=20,
                away_score=10,
            )
            for w in range(1, 19)
        ]
    )
    stats = pl.DataFrame(
        [dict(season=2026, week=w, team=t) for w in range(1, 19) for t in ("A", "B")]
    )
    assert completed_cutoff(schedule, stats, 2026) == 18
    missing = stats.filter(~((pl.col("week") == 16) & (pl.col("team") == "B")))
    assert completed_cutoff(schedule, missing, 2026) == 15
    unfinished = schedule.with_columns(
        pl.when(pl.col("week") == 17).then(None).otherwise(pl.col("home_score")).alias("home_score")
    )
    assert completed_cutoff(unfinished, stats, 2026) == 16


def test_refresh_noop_force_and_cutoff_regression():
    assert not refresh.needs_refresh(observations(3), observations(3))
    assert refresh.needs_refresh(observations(3), observations(3), force=True)
    assert refresh.needs_refresh(observations(3), observations(4))
    with pytest.raises(ValueError, match="backward"):
        refresh.needs_refresh(observations(3), observations(2), force=True)
    with pytest.raises(ValueError, match="new season"):
        refresh.needs_refresh(observations(3), dict(season=2027, through_week=1))


def test_capture_unchanged_week_skips_expensive_sources(monkeypatch):
    from engine.data import weekly

    monkeypatch.setattr(weekly.nflverse, "load_player_weeks", lambda _: pl.DataFrame())
    monkeypatch.setattr(weekly.nflverse, "load_schedules", lambda _: pl.DataFrame())
    monkeypatch.setattr(weekly, "completed_cutoff", lambda *_: 3)
    monkeypatch.setattr(weekly.nflverse, "load_players", lambda: pytest.fail("Unneeded fetch"))
    assert capture(2026, after_week=3) is None
    with pytest.raises(ValueError, match="regressed"):
        capture(2026, after_week=4)


def setup_run(tmp_path, monkeypatch):
    monkeypatch.setattr(refresh, "DATA", tmp_path)
    monkeypatch.setattr(refresh, "refresh_named_tables", lambda *_, **__: {})
    monkeypatch.setattr(
        refresh, "read_registry", lambda: {"team_player_games": SimpleNamespace(refresh_hours=24)}
    )
    monkeypatch.setenv("NFLVERSE_CACHE_DURATION", "86400")
    atomic_json(tmp_path / "current.json", {"products": {"analysis": {"version": "old"}}})
    monkeypatch.setattr(
        refresh,
        "load_tables",
        lambda *_: SimpleNamespace(
            manifest={
                "current_observations": observations(2),
                "input": {},
                "history": {"version": "history"},
            }
        ),
    )
    monkeypatch.setattr(refresh, "load_analysis", lambda *_: (None, {"evidence": {"version": "e"}}))
    monkeypatch.setattr(
        refresh, "load_manifest", lambda *_: (None, {"input": {}, "college_source": "c"})
    )
    report = tmp_path / "capture.json"
    atomic_json(report, observations(3))
    args = SimpleNamespace(version="test_weekly", current_report=report, force=False)
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    monkeypatch.setattr(
        refresh,
        "stages",
        lambda *_: [("first", ["first"], [first]), ("second", ["second"], [second])],
    )
    return args, first, second


def test_failed_rebuild_preserves_catalog_and_resume_skips_completed_stage(tmp_path, monkeypatch):
    args, first, second = setup_run(tmp_path, monkeypatch)
    before = (tmp_path / "current.json").read_bytes()
    calls = []

    def fail_second(command, **_):
        calls.append(command[0])
        if command == ["second"]:
            raise subprocess.CalledProcessError(1, command)
        atomic_json(first, {"complete": True})

    monkeypatch.setattr(refresh.subprocess, "run", fail_second)
    monkeypatch.setattr(
        refresh, "publish_catalog", lambda *_, **__: pytest.fail("Premature publish")
    )
    with pytest.raises(subprocess.CalledProcessError):
        refresh.run(args)
    assert (tmp_path / "current.json").read_bytes() == before
    state = json.loads((tmp_path / ".runtime/nextgen_refresh.json").read_text())
    assert state["status"] == "failed" and state["stage"] == "second"

    def finish(command, **_):
        calls.append(command[0])
        atomic_json(second, {"complete": True})

    published = []
    table_refreshes = []
    monkeypatch.setattr(refresh.subprocess, "run", finish)
    monkeypatch.setattr(refresh, "publish_catalog", lambda *args, **_: published.append(args))

    def refresh_tables(data, *, names, **kwargs):
        assert published, "Named tables must use the newly published source catalog"
        table_refreshes.append((data, names))

    monkeypatch.setattr(refresh, "refresh_named_tables", refresh_tables)
    assert refresh.run(args)["status"] == "published"
    assert calls == ["first", "second", "second"]
    assert published[0][2]["analysis"] == "test_weekly_delivery"
    assert table_refreshes == [(tmp_path, ["team_player_games"])]


def test_table_refresh_failure_after_publication_can_resume(tmp_path, monkeypatch):
    args, first, second = setup_run(tmp_path, monkeypatch)
    stages_run = []

    def stage(command, **_):
        stages_run.append(command)
        atomic_json(first if command == ["first"] else second, {"complete": True})

    def publish(*_, **__):
        atomic_json(
            tmp_path / "current.json",
            {
                "products": {"analysis": {"version": "test_weekly_delivery"}},
            },
        )

    attempts = []

    def tables(data, *, names, **kwargs):
        attempts.append(names)
        if len(attempts) == 1:
            raise ValueError("Table refresh interrupted")

    monkeypatch.setattr(refresh.subprocess, "run", stage)
    monkeypatch.setattr(refresh, "publish_catalog", publish)
    monkeypatch.setattr(refresh, "refresh_named_tables", tables)
    with pytest.raises(ValueError, match="Table refresh interrupted"):
        refresh.run(args)
    assert (
        json.loads((tmp_path / ".runtime/nextgen_refresh.json").read_text())["status"] == "failed"
    )
    assert refresh.run(args)["status"] == "current"
    assert attempts == [["team_player_games"], ["team_player_games"]]
    assert stages_run == [["first"], ["second"]]


def test_unchanged_week_repairs_missing_or_outdated_tables(tmp_path, monkeypatch):
    args, _, _ = setup_run(tmp_path, monkeypatch)
    atomic_json(args.current_report, observations(2))
    calls = []
    monkeypatch.setattr(
        refresh, "refresh_named_tables", lambda data, **kw: calls.append((data, kw))
    )
    assert refresh.run(args)["status"] == "current"
    assert calls == [
        (
            tmp_path,
            {"names": ["team_player_games"], "upload": False, "store": refresh.DEFAULT_STORE},
        )
    ]


def test_unchanged_week_retries_upload_and_selects_all_refreshable_tables(tmp_path, monkeypatch):
    args, _, _ = setup_run(tmp_path, monkeypatch)
    args.current_report = None
    args.upload = True
    args.store = "file:///fixture-store"
    monkeypatch.setattr(refresh, "capture", lambda *_, **__: None)
    monkeypatch.setattr(
        refresh,
        "read_registry",
        lambda: {
            "new_weekly_summary": SimpleNamespace(refresh_hours=168),
            "frozen_study": SimpleNamespace(refresh_hours=None),
        },
    )
    calls = []

    def finish(data, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise OSError("network unavailable")
        return {"upload": {"status": "published"}}

    monkeypatch.setattr(refresh, "refresh_named_tables", finish)
    before = (tmp_path / "current.json").read_bytes()
    with pytest.raises(OSError, match="network unavailable"):
        refresh.run(args)
    result = refresh.run(args)
    assert result["tables"]["upload"]["status"] == "published"
    assert calls == [{"names": ["new_weekly_summary"], "upload": True, "store": args.store}] * 2
    assert (tmp_path / "current.json").read_bytes() == before


def test_concurrent_publication_is_not_overwritten(tmp_path, monkeypatch):
    args, first, second = setup_run(tmp_path, monkeypatch)

    def changed(command, **_):
        atomic_json(first if command == ["first"] else second, {"complete": True})
        atomic_json(tmp_path / "current.json", {"newer": True})

    monkeypatch.setattr(refresh.subprocess, "run", changed)
    monkeypatch.setattr(
        refresh, "publish_catalog", lambda *_, **__: pytest.fail("Overwrote new catalog")
    )
    with pytest.raises(ValueError, match="Catalog changed"):
        refresh.run(args)
    assert json.loads((tmp_path / "current.json").read_text()) == {"newer": True}


def test_catalog_compare_and_swap_is_checked_inside_publication_lock(tmp_path, monkeypatch):
    from engine.data import catalog
    from engine.data.releases import digest

    atomic_json(tmp_path / "current.json", {"version": "old"})
    expected = digest(tmp_path / "current.json")
    atomic_json(tmp_path / "current.json", {"version": "new"})
    monkeypatch.setattr(
        catalog, "_publish_catalog", lambda *_: pytest.fail("Replaced newer release")
    )
    with pytest.raises(ValueError, match="Catalog changed"):
        catalog.publish_catalog(tmp_path, "staged", {}, expected_catalog_sha256=expected)


def test_weekly_refresh_carries_only_unchanged_retirements():
    from engine.data.retirements import carry_retirements

    row = dict(
        id="stat:a",
        kind="research_stat",
        validity="revalidation_required",
        serving="archive",
        allowed_uses=[],
        reason="pending",
    )
    prior = dict(row, validity="archived", reason="retired after review")
    kwargs = dict(evidence_unchanged=True, legacy_unchanged=False, source={"version": "old"})
    assert carry_retirements([row], [prior], **kwargs)[0]["validity"] == "archived"
    assert row["validity"] == "revalidation_required"
    kwargs["evidence_unchanged"] = False
    assert carry_retirements([row], [prior], **kwargs) == [row]
    kwargs["evidence_unchanged"] = True
    assert carry_retirements([row], [dict(prior, validity="verified")], **kwargs) == [row]


def test_final_week_is_captured_without_publishing_nonexistent_future_games(tmp_path, monkeypatch):
    args, _, _ = setup_run(tmp_path, monkeypatch)
    atomic_json(args.current_report, observations(18))
    monkeypatch.setattr(refresh, "stages", lambda *_: pytest.fail("Season has ended"))
    before = (tmp_path / "current.json").read_bytes()
    result = refresh.run(args)
    assert result["status"] == "season_complete"
    assert Path(result["capture"]).exists()
    assert (tmp_path / "current.json").read_bytes() == before
    args.current_report = None
    monkeypatch.setattr(refresh, "capture", lambda *_, **__: pytest.fail("Repeated final capture"))
    assert refresh.run(args) == result


def test_late_season_qb_ranking_retains_zero_horizon_without_passing_exposure():
    from research.qb_variations import ranking_only_rows

    template = dict(
        player_id="a",
        season=2020,
        through_week=16,
        horizon="rest_of_season",
        scheduled_games=1,
        end_week=17,
    )
    rank = dict(player_id="a", season=2020, scheduled_games=0, end_week=17)
    other = dict(template, player_id="b", through_week=17)
    result = ranking_only_rows([template, other], rank, 17)
    assert result[0]["through_week"] == 17
    assert result[0]["scheduled_games"] == 0
    assert result[0]["_ranking_only"]
    assert template["scheduled_games"] == 1
    assert result[0]["player_id"] == "a"
    assert ranking_only_rows([template, other], dict(rank, scheduled_games=1), 17) == []
