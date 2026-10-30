"""Monitor passes with unchanged official reports must not re-capture or republish."""

import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pandas as pd
import pytest
from experiments.future_player_lab.campaigns.weekly_v32 import live as v32_live
from experiments.future_player_lab.campaigns.weekly_v32 import official
from experiments.future_player_lab.campaigns.weekly_v41 import track

SEASON, WEEK = 2026, 9
START = datetime(2026, 10, 30, 12, tzinfo=UTC)


def page(status="Questionable"):
    row = f"<tr><td>Joe Back</td><td>RB</td><td>Knee</td><td>Limited</td><td>{status}</td></tr>"
    return (
        f"<title>Week {WEEK} of the {SEASON} Season</title>"
        '<section class="nfl-o-injury-report__unit">'
        '<span class="nfl-c-matchup-strip__team-abbreviation">BUF</span>'
        '<span class="nfl-c-matchup-strip__team-abbreviation">MIA</span>'
        f"<tbody>{row}</tbody><tbody></tbody></section>"
    )


IDENTITY = pd.DataFrame(
    dict(player_id=["p1"], player_name=["Joe Back"], team=["BUF"], position=["RB"])
)


@pytest.fixture
def capture(tmp_path, monkeypatch):
    clock = SimpleNamespace(now=START, page=page())
    monkeypatch.setattr(official, "CAPTURES", tmp_path / "captures")
    monkeypatch.setattr(official, "now", lambda: clock.now)
    monkeypatch.setattr(
        official.requests,
        "get",
        lambda *a, **k: SimpleNamespace(
            text=clock.page, content=clock.page.encode(), raise_for_status=lambda: None
        ),
    )

    def run(minutes=15, status="Questionable"):
        clock.now += timedelta(minutes=minutes)
        clock.page = page(status)
        return official.capture(SEASON, WEEK, IDENTITY)

    return run


def test_unchanged_official_page_reuses_recent_capture(capture):
    first = capture()
    assert capture() == first
    assert capture() == first
    changed = capture(status="Out")
    assert changed != first
    assert capture(status="Out") == changed
    # Re-sealed periodically so admitted reports stay inside the freshness window.
    refreshed = capture(minutes=official.UNCHANGED_REFRESH.total_seconds() / 60 + 1, status="Out")
    assert refreshed != changed
    assert len(list(official.CAPTURES.glob("[!.]*/summary.json"))) == 3


def test_report_content_ignores_observation_times():
    def reports(at, status="Out"):
        return pd.DataFrame(
            dict(
                player_id=["p1", "p2"],
                season=SEASON,
                week=WEEK,
                team="BUF",
                usable_at=pd.Timestamp(at),
                report_status=[status, None],
                practice_status="Limited",
                source="official_nfl_weekly_page",
            )
        )

    early, later = reports(START), reports(START + timedelta(hours=3))
    assert track.report_content(early) == track.report_content(later)
    # A repeated older capture alongside the latest one is still the same content.
    assert track.report_content(pd.concat([early, later])) == track.report_content(later)
    newer = reports(START + timedelta(hours=4), status="Questionable")
    assert track.report_content(pd.concat([early, newer])) != track.report_content(early)


@pytest.fixture
def pregame(tmp_path, monkeypatch):
    """Drive ``track.pregame`` with stubbed captures, models and publication."""
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text(json.dumps(dict(current_observations=dict(season=SEASON, through_week=8))))
    research = tmp_path / "run"
    research.mkdir()
    (research / "manifest.json").write_text("{}")
    outputs = tmp_path / "outputs"
    early = outputs / "forecasts" / f"{SEASON}_{WEEK:02d}" / "early_week"
    early.mkdir(parents=True)
    players = pd.DataFrame(dict(player_id=["p1"], team=["BUF"], is_bye=[0], prediction=[1.0]))
    players.to_parquet(early / "players.parquet")
    games = pd.DataFrame(
        dict(
            season=SEASON,
            week=WEEK,
            game_type="REG",
            game_id=["g1", "g2"],
            home_team=["BUF", "NYJ"],
            away_team=["MIA", "NE"],
            kickoff=[pd.Timestamp(START + timedelta(days=2))] * 2,
        )
    )
    state = SimpleNamespace(status="Out", manifest=0, published=0, games=games)
    monkeypatch.setattr(v32_live, "asset", lambda *a: None)
    monkeypatch.setattr(v32_live, "schedule_times", lambda _: state.games)
    monkeypatch.setattr(track, "official_directories", lambda: [])

    def official_reports(directories, season, week, captured_at):
        # Each pass seals a new capture: new path, manifest hash and observation time.
        state.manifest += 1
        capture = tmp_path / "captures" / str(state.manifest)
        capture.mkdir(parents=True)
        (capture / "summary.json").write_text(json.dumps(dict(teams=["BUF", "MIA", "NYJ", "NE"])))
        reports = pd.DataFrame(
            dict(
                player_id=["p1"],
                season=SEASON,
                week=WEEK,
                team="BUF",
                usable_at=pd.Timestamp(captured_at),
                report_status=state.status,
                practice_status="Did Not Participate",
                source="official_nfl_weekly_page",
            )
        )
        audit = [dict(path=str(capture), manifest_sha256=str(state.manifest), admitted=True)]
        return reports, audit

    def publish(*args, **kwargs):
        state.published += 1
        return dict(status="published")

    monkeypatch.setattr(track.live, "official_reports", official_reports)
    monkeypatch.setattr(track.live, "forecast", lambda *a: (players, None, None))
    monkeypatch.setattr(track.live, "projection", lambda *a: None)
    monkeypatch.setattr(track, "save_outputs", lambda *a: None)
    monkeypatch.setattr(track, "publish", publish)

    def run(minutes=15, status="Out"):
        state.status = status
        state.now = getattr(state, "now", START) + timedelta(minutes=minutes)
        return track.pregame(snapshot, research, None, outputs, captured_at=state.now)["status"]

    return SimpleNamespace(run=run, state=state)


def test_pregame_republishes_only_when_reports_change(pregame):
    assert pregame.run() == "published"
    assert pregame.run() == "current"
    assert pregame.run(minutes=120) == "current"
    assert pregame.state.published == 1
    assert pregame.run(status="Questionable") == "published"
    assert pregame.run(status="Questionable") == "current"
    assert pregame.state.published == 2


def test_pregame_kickoff_alone_does_not_republish(pregame):
    assert pregame.run() == "published"
    # g2 kicks off: fewer eligible games, nothing new to forecast.
    pregame.state.games = pregame.state.games.assign(
        kickoff=[pd.Timestamp(START + timedelta(days=2)), pd.Timestamp(START)]
    )
    assert pregame.run() == "current"
    assert pregame.state.published == 1
