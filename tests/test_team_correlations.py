"""Joint variance must stay coherent when pairwise samples differ."""

from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
import yaml
from fastapi import HTTPException

from engine.api import team_analysis_routes as routes
from engine.data.team_analysis import TABLES, _panel, normalize
from engine.metrics.team_correlations import analyze_team, covariance, lineup_risk, pair_summary
from engine.tables.build import refresh
from engine.tables.storage import current, load_table


def panel(series, years=None):
    rows = []
    for pid, values in series.items():
        for index, value in enumerate(values):
            if value is None:
                continue
            season = years[index] if years else 2025
            rows.append(
                dict(
                    game_id=f"game-{index}",
                    season=season,
                    week=index + 1,
                    gameday=f"{season}-10-{index + 1:02}",
                    team="LA",
                    player_id=pid,
                    name=pid,
                    position="QB" if pid == "qb" else "WR",
                    starting_qb_id="qb",
                    starter_game=True,
                    league_points=float(value),
                    offense_pct=0.9,
                )
            )
    return pl.DataFrame(rows)


def test_positive_covariance_adds_exact_variance_not_mean_points():
    data = panel({"qb": [1, 2, 3, 4], "wr": [2, 4, 6, 8]})
    result = lineup_risk(data, ["qb", "wr"], "raw")
    assert result["mean"] == pytest.approx(7.5)
    assert result["variance"] == pytest.approx(np.var([3, 6, 9, 12], ddof=1))
    assert result["independent_variance"] + result["covariance_effect"] == pytest.approx(
        result["variance"]
    )
    assert result["variance_change_pct"] == pytest.approx(80)
    assert sum(row["total"] for row in result["contributions"]) == pytest.approx(result["variance"])
    assert np.linalg.eigvalsh(result["covariance"]).min() >= -1e-10


def test_perfect_cancellation_has_zero_joint_risk():
    result = lineup_risk(panel({"a": [1, 2, 3, 4], "b": [4, 3, 2, 1]}), ["a", "b"], "raw")
    assert result["sd"] == pytest.approx(0)
    assert result["variance_change_pct"] == pytest.approx(-100)
    assert result["quantiles"]["p10"] == result["quantiles"]["p90"] == 5


def test_season_intercepts_removed_and_denominator_correct():
    values = np.array([[0, 100], [1, 101], [100, 0], [101, 1]])
    years = np.array([2024, 2024, 2025, 2025])
    raw, _ = covariance(values, years, "raw")
    adjusted, df = covariance(values, years, "season")
    assert df == 2 and raw[0, 1] < 0
    np.testing.assert_allclose(adjusted, [[0.5, 0.5], [0.5, 0.5]])


def test_pairwise_overlap_cannot_be_stitched_into_joint_risk():
    data = panel(
        {
            "a": [1, 2, 3, 4, 5, 6, None, None, None],
            "b": [1, 2, 3, None, None, None, 7, 8, 9],
            "c": [None, None, None, 4, 5, 6, 7, 8, 9],
        }
    )
    assert pair_summary(data, "a", "b", "raw")["r"] == pytest.approx(1)
    assert pair_summary(data, "a", "c", "raw")["r"] == pytest.approx(1)
    result = lineup_risk(data, ["a", "b", "c"], "raw")
    assert result["n"] == 0 and result["status"] == "insufficient_overlap"
    assert result["variance"] is None


def test_single_player_constant_points_and_insufficient_df():
    data = panel({"a": [7, 7, 7]})
    result = lineup_risk(data, ["a"], "raw")
    assert result["sd"] == 0
    assert result["variance_change_pct"] is None
    assert result["covariance_effect"] == 0
    assert lineup_risk(data, [], "raw")["status"] == "empty_selection"
    sparse = panel({"a": [1, 2, 3]}, [2023, 2024, 2025])
    assert lineup_risk(sparse, ["a"], "season")["variance"] is None


def test_absences_are_missing_but_real_zero_games_are_included():
    data = panel({"qb": [1, 2, 3, 4], "wr": [0, None, 8, 12]})
    result = lineup_risk(data, ["qb", "wr"], "raw")
    assert result["n"] == 3
    assert result["mean"] == pytest.approx((1 + 11 + 16) / 3)


def test_participation_filter_applies_to_every_pair():
    data = panel({"qb": [1, 2, 3, 4, 5], "wr": [0, 2, 3, 4, 5]}).with_columns(
        pl.when((pl.col("player_id") == "wr") & (pl.col("week") == 1))
        .then(False)
        .otherwise(pl.col("starter_game"))
        .alias("starter_game")
    )
    args = dict(team="LA", start=2021, end=2025, qb="current", basis="raw", selected=["qb", "wr"])
    starter = analyze_team(data, sample="starter", **args)
    active = analyze_team(data, sample="active", **args)
    assert starter["risk"]["n"] == starter["pairs"][0]["n"] == 4
    assert active["risk"]["n"] == 5


def test_old_and_current_franchise_codes_match():
    result = normalize(pl.DataFrame({"team": ["LAR", "STL", "OAK", "SD", "BUF"]}))
    assert result["team"].to_list() == ["LA", "LA", "LV", "LAC", "BUF"]


def test_latest_spot_start_does_not_replace_primary_qb_default():
    data = panel(
        {"qb": [1, 2, 3, 4, None], "backup": [None, None, None, None, 5], "wr": [1, 2, 3, 4, 5]}
    ).with_columns(
        pl.when(pl.col("player_id") == "backup")
        .then(pl.lit("QB"))
        .otherwise(pl.col("position"))
        .alias("position"),
        pl.when(pl.col("week") == 5)
        .then(pl.lit("backup"))
        .otherwise(pl.col("starting_qb_id"))
        .alias("starting_qb_id"),
    )
    result = analyze_team(
        data,
        team="LA",
        start=2021,
        end=2025,
        qb="current",
        sample="starter",
        basis="raw",
        selected=None,
    )
    assert result["qb"] == "qb" and result["team_games"] == 4


def test_short_qb_history_keeps_established_receivers_with_missing_risk():
    data = panel(
        {"qb": [1, 2, 3, None, None], "backup": [None, None, None, 4, 5], "wr": [1, 2, 3, 4, 5]}
    ).with_columns(
        pl.when(pl.col("player_id") == "backup")
        .then(pl.lit("QB"))
        .otherwise(pl.col("position"))
        .alias("position"),
        pl.when(pl.col("week") >= 4)
        .then(pl.lit("backup"))
        .otherwise(pl.col("starting_qb_id"))
        .alias("starting_qb_id"),
    )
    result = analyze_team(
        data,
        team="LA",
        start=2021,
        end=2025,
        qb="backup",
        sample="starter",
        basis="raw",
        selected=["backup", "wr"],
    )
    assert {p["player_id"] for p in result["players"]} == {"backup", "wr"}
    assert result["risk"]["n"] == 2
    assert result["risk"]["status"] == "insufficient_overlap"


def test_source_panel_restores_real_zeros_and_honors_current_schedule_and_cutoff(tmp_path):
    def stat(pid, season, week):
        return dict(
            game_id=f"{season}-{week}",
            player_id=pid,
            player_display_name=pid,
            position="QB" if pid == "qb" else "WR",
            team="LA",
            season=season,
            week=week,
            league_points=10.0,
            fantasy_points_ppr=10.0,
            targets=0,
        )

    def game(season, week, score):
        return dict(
            game_id=f"{season}-{week}",
            season=season,
            week=week,
            game_type="REG",
            gameday=f"{season}-09-{week + 10:02}",
            home_team="LA",
            away_team="BUF",
            home_score=score,
            home_qb_id="qb",
            away_qb_id="other",
        )

    historical = [stat(p, 2025, w) for p in ["qb", "wr"] for w in [1, 2, 3] if (p, w) != ("wr", 2)]
    snaps = [
        dict(
            game_id=f"2025-{w}",
            season=2025,
            week=w,
            team="LA",
            player=p,
            pfr_player_id=p,
            position="QB" if p == "qb" else "WR",
            offense_snaps=55.0,
            offense_pct=0.9,
        )
        for p in ["qb", "wr"]
        for w in [1, 2, 3]
    ]
    sources = {
        "nfl_player_weeks": pl.DataFrame(historical),
        "current_weeks": pl.DataFrame([stat("qb", 2026, 1), stat("qb", 2026, 2)]),
        "nfl_snap_counts": pl.DataFrame(snaps),
        "current_snaps": pl.DataFrame(
            [
                dict(player_id=p, season=2026, week=w, offense_snaps=55.0, offense_pct=0.9)
                for p in ["qb", "wr"]
                for w in [1, 2]
            ]
        ),
        "players": pl.DataFrame({"gsis_id": ["qb", "wr"], "pfr_id": ["qb", "wr"]}),
        "current_players": pl.DataFrame(
            [
                dict(
                    player_id=p,
                    team="LA",
                    player_display_name=p,
                    position="QB" if p == "qb" else "WR",
                )
                for p in ["qb", "wr"]
            ]
        ),
        "nfl_schedule": pl.DataFrame(
            [game(2025, w, 17) for w in [1, 2, 3]] + [game(2026, 1, None), game(2026, 2, 21)]
        ),
        "current_schedule": pl.DataFrame([game(2026, 1, 17), game(2026, 2, 21)]),
    }
    paths = []
    for name in TABLES:
        path = tmp_path / f"{name}.parquet"
        sources[name].write_parquet(path)
        paths.append(str(path))
    result = _panel(tuple(paths), "fixture", 2026, 1)
    assert result.height == 8
    assert result["game_id"].n_unique() == 4
    assert (
        result.filter((pl.col("player_id") == "wr") & (pl.col("season") == 2026))[
            "league_points"
        ].item()
        == 0
    )
    assert (
        result.filter((pl.col("player_id") == "wr") & (pl.col("week") == 2))["league_points"].item()
        == 0
    )
    assert result.filter(pl.col("season") == 2026)["week"].max() == 1


@pytest.fixture
def table_source(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    observations = pl.concat(
        [
            panel({"qb": [1, 2, 3], "wr": [2, 4, 6]}, years=[2013] * 3).with_columns(
                (pl.lit("historical-") + pl.col("game_id")).alias("game_id")
            ),
            panel({"qb": [1, 2, 3, 4], "wr": [4, 5, 6, 7]}),
        ]
    )
    observations.write_parquet(data / "source.parquet")
    registry = tmp_path / "tables.yaml"
    registry.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "tables": {
                    "team_player_games": {
                        "kind": "parquet",
                        "source": "source.parquet",
                        "description": "Team observations",
                        "grain": "One player per team and game",
                        "primary_key": ["game_id", "team", "player_id"],
                        "observation_cutoff": {"season": 2025, "through_week": 4},
                    }
                },
            }
        )
    )
    refresh(data, registry_path=registry)
    monkeypatch.setattr(routes, "get_settings", lambda: SimpleNamespace(data_dir=data))
    return data, registry, observations


@pytest.fixture
def client(table_source):
    defaults = dict(
        team="LA",
        start=2021,
        end=None,
        qb="current",
        sample="starter",
        basis="season",
        players=None,
    )
    return lambda **params: routes.team_analysis(**(defaults | params))


def test_api_selection_empty_serialization_and_validation(client):
    assert client()["risk"]["n"] == 4
    assert client(players="")["risk"]["status"] == "empty_selection"
    for params in [
        {"players": "absent"},
        {"players": "qb,qb"},
        {"qb": "absent"},
        {"team": "bad"},
        {"start": 2026},
        {"start": 2025, "end": 2024},
        {"players": ",".join(str(i) for i in range(9))},
    ]:
        with pytest.raises(HTTPException) as exc:
            client(**params)
        assert exc.value.status_code == 422


def test_api_fails_closed_on_unverified_sources(client, table_source):
    data, _, _ = table_source
    root, _ = load_table(data, "team_player_games", current(data)["tables"]["team_player_games"])
    (root / "data.parquet").write_bytes(b"corrupt")
    with pytest.raises(HTTPException) as exc:
        client()
    assert exc.value.status_code == 409 and "corrupt" in exc.value.detail


@pytest.mark.parametrize(
    "sample,basis,qb,start,end",
    [
        ("starter", "season", "current", 2021, 2025),
        ("active", "raw", "all", 2013, 2025),
        ("starter", "raw", "qb", 2013, 2013),
    ],
)
def test_table_api_preserves_analysis_and_historical_coverage(
    client,
    table_source,
    sample,
    basis,
    qb,
    start,
    end,
):
    _, _, observations = table_source
    expected = analyze_team(
        observations.sort("season", "week", "player_id"),
        team="LA",
        start=start,
        end=end,
        qb=qb,
        sample=sample,
        basis=basis,
        selected=["qb", "wr"],
    )
    result = client(start=start, end=end, sample=sample, basis=basis, qb=qb, players="qb,wr")
    for field, value in expected.items():
        assert result[field] == value
    assert result["report"]["table"] == "analytics.team_player_games"
    assert result["report"]["earliest_season"] == 2013


def test_api_requires_table_and_does_not_fall_back_to_gold(client, table_source):
    data, _, _ = table_source
    (data / "tables/current.json").unlink()
    with pytest.raises(HTTPException) as exc:
        client()
    assert exc.value.status_code == 409 and "not been published" in exc.value.detail


def test_table_publication_changes_token_and_rejects_old_requests(client, table_source):
    data, registry, observations = table_source
    first = routes.team_analysis_catalog()
    assert first["earliest_season"] == 2013
    observations.with_columns((pl.col("league_points") * 2).alias("league_points")).write_parquet(
        data / "source.parquet"
    )
    refresh(data, registry_path=registry)
    with pytest.raises(HTTPException) as exc:
        client(table_version=first["table_version"])
    assert exc.value.status_code == 409 and exc.value.headers["X-Table-Catalog-Stale"] == "true"
    new = routes.team_analysis_catalog()
    assert new["table_version"] != first["table_version"]
    result = client(table_version=new["table_version"])
    assert result["risk"]["mean"] == 16.0


def test_table_replacement_during_calculation_rejects_response(client, table_source, monkeypatch):
    data, registry, _ = table_source
    calculate = routes.analyze_team

    def race(*args, **kwargs):
        result = calculate(*args, **kwargs)
        refresh(data, registry_path=registry, force=True)
        return result

    monkeypatch.setattr(routes, "analyze_team", race)
    with pytest.raises(HTTPException) as exc:
        client()
    assert exc.value.status_code == 409
    assert exc.value.headers == {"X-Table-Catalog-Stale": "true"}


def test_app_gold_change_requires_table_refresh(client, monkeypatch):
    from engine.tables import team_analysis

    monkeypatch.setattr(team_analysis, "current_catalog", lambda _: {"gold": {"version": "new"}})
    with pytest.raises(HTTPException) as exc:
        client()
    assert exc.value.status_code == 409 and "behind the selected data release" in exc.value.detail
