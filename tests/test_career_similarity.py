import polars as pl
import pytest
from fastapi import HTTPException

from patron.metrics.career_similarity import METRICS, career_neighbors
from patron.metrics.career_stat_evidence import ZERO_STAT_REVIEWS, apply_zero_reviews


def fixture():
    people = []
    weeks = []
    for i in range(10):
        entry = 2010 if i == 0 else 2000 + i
        people.append(
            dict(
                player_id=str(i),
                player_display_name=f"Player {i}",
                position="WR",
                rookie_season=entry,
                history_left_truncated=False,
            )
        )
        for year in range(2):
            for week in range(1, 9):
                rate = (year + 1) * 10.0 + (0 if i in (0, 1) else i)
                weeks.append(
                    dict(
                        player_id=str(i),
                        season=entry + year,
                        week=week,
                        stat_recorded=True,
                        offense_snaps=20,
                        targets=rate,
                        receptions=rate / 2,
                        receiving_yards=rate * 10,
                    )
                )
    return pl.DataFrame(people), pl.DataFrame(weeks)


def test_same_stage_match_ignores_later_career_and_self():
    people, weeks = fixture()
    result = career_neighbors(people, weeks, "0", 2011)
    assert result["matches"][0]["player_id"] == "1"
    assert result["matches"][0]["distance"] == 0
    assert result["seasons_compared"] == 2
    assert all(p["player_id"] != "0" for p in result["matches"])
    future = weeks.filter(pl.col("player_id") == "0").with_columns(pl.lit(2020).alias("season"))
    later_peer = weeks.filter(pl.col("player_id") == "1").with_columns(
        pl.lit(2011).alias("season"), pl.lit(99999.0).alias("receiving_yards")
    )
    assert (
        career_neighbors(
            people, pl.concat([weeks, future, later_peer], how="vertical_relaxed"), "0", 2011
        )
        == result
    )


def test_wrong_position_truncation_missing_stats_and_small_samples_excluded():
    people, weeks = fixture()
    people = people.with_columns(
        pl.when(pl.col("player_id") == "2")
        .then(pl.lit("QB"))
        .otherwise(pl.col("position"))
        .alias("position"),
        (pl.col("player_id") == "3").alias("history_left_truncated"),
    )
    weeks = weeks.with_columns(
        pl.when(pl.col("player_id") == "4").then(None).otherwise(pl.col("targets")).alias("targets")
    )
    weeks = weeks.filter(~((pl.col("player_id") == "5") & (pl.col("week") == 8)))
    result = career_neighbors(people, weeks, "0", 2011)
    assert result["eligible_peers"] == 5
    assert {r["player_id"] for r in result["matches"]} == {"1", "6", "7", "8", "9"}


def test_no_completed_rookie_season_and_missing_target_are_explicit():
    people, weeks = fixture()
    assert "No completed" in career_neighbors(people, weeks, "0", 2009)["reason"]
    missing = weeks.with_columns(
        pl.when(pl.col("player_id") == "0").then(None).otherwise(pl.col("targets")).alias("targets")
    )
    assert career_neighbors(people, missing, "0", 2011)["matches"] == []


def test_api_rejects_future_cutoff(monkeypatch, tmp_path):
    from patron.api import profile_routes

    monkeypatch.setattr(
        profile_routes, "load_profiles", lambda: (tmp_path, {"season": 2026, "through_week": 2})
    )
    with pytest.raises(HTTPException) as exc:
        profile_routes.similar_players("0", season=2026, week=3)
    assert exc.value.status_code == 422


@pytest.mark.parametrize("position", METRICS)
def test_small_gaps_preserve_comparisons_for_every_position_without_inventing_zeros(position):
    people, weeks = fixture()
    people = people.with_columns(pl.lit(position).alias("position"))
    weeks = weeks.with_columns(*[pl.col("targets").alias(m) for m in METRICS[position]])
    # Eight known weeks plus two snap-only weeks: exactly 80% coverage.
    unknown = weeks.filter(pl.col("week") <= 2).with_columns(
        (pl.col("week") + 8).alias("week"),
        pl.lit(False).alias("stat_recorded"),
        *[pl.lit(None, dtype=pl.Float64).alias(m) for m in METRICS[position]],
    )
    result = career_neighbors(people, pl.concat([weeks, unknown]), "0", 2011)
    assert result["reason"] is None
    assert result["eligible_peers"] == 9
    assert result["matches"][0]["distance"] == 0
    target = result["target"]
    assert target["observed_weeks"] == 20
    assert target["comparison_weeks"] == 16
    assert target["verified_zero_weeks"] == 0
    first = target["seasons"][0]
    assert first["missing_weeks"] == [9, 10]
    assert first["coverage_fraction"] == 0.8
    assert all(first[m] == 10 for m in METRICS[position])
    # One additional uncovered week crosses the floor; data remains visible.
    extra = unknown.filter((pl.col("player_id") == "0") & (pl.col("week") == 9))
    extra = extra.with_columns(pl.lit(11).alias("week"))
    blocked = career_neighbors(
        people, pl.concat([weeks, unknown, extra], how="vertical_relaxed"), "0", 2011
    )
    assert blocked["matches"] == []
    assert "2010: 8/11 weeks covered" in blocked["reason"]
    assert blocked["eligible_peers"] == 9
    assert blocked["coverage"][0]["missing_weeks"] == [9, 10, 11]


def test_metrics_share_one_denominator_and_nonfinite_values_are_missing():
    people, weeks = fixture()
    weeks = weeks.with_columns((pl.col("week") + 2).alias("week"))
    extra = weeks.filter(pl.col("week") <= 4).with_columns(
        (pl.col("week") - 2).alias("week"),
        pl.when(pl.col("week") == 3).then(None).otherwise(99999).alias("targets"),
        pl.when(pl.col("week") == 4).then(float("inf")).otherwise(99999).alias("receptions"),
        pl.lit(99999.0).alias("receiving_yards"),
    )
    result = career_neighbors(people, pl.concat([weeks, extra], how="vertical_relaxed"), "0", 2011)
    first = result["target"]["seasons"][0]
    assert first["comparison_weeks"] == 8
    assert first["missing_weeks"] == [1, 2]
    assert first["targets"] == 10
    assert first["receptions"] == 5
    assert first["receiving_yards"] == 100


def test_fewer_than_eight_covered_weeks_remains_ineligible():
    people, weeks = fixture()
    weeks = weeks.with_columns(
        pl.when((pl.col("player_id") == "0") & (pl.col("week") == 8))
        .then(None)
        .otherwise(pl.col("targets"))
        .alias("targets")
    )
    result = career_neighbors(people, weeks, "0", 2011)
    assert "2010: 7/8 weeks covered" in result["reason"]
    assert result["matches"] == []


def reviewed_week():
    return dict(
        player_id="00-0034857",
        season=2024,
        week=18,
        team="BUF",
        offense_snaps=1,
        stat_recorded=False,
        attempts=None,
        passing_yards=None,
        carries=None,
        rushing_yards=None,
    )


def test_reviewed_zero_is_distinct_from_an_unknown_or_a_recorded_stat():
    zero = reviewed_week()
    unknown = {**zero, "week": 17}
    frame = pl.DataFrame([zero, unknown])
    result, reviews = apply_zero_reviews(frame, METRICS["QB"])
    assert result["attempts"].to_list() == [0, None]
    assert result["verified_zero"].to_list() == [True, False]
    assert result["stat_recorded"].to_list() == [False, False]
    assert reviews == [ZERO_STAT_REVIEWS[0]]
    assert frame["attempts"].null_count() == 2


@pytest.mark.parametrize(
    "change",
    [
        {"team": "NYJ"},
        {"offense_snaps": 2},
        {"player_id": "someone-else"},
        {"stat_recorded": True},
        {"attempts": 1.0},
    ],
)
def test_review_does_not_override_conflicting_source_evidence(change):
    frame = pl.DataFrame([{**reviewed_week(), **change}])
    result, reviews = apply_zero_reviews(frame, METRICS["QB"])
    assert reviews == []
    assert result["verified_zero"].to_list() == [False]
    assert result.drop("verified_zero").equals(frame)


def test_verified_zero_enters_denominator_and_respects_cutoff():
    people, weeks = fixture()
    people = people.with_columns(
        pl.when(pl.col("player_id") == "0")
        .then(pl.lit("00-0034857"))
        .otherwise(pl.col("player_id"))
        .alias("player_id"),
        (pl.col("rookie_season") + 14).alias("rookie_season"),
        pl.lit("QB").alias("position"),
    )
    weeks = weeks.with_columns(
        pl.when(pl.col("player_id") == "0")
        .then(pl.lit("00-0034857"))
        .otherwise(pl.col("player_id"))
        .alias("player_id"),
        (pl.col("season") + 14).alias("season"),
        pl.lit("BUF").alias("team"),
        *[pl.col("targets").alias(m) for m in METRICS["QB"]],
    )
    zeros = pl.DataFrame([reviewed_week(), {**reviewed_week(), "season": 2025}])
    with_zeros = pl.concat([weeks, zeros], how="diagonal_relaxed")
    result = career_neighbors(people, with_zeros, "00-0034857", 2024)
    first = result["target"]["seasons"][0]
    assert first["comparison_weeks"] == 9
    assert first["attempts"] == pytest.approx(80 / 9)
    assert result["target"]["zero_stat_reviews"] == [ZERO_STAT_REVIEWS[0]]
    assert result["target"]["verified_zero_weeks"] == 1
    without_future = with_zeros.filter(pl.col("season") <= 2024)
    assert career_neighbors(people, without_future, "00-0034857", 2024) == result


def test_small_peer_pool_retains_target_and_coverage():
    people, weeks = fixture()
    result = career_neighbors(people.head(3), weeks, "0", 2011)
    assert "Fewer than five" in result["reason"]
    assert result["eligible_peers"] == 2
    assert result["target"]["comparison_weeks"] == 16
    assert len(result["coverage"]) == 2


def test_api_returns_coverage_and_honors_completed_season_cutoff(monkeypatch, tmp_path):
    from patron.api import profile_routes

    people, weeks = fixture()
    people.write_parquet(tmp_path / "players.parquet")
    weeks.write_parquet(tmp_path / "nfl_weeks.parquet")
    monkeypatch.setattr(
        profile_routes,
        "load_profiles",
        lambda: (tmp_path, {"version": "fixture", "season": 2012, "through_week": 2}),
    )
    result = profile_routes.similar_players("0", season=2011, week=18)
    assert result["version"] == "fixture"
    assert result["method_version"] == 2
    assert result["complete_through"] == 2011
    assert result["target"]["comparison_weeks"] == 16
    assert [r["season"] for r in result["coverage"]] == [2010, 2011]
    partial = profile_routes.similar_players("0", season=2011, week=17)
    assert partial["complete_through"] == 2010
    assert [r["season"] for r in partial["coverage"]] == [2010]
