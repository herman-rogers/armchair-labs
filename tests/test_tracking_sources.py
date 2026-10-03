import json
import zipfile

import polars as pl
import pytest

from patron.data.releases import load_gold
from patron.data.tracking import (
    build,
    capture,
    extract_zip,
    load_source,
    normalize_tracking,
    register,
)
from patron.data.tracking_sources import SOURCES


def frames():
    return pl.DataFrame(
        {
            "gameId": [2017090700, 2017090700],
            "playId": [44, 44],
            "nflId": [123, None],
            "frame.id": [1, 1],
            "x": [20.0, 21.0],
            "y": [10.0, 11.0],
            "team": ["home", "ball"],
            "playDirection": ["left", "left"],
        }
    )


def test_coordinates_identity_and_missing_observations():
    f = normalize_tracking(frames(), "sample", "tracking.csv")
    assert f["x_attack"].to_list() == [100, 99]
    assert f["y_attack"][0] == pytest.approx(160 / 3 - 10)
    assert f["entity_id"].to_list() == ["nfl:123", "football"]
    assert f["acceleration"].null_count() == 2
    unknown = normalize_tracking(frames().drop("playDirection"), "sample", "tracking.csv")
    assert unknown["x_attack"].null_count() == 2
    missing = normalize_tracking(
        frames().with_columns(pl.lit(None).alias("x")), "sample", "tracking.csv"
    )
    assert not missing["coordinates_available"].any()
    assert missing["x"].null_count() == 2
    outside = normalize_tracking(
        frames().with_columns(pl.lit(-1.0).alias("x")), "sample", "tracking.csv"
    )
    assert outside["outside_field"].all()


def test_unknown_player_is_not_reinterpreted_as_ball():
    with pytest.raises(ValueError, match="Unknown tracking identity"):
        normalize_tracking(
            frames().with_columns(pl.lit(None).alias("nflId")), "sample", "tracking.csv"
        )


def test_duplicates_and_nonfinite_fail_closed():
    with pytest.raises(ValueError, match="Duplicate tracking"):
        normalize_tracking(pl.concat([frames(), frames()]), "sample", "tracking.csv")
    with pytest.raises(ValueError, match="Nonfinite x"):
        normalize_tracking(
            frames().with_columns(pl.lit(float("inf")).alias("x")), "sample", "tracking.csv"
        )


def test_2026_input_output_clocks_are_separate():
    f = (
        frames()
        .filter(pl.col("nflId").is_not_null())
        .rename(
            {"gameId": "game_id", "playId": "play_id", "nflId": "nfl_id", "frame.id": "frame_id"}
        )
    )
    a = normalize_tracking(f, "bdb_2026", "train/input_2023_w01.csv")
    b = normalize_tracking(f, "bdb_2026", "train/output_2023_w01.csv")
    assert a["phase"].item() == "input"
    assert b["phase"].item() == "output"
    assert a["frame_id"].item() == b["frame_id"].item()


def test_handoff_is_a_snapshot_not_an_invented_trajectory():
    f = frames().filter(pl.col("nflId").is_not_null()).drop("frame.id")
    result = normalize_tracking(f, "bdb_2020", "train.csv", handoff=True)
    assert result["phase"].item() == "handoff"
    assert result["id_namespace"].item() == "legacy_nfl"
    assert result["frame_id"].item() == 0


def test_roundtrip_seals_dependencies_and_leaves_serving_catalog_untouched(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    current = data / "current.json"
    current.write_text('{"untouched": true}')
    local = tmp_path / "input"
    local.mkdir()
    frames().write_csv(local / "tracking.csv")
    source = SOURCES["bdb_2019_sample"]
    raw = capture(data, source, "test_v1", local)
    ref = build(data, raw)
    register(data, source, ref)
    release = load_source(data, source.id)
    assert release.read("tracking").height == 2
    assert current.read_text() == '{"untouched": true}'
    assert build(data, raw) == ref
    # Editing the original import cannot edit sealed provider evidence.
    (local / "tracking.csv").write_text("changed")
    assert load_gold(data, "test_v1").read("tracking").height == 2
    manifest = json.loads((data / "raw/snapshots/test_v1/manifest.json").read_text())
    obj = data / manifest["assets"]["tracking.csv"]["object"]
    obj.write_text("tampered")
    with pytest.raises(ValueError, match="artifact changed"):
        load_source(data, source.id)


def test_ngs_season_summary_never_becomes_weekly_row(tmp_path):
    local = tmp_path / "input"
    local.mkdir()
    pl.DataFrame(
        {
            "season": [2025, 2025],
            "season_type": ["REG", "REG"],
            "week": [0, 1],
            "player_gsis_id": ["00-001", "00-001"],
            "attempts": [100, 10],
        }
    ).write_parquet(local / "ngs_passing.parquet")
    data = tmp_path / "data"
    raw = capture(data, SOURCES["nflverse_ngs"], "ngs_v1", local)
    build(data, raw)
    release = load_gold(data, "ngs_v1")
    assert release.read("ngs_passing_weekly")["week"].to_list() == [1]
    assert release.read("ngs_passing_season")["week"].to_list() == [0]


def test_metadata_only_and_overlapping_archives_do_not_publish(tmp_path):
    local = tmp_path / "input"
    local.mkdir()
    pl.DataFrame({"gameId": [1]}).write_csv(local / "games.csv")
    source = SOURCES["bdb_2025"]
    data = tmp_path / "data"
    with pytest.raises(ValueError, match="No spatial tracking"):
        capture(data, source, "metadata", local)
    frames().write_csv(local / "tracking_week_1.csv")
    frames().write_csv(local / "tracking_week_2.csv")
    raw = capture(data, source, "overlap", local)
    with pytest.raises(ValueError, match="Overlapping tracking"):
        build(data, raw)
    assert not (data / "source_catalog.json").exists()
    assert not (data / "gold/releases/overlap/manifest.json").exists()


def test_zip_traversal_and_size_limit(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("../escape.csv", "bad")
    with pytest.raises(ValueError, match="Unsafe archive"):
        extract_zip(archive, tmp_path / "extract")
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("safe.csv", "abcdef")
    with pytest.raises(ValueError, match="byte limit"):
        extract_zip(archive, tmp_path / "extract", max_bytes=2)


def test_identity_namespace_and_ambiguity_are_not_guessed():
    from patron.data.tracking import link_tracking_players, player_crosswalk

    ids = player_crosswalk(
        pl.DataFrame(
            {
                "gsis_it_id": ["123", "456", "456"],
                "gsis_id": ["00-001", "00-002", "00-003"],
            }
        )
    )
    assert ids.filter(pl.col("nfl_id") == "456")["player_id"].item() is None

    class Rosters:
        def read(self, name):
            assert name == "tracking_player_ids"
            return ids

    modern = normalize_tracking(frames(), "bdb_2025", "tracking.csv")
    legacy = normalize_tracking(frames(), "bdb_2019_sample", "tracking.csv")
    assert link_tracking_players(modern.lazy(), Rosters()).collect()["player_id"][0] == "00-001"
    assert link_tracking_players(legacy.lazy(), Rosters()).collect()["player_id"].null_count() == 2


def test_download_receipts_reuse_verified_files_and_reject_truncation(tmp_path):
    from patron.data.tracking import fetch_asset

    class Response:
        headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def raise_for_status(self):
            pass

        def iter_content(self, size):
            yield b"abc"

    class Session:
        calls = 0

        def get(self, *args, **kwargs):
            self.calls += 1
            return Response()

    session = Session()
    spec = {"name": "a.csv", "url": "https://example.test/a.csv", "size": 3}
    target = fetch_asset(session, spec, tmp_path)
    assert fetch_asset(session, spec, tmp_path) == target
    assert session.calls == 1
    target.write_bytes(b"bad!")
    assert fetch_asset(session, spec, tmp_path).read_bytes() == b"abc"
    assert session.calls == 2
    with pytest.raises(ValueError, match="Truncated"):
        fetch_asset(session, {**spec, "name": "short.csv", "size": 4}, tmp_path)
    assert not (tmp_path / "short.csv").exists()


def test_prediction_test_input_is_spatial_and_keeps_test_split():
    from patron.data.tracking import is_spatial_file

    source = SOURCES["bdb_2026_prediction"]
    assert is_spatial_file("test_input.csv", source)
    assert not is_spatial_file("test.csv", source)
    f = normalize_tracking(frames(), source.id, "test_input.csv")
    assert f["phase"].unique().to_list() == ["input"]
    assert f["split"].unique().to_list() == ["test"]


def test_historical_weather_text_is_preserved(tmp_path):
    from patron.data.tracking import read_source

    path = tmp_path / "train.csv"
    path.write_text("WindSpeed,WindDirection,x\n" + "10,180,20.0\n" * 10001 + "SSW,North,21.0\n")
    frame = read_source(path, "train.csv")
    assert frame["WindSpeed"][-1] == "SSW"
    assert frame["WindDirection"][-1] == "North"
    assert frame["x"][-1] == 21.0
