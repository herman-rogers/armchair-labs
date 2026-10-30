import pytest
from experiments.future_player_lab.campaigns.weekly_v4 import horizons
from experiments.future_player_lab.tests.test_weekly_v4_horizons import live_inputs
from experiments.future_player_lab.tests.test_weekly_v4_release import mutate, release

# Publication only warns; the coherence checks themselves still raise.
from engine.chairstats.horizons import _validate as validate_opening_drive_release


def test_pregame_injury_changes_current_components_without_extending_absence(tmp_path, monkeypatch):
    # Reuse a minimal accepted-source fixture with frozen fit and past observations.
    players, panel, snapshot, fitted = live_inputs.__wrapped__(tmp_path, monkeypatch)
    early = horizons.build_projection(players, panel, snapshot, fitted)
    players.loc[players.player_id.eq("qb"), "prediction"] = 0
    players.loc[players.player_id.eq("qb"), "report_out"] = 1
    players.loc[players.player_id.eq("qb"), "room_out_usage"] = 1000
    late = horizons.build_projection(players, panel, snapshot, fitted)
    for horizon in ("remaining", "season"):
        original, updated = early[horizon][0], late[horizon][0]
        assert (
            original["projection"]["future_weeks"][1:] == updated["projection"]["future_weeks"][1:]
        )
        assert original["predictions"]["points"] - updated["predictions"]["points"] == 12
        assert original["predictions"]["attempts"] - updated["predictions"]["attempts"] == 1
    assert early["weekly"][1] == late["weekly"][1]


def test_release_rejects_current_components_that_disagree_with_curve(tmp_path):
    manifest = release(tmp_path)
    mutate(tmp_path, "rb_weekly.json", lambda row: row["predictions"].update(targets=9000))
    with pytest.raises(ValueError, match="[Cc]urrent.*component|component.*weekly"):
        validate_opening_drive_release(tmp_path, manifest)


@pytest.mark.parametrize("samples", [[float("inf")], [float("nan")], [True], ["5"]])
def test_release_rejects_nonfinite_or_nonnumeric_distributions(tmp_path, samples):
    manifest = release(tmp_path)
    mutate(tmp_path, "rb_weekly.json", lambda row: row["distribution"].update(samples=samples))
    with pytest.raises(ValueError, match="finite numeric samples"):
        validate_opening_drive_release(tmp_path, manifest)


def test_release_rejects_nonzero_bye_components(tmp_path):
    manifest = release(tmp_path)

    def bye(row):
        row["scheduled_games"] = 0
        row["predictions"] = {target: 0 for target in row["predictions"]}
        row["predictions"]["targets"] = 2
        row["distribution"] = {"samples": [0, 0]}

    mutate(tmp_path, "rb_weekly.json", bye)
    with pytest.raises(ValueError, match="bye component"):
        validate_opening_drive_release(tmp_path, manifest)
