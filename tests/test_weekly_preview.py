"""Calibration-declared disclosures and withheld uncertainty must not change point totals."""

from engine.metrics.weekly_preview import forecast_preview
from tests.test_weekly_lineup import schedule, snapshot


def test_preview_keeps_points_and_withholds_unvalidated_special_and_mixed_ranges():
    players = [dict(player_id="qb-allen", position="QB", team="BUF", prediction=20.0, std=5.0)]
    calibration = dict(
        campaign_id="opening_drive_v32",
        campaign_name="Opening Drive v3.2 preview",
        special_teams={},
        range_reliability_note="V3.2 preview ranges; prospective evidence pending.",
    )
    result = forecast_preview(snapshot(), players, {11: "qb-allen"}, schedule(), calibration)
    home = result["matchups"][0]["home"]
    assert home["model_projection"] == 29
    assert home["prediction_interval_80"] is None
    assert home["offense_only_interval_80"] is not None
    assert any(
        r["basis"] == "Opening Drive v3.2 preview weekly distribution" for r in home["forecasts"]
    )
    assert all("v3 study" not in text for text in result["limitations"])
    mixed = forecast_preview(
        snapshot(), players, {11: "qb-allen"}, schedule(), dict(calibration, mixed_horizons=True)
    )["matchups"][0]["home"]
    assert mixed["model_projection"] == 29
    assert mixed["offense_only_interval_80"] is None
    assert mixed["prediction_std"] is None
    assert mixed["uncertainty_status"] == "withheld_mixed_forecast_timing"
