"""Calibration-declared range disclosures, without changing sealed v3 inference code."""

from engine.metrics.weekly_distribution import forecast_snapshot


def forecast_preview(snapshot, players, identities, schedule, calibration):
    result = forecast_snapshot(snapshot, players, identities, schedule, calibration)
    note = calibration["range_reliability_note"]
    result["range_reliability_note"] = note
    result["limitations"] = [
        note,
        *[
            text
            for text in result["limitations"]
            if "v3 study" not in text and "K/DST scales" not in text
        ],
    ]
    for game in result["matchups"]:
        for side in (game["home"], game["away"]):
            for row in side["forecasts"]:
                if row.get("basis") == "Opening Drive v3 weekly distribution":
                    row["basis"] = (
                        f"{calibration.get('campaign_name', 'Opening Drive')} weekly distribution"
                    )
            if calibration.get("mixed_horizons"):
                side.update(
                    prediction_interval_80=None,
                    independent_interval_80=None,
                    prediction_std=None,
                    offense_only_interval_80=None,
                    uncertainty_status="withheld_mixed_forecast_timing",
                )
    return result
