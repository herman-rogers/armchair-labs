"""Pure projection arithmetic shared by historical evaluation and publication."""

from __future__ import annotations

import math

HALF_LIVES = (0, 2, 4, 8, 10000)


def future_curve(games, forecast_week, weekly, anchor, signal, half_life):
    """Use the saved weekly mean now; fade early-week information toward a long anchor.

    `games` contains only games known on the player's current team's schedule.
    Future observations, later injury reports and later weekly forecasts are never inputs.
    Zero half-life means that only the currently forecast week overrides the anchor.
    """
    if len(games) != len(set(games)) or any(w < forecast_week for w in games):
        raise ValueError("Future schedule contains duplicate or past weeks")
    if half_life not in HALF_LIVES or not all(math.isfinite(v) for v in (weekly, anchor, signal)):
        raise ValueError("Invalid projection policy or nonfinite points")
    result = []
    for week in sorted(games):
        if week == forecast_week:
            points, weight, basis = weekly, 1.0, "saved_weekly_forecast"
        else:
            weight = 2 ** (-(week - forecast_week) / half_life) if half_life else 0.0
            points = (1 - weight) * anchor + weight * signal
            basis = "fixed_origin_decay"
        result.append(dict(week=int(week), points=float(points), signal_weight=weight, basis=basis))
    return result


def durable_signal(early, anchor):
    """Do not extend this week's injury designation or teammate transfer into later weeks."""
    report_columns = ("report_known", "room_out_usage", "room_questionable_usage", "room_dnp_usage")
    affected = any(float(early.get(c, 0) or 0) > 0 for c in report_columns)
    return float(anchor if affected else early["prediction"])
