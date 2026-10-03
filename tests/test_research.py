"""Residual discovery must confirm a pattern in non-overlapping eras."""

from __future__ import annotations

import polars as pl

from engine.metrics.research import analyze_residual_patterns


def test_residual_pattern_requires_same_direction_in_every_era() -> None:
    rows = []
    for season in range(2007, 2026):
        for index in range(24):
            signal = float(index)
            # Stable signal always explains an underprediction; the unstable field
            # reverses after the discovery period and must not survive the screen.
            stable_error = signal / 10.0
            unstable = signal if season <= 2014 else -signal
            rows.append(
                {
                    "forecast_season": season,
                    "outcome_complete": True,
                    "position": "RB",
                    "preseason_rostered": 1.0,
                    "stable_signal": signal,
                    "unstable_signal": unstable,
                    "actual_ppg": 10.0 + stable_error,
                    "fitted_ppg": 10.0,
                    "actual_games": 12.0 + stable_error,
                    "fitted_games": 12.0,
                    "fitted_status_games": 12.0,
                    "actual_season_points": 120.0 + stable_error,
                    "fitted_season_points": 120.0,
                }
            )

    results = analyze_residual_patterns(
        pl.DataFrame(rows), ["stable_signal", "unstable_signal"]
    )
    selected = {
        (row["population"], row["residual"], row["metric"]): row
        for row in results
        if row["position"] == "RB"
    }

    stable = selected[("cutoff_rostered", "ppg", "stable_signal")]
    assert stable["stable"]
    assert stable["same_direction"]
    assert stable["stability_floor"] == 1.0

    unstable = selected[("cutoff_rostered", "ppg", "unstable_signal")]
    assert not unstable["stable"]
    assert not unstable["same_direction"]
