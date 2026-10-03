"""Information-date boundaries and deterministic availability constraints."""

from copy import deepcopy
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from engine.metrics.availability import (
    attach_known_absences,
    audit_absence_constraints,
    coverage_audit,
    load_absences,
    validate_absences,
)
from engine.metrics.fit import (
    FitConfig,
    ModelSpec,
    apply_fitted_models,
    fit_walk_forward,
    models_for_season,
)

LEDGER = Path(__file__).resolve().parents[1] / "data/static/historical_absences.json"


def forecast(player="00-0033923", season=2019, cutoff="2019-07-17", **extra):
    return pl.DataFrame(
        [
            {
                "player_id": player,
                "forecast_season": season,
                "forecast_cutoff_date": date.fromisoformat(cutoff),
                "expected_games": 15.0,
                "projected_availability": 0.95,
                "cutoff_state_resolution": "inferred_prior_team",
                **extra,
            }
        ]
    )


def test_hunt_public_date_not_collection_date_controls_known_absence():
    evidence = load_absences(LEDGER)
    before = attach_known_absences(forecast(cutoff="2019-03-14"), evidence).row(0, named=True)
    after = attach_known_absences(forecast(cutoff="2019-03-15"), evidence).row(0, named=True)
    july = attach_known_absences(forecast(), evidence).row(0, named=True)
    assert before["availability_evidence_status"] == "unknown"
    assert before["known_absence_games"] is None
    assert before["expected_games"] == 15
    assert after["known_absence_games"] == july["known_absence_games"] == 8
    assert july["known_available_games_cap"] == july["expected_games"] == 8
    assert july["cutoff_state_resolution"] == "inferred_prior_team"
    assert july["projected_availability"] == 0.5


@pytest.mark.parametrize(
    "pid,season,cutoff,missed,cap",
    [
        ("00-0034766", 2019, "2019-07-17", 4, 12),  # bye does not serve a suspension
        ("00-0034837", 2022, "2022-08-26", 17, 0),
        ("00-0033537", 2022, "2022-08-10", 6, 11),
        ("00-0033537", 2022, "2022-08-18", 11, 6),
        ("00-0037240", 2023, "2023-08-25", 6, 11),  # September reduction not yet known
        ("00-0037240", 2023, "2023-09-29", 4, 13),
    ("00-0027891", 2019, "2019-07-17", None, None),  # Tate announced July 27
    ("00-0039067", 2025, "2025-08-29", 6, 11),
    ("00-0033891", 2024, "2024-08-30", 5, 12),
    ("00-0034791", 2024, "2024-08-25", None, None),
    ("00-0034791", 2024, "2024-08-30", 4, 13),  # PUP minimum, not guaranteed return
    ],
)
def test_real_cutoffs_and_revisions(pid, season, cutoff, missed, cap):
    row = attach_known_absences(forecast(pid, season, cutoff), load_absences(LEDGER)).row(
        0, named=True
    )
    assert row["known_absence_games"] == missed
    assert row["known_available_games_cap"] == cap


def test_unknown_and_cleared_incident_are_not_verified_healthy():
    event = deepcopy(load_absences(LEDGER)[0])
    event.update(unavailable_games=[], known_on="2019-06-01", source_published_on="2019-06-01")
    row = attach_known_absences(forecast(), [event]).row(0, named=True)
    assert row["availability_evidence_status"] == "specific_absence_cleared"
    assert row["known_available_games_cap"] is None
    unrelated = attach_known_absences(forecast(player="other"), [event]).row(0, named=True)
    assert unrelated["known_absence_games"] is None


def test_overlapping_games_and_same_day_conflicts_are_order_independent():
    first = deepcopy(load_absences(LEDGER)[0])
    second = {
        **first,
        "evidence_id": "another",
        "absence_id": "injury",
        "kind": "injury",
        "unavailable_games": [7, 8, 9],
    }
    row = attach_known_absences(forecast(), [first, second]).row(0, named=True)
    assert row["known_absence_games"] == 9
    assert row["known_suspension_games"] == 8
    conflicting = {**first, "evidence_id": "conflicting", "unavailable_games": [1, 2]}
    a = attach_known_absences(forecast(), [first, conflicting])
    b = attach_known_absences(forecast(), [conflicting, first])
    assert a.equals(b)
    assert a["known_available_games_cap"][0] is None
    assert a["availability_evidence_status"][0] == "conflicting_evidence"


@pytest.mark.parametrize(
    "changes",
    [
        {"unavailable_games": [1, 1]},
        {"unavailable_games": [17]},
        {"known_on": "2019-03-14"},
        {"source_url": ""},
        {"player_id": ""},
    ],
)
def test_bad_evidence_fails_closed(changes):
    event = {**load_absences(LEDGER)[0], **changes}
    with pytest.raises(ValueError):
        validate_absences([event])


def test_revised_ledger_can_relax_baseline_cap_without_double_discounting():
    events = load_absences(LEDGER)
    first = attach_known_absences(forecast(expected_games=6.0), events)
    assert first["expected_games"][0] == 6  # do not subtract eight again
    assert attach_known_absences(first, events).equals(first)
    capped = attach_known_absences(forecast(), events)
    restored = attach_known_absences(capped, [])
    assert restored["expected_games"][0] == 15
    assert restored["projected_availability"][0] == 0.95


def test_caps_flow_through_walk_forward_products_selectors_and_live_scoring():
    rows = [
        {
            "forecast_season": season,
            "player_id": "00-0033923",
            "position": "RB",
            "outcome_complete": season < 2019,
            "actual_games": 16.0,
            "actual_ppg": 20.0,
            "actual_season_points": 320.0,
            "signal": 1.0,
        }
        for season in range(2015, 2020)
    ]
    inputs = attach_known_absences(pl.DataFrame(rows), load_absences(LEDGER))
    specs = (
        ModelSpec(name="ppg", features=("signal",), lambda_grid=(1.0,)),
        ModelSpec(name="games", target="actual_games", features=("signal",), lambda_grid=(1.0,)),
        ModelSpec(name="points", kind="product", factors=("ppg", "games")),
        ModelSpec(name="selected", kind="coalesce", factors=("points",)),
    )
    config = FitConfig(models=specs, min_train_folds=2, per_position=False, positions=("RB",))
    fitted, records = fit_walk_forward(inputs, config)
    july = fitted.filter(pl.col("forecast_season") == 2019)
    assert july["games"][0] == 8
    assert july["points"][0] == july["selected"][0] == 160
    live = apply_fitted_models(inputs.tail(1), models_for_season(records, 2019), config)
    assert live["games"][0] == 8
    assert live["points"][0] == 160
    assert live["selected"][0] == 160
    assert audit_absence_constraints(fitted, load_absences(LEDGER), ["games"])["accepted"]
    tampered = fitted.with_columns(pl.lit(17.0).alias("games"))
    assert not audit_absence_constraints(tampered, load_absences(LEDGER), ["games"])["accepted"]


def test_roster_coverage_and_absence_evidence_remain_separate():
    frame = attach_known_absences(forecast(player="unknown", cutoff_suspended=1.0), [])
    report, queue = coverage_audit(frame)
    assert report["availability_evidence"]["unknown"] == 1
    assert report["roster_evidence"]["inferred_prior_team"] == 1
    assert not report["coverage_complete"]
    assert "restriction_duration_or_classification_missing" in queue["review_reasons"][0]


def test_missing_ledger_is_not_silently_accepted(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_absences(tmp_path / "missing.json")


def test_known_full_season_absence_forces_zero_games_points_and_return_probability():
    rows = [
        {
            "forecast_season": season,
            "player_id": "00-0034837",
            "position": "WR",
            "outcome_complete": season < 2022,
            "actual_games": 17.0,
            "actual_ppg": 10.0,
            "actual_season_points": 170.0,
            "signal": 1.0,
        }
        for season in range(2018, 2023)
    ]
    inputs = attach_known_absences(pl.DataFrame(rows), load_absences(LEDGER))
    config = FitConfig(
        models=tuple(
            ModelSpec(name=name, target=target, features=("signal",), lambda_grid=(1.0,))
            for name, target in (
                ("games", "actual_games"),
                ("direct", "actual_season_points"),
                ("return_prob", "actual_played"),
            )
        ),
        min_train_folds=2,
        per_position=False,
        positions=("WR",),
    )
    fitted, _ = fit_walk_forward(inputs, config)
    assert fitted.tail(1).select("games", "direct", "return_prob").row(0) == (0, 0, 0)
    assert fitted["actual_season_points"].to_list() == inputs["actual_season_points"].to_list()
