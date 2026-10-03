import polars as pl

from patron.metrics.current_rankings import make_panel
from patron.metrics.weekly_points import add_context, choose, fit_fold
from tests.test_current_rankings import panel_inputs


def test_one_week_target_and_future_suffix_invariance():
    inputs = panel_inputs()
    schedule = inputs[2].with_columns(pl.lit("2005-09-01").alias("gameday"))
    rows = make_panel(*inputs, season=2006, through_week=2, horizons=("next_week",))
    add_context(rows, inputs[1].to_dicts(), schedule.to_dicts())
    changed = inputs[1].to_dicts()
    changed[-1].update(league_points=9999, targets=999, carries=999, team="NYJ")
    changed.append(dict(changed[-1], player_id="future_only"))
    revised = make_panel(
        inputs[0],
        pl.DataFrame(changed),
        *inputs[2:],
        season=2006,
        through_week=2,
        horizons=("next_week",),
    )
    add_context(revised, changed, schedule.to_dicts())

    def strip(row):
        return {k: v for k, v in row.items() if k not in ("actual", "observed_actual")}

    assert [strip(r) for r in rows] == [strip(r) for r in revised]
    assert all(r["end_week"] == 3 for r in rows)
    assert any(r["actual"] != z["actual"] for r, z in zip(rows, revised, strict=True))


def test_fitting_ignores_test_outcomes_and_known_bye_is_zero():
    train = [
        dict(
            prior_rate=float(i % 5),
            current_rate=float(i % 3),
            elapsed_games=2,
            scheduled_games=1,
            x_last3_points=float(i % 4),
            x_volume=float(i),
            actual=float(i % 8),
        )
        for i in range(90)
    ]
    test = [dict(train[0], actual=9999), dict(train[1], scheduled_games=0)]
    original = fit_fold(train, test)
    test[0]["actual"] = -12345
    assert fit_fold(train, test) == original
    assert all(values[1] == 0 for values in original.values())


def test_selection_waits_for_prior_evaluation_seasons():
    row = dict(
        season=2008, actual=1, prior=2, current=3, blend=2, last3=2, weekly_ridge=1, weekly_boost=1
    )
    assert choose([]) == ("blend", "blend")
    reference, policy = choose([row])
    assert reference == policy
    earlier = [dict(row, season=y) for y in (2008, 2009, 2010)]
    assert choose(earlier)[1] in ("weekly_ridge", "weekly_boost")
