import json

import pytest
from experiments.future_player_lab.campaigns.weekly_v4.horizons import TARGETS

# Publication only warns; the coherence checks themselves still raise.
from engine.chairstats.horizons import _validate as validate_opening_drive_release


def release(root):
    manifest = dict(
        campaign_id="opening_drive_v4",
        base_campaign="opening_drive_v32",
        horizon_contract=dict(version=1, components_required=True),
        models=[],
    )
    for position, targets in TARGETS.items():
        projection = dict(
            anchor_basis="v3_learned_residual",
            future_weeks=[
                dict(week=8, points=5, components={t: 2 for t in targets}),
                dict(week=9, points=6, components={t: 3 for t in targets}),
            ],
            remaining_points=11,
            actual_points_to_date=10,
            actual_components_to_date={t: 1 for t in targets},
            remaining_scheduled_games=2,
            season_scheduled_games=9,
        )
        for horizon, points, count, component in (
            ("weekly", 5, 1, 2),
            ("remaining", 11, 2, 5),
            ("season", 21, 9, 6),
        ):
            name = f"{position.lower()}_{horizon}"
            row = dict(
                player_id=position,
                position=position,
                base_campaign="opening_drive_v32",
                predictions=dict(points=points, **{t: component for t in targets}),
                scheduled_games=count,
                points_per_scheduled_game=points / count,
                distribution=dict(samples=[points - 1, points + 1]),
            )
            if horizon != "weekly":
                row["projection"] = projection
            payload = dict(
                rows=[row],
                capture_hashes={"same": "origin"},
                forecast_kind="coherent_schedule_projection",
            )
            (root / f"{name}.json").write_text(json.dumps(payload))
            manifest["models"].append(
                dict(
                    model_id=name,
                    position=position,
                    horizon=horizon,
                    artifacts=dict(projection_policy={}, projection_evaluation={}),
                    forecasts=[dict(season=2026, origin_week=7, path=f"{name}.json")],
                )
            )
    return manifest


def mutate(root, name, callback):
    path = root / name
    payload = json.loads(path.read_text())
    callback(payload["rows"][0])
    path.write_text(json.dumps(payload))


def test_v4_release_retains_coherent_components_and_distributions(tmp_path):
    validate_opening_drive_release(tmp_path, release(tmp_path))


@pytest.mark.parametrize("horizon", ["weekly", "remaining", "season"])
def test_v4_rejects_missing_live_component(tmp_path, horizon):
    manifest = release(tmp_path)
    mutate(tmp_path, f"rb_{horizon}.json", lambda r: r["predictions"].pop("targets"))
    with pytest.raises(ValueError, match="present in every horizon"):
        validate_opening_drive_release(tmp_path, manifest)


def test_v4_rejects_component_total_and_distribution_drift(tmp_path):
    manifest = release(tmp_path)
    mutate(tmp_path, "rb_remaining.json", lambda r: r["predictions"].update(targets=100))
    with pytest.raises(ValueError, match="[Cc]omponent totals"):
        validate_opening_drive_release(tmp_path, manifest)
    manifest = release(tmp_path)
    mutate(tmp_path, "rb_remaining.json", lambda r: r["distribution"].update(samples=[0, 1]))
    with pytest.raises(ValueError, match="preserve declared point means"):
        validate_opening_drive_release(tmp_path, manifest)
