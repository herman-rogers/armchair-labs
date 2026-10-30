"""Release invariants for Opening Drive's weekly, remaining and season projections."""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path


def validate_opening_drive_release(root: Path, manifest: dict) -> None:
    """Report horizon-coherence problems as warnings; never block a publication.

    Only manifests that declare a ``horizon_contract`` are checked, so bundles from
    any family or vintage stay publishable and readable.
    """
    try:
        _validate(root, manifest)
    except (ValueError, KeyError, TypeError) as error:
        warnings.warn(f"{manifest['campaign_id']}: {error}", stacklevel=2)


def _validate(root: Path, manifest: dict) -> None:
    contract = manifest.get("horizon_contract")
    if not contract:
        return
    components_required = bool(contract.get("components_required"))
    models = {(m.get("position"), m["horizon"]): m for m in manifest["models"]}
    expected = {(p, h) for p in ("QB", "RB", "WR", "TE") for h in ("weekly", "remaining", "season")}
    if not expected <= set(models):
        raise ValueError(
            "Opening Drive release must cover all four positions and all three horizons"
        )
    if len(models) != len(manifest["models"]):
        raise ValueError("Duplicate position/horizon models in Opening Drive release")
    payloads, origins = {}, set()
    for key in expected:
        model = models[key]
        if not model["forecasts"]:
            raise ValueError("Every Opening Drive horizon needs a current forecast")
        ref = max(model["forecasts"], key=lambda r: (r["season"], r["origin_week"]))
        origins.add((ref["season"], ref["origin_week"]))
        payload = json.loads((root / ref["path"]).read_text())
        if key[1] != "weekly":
            if payload.get("forecast_kind") != "coherent_schedule_projection":
                raise ValueError(
                    "Opening Drive cannot publish silently inherited current horizon forecasts"
                )
            if not {"projection_policy", "projection_evaluation"} <= set(
                model.get("artifacts", {})
            ):
                raise ValueError("Long-horizon projections require policy and evaluation artifacts")
        payloads[key] = payload
    if len(origins) != 1:
        raise ValueError(
            "Opening Drive horizons must share one current season and observation cutoff"
        )
    _, origin = next(iter(origins))

    def equal(a, b):
        return (
            isinstance(a, (int, float))
            and isinstance(b, (int, float))
            and not isinstance(a, bool)
            and not isinstance(b, bool)
            and math.isfinite(a)
            and math.isfinite(b)
            and math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-8)
        )

    for position in ("QB", "RB", "WR", "TE"):
        weekly_payload = payloads[position, "weekly"]
        weekly = {r["player_id"]: r for r in weekly_payload["rows"]}
        remaining = {r["player_id"]: r for r in payloads[position, "remaining"]["rows"]}
        season = {r["player_id"]: r for r in payloads[position, "season"]["rows"]}
        if not weekly.keys() == remaining.keys() == season.keys():
            raise ValueError("Opening Drive horizons must share the weekly eligible population")
        for horizon in ("remaining", "season"):
            if payloads[position, horizon].get("capture_hashes") != weekly_payload.get(
                "capture_hashes"
            ):
                raise ValueError("Horizon forecasts must bind the same weekly captures")
        for pid, w in weekly.items():
            r, s = remaining[pid], season[pid]
            projection = r.get("projection", {})
            if not projection or projection != s.get("projection"):
                raise ValueError("Season and remaining must use the identical future-week curve")
            if components_required:
                components = set().union(*(x["predictions"] for x in (w, r, s))) - {"points"}
                for row in (w, r, s):
                    if not components <= set(row["predictions"]):
                        raise ValueError("Component forecasts must be present in every horizon")
                    distribution = row.get("distribution", {})
                    samples = distribution.get("samples")
                    if (
                        not isinstance(samples, list)
                        or not samples
                        or any(
                            not isinstance(value, (int, float))
                            or isinstance(value, bool)
                            or not math.isfinite(value)
                            for value in samples
                        )
                    ):
                        raise ValueError("Distributions require finite numeric samples")
                    if not equal(
                        math.fsum(samples) / len(samples),
                        row["predictions"]["points"],
                    ):
                        raise ValueError("Distributions must preserve declared point means")
                for component in components:
                    current_components = [
                        game.get("components", {}).get(component)
                        for game in projection["future_weeks"]
                        if game["week"] == origin + 1
                    ]
                    if w["scheduled_games"] == 1:
                        if len(current_components) != 1 or not equal(
                            current_components[0], w["predictions"][component]
                        ):
                            raise ValueError(
                                "Current curve components must equal weekly component forecasts"
                            )
                    elif w["scheduled_games"] == 0 and not equal(w["predictions"][component], 0):
                        raise ValueError("Weekly bye component forecasts must be zero")
                    component_total = sum(
                        g.get("components", {}).get(component, float("nan"))
                        for g in projection["future_weeks"]
                    )
                    actual = projection.get("actual_components_to_date", {}).get(component)
                    if (
                        not equal(component_total, r["predictions"][component])
                        or actual is None
                        or not equal(component_total + actual, s["predictions"][component])
                    ):
                        raise ValueError("Component totals must reconcile across horizons")
            curve = projection["future_weeks"]
            weeks = [g["week"] for g in curve]
            if len(weeks) != len(set(weeks)) or any(x <= origin or x > 18 for x in weeks):
                raise ValueError("Projection curve contains duplicate or out-of-horizon weeks")
            current = [g["points"] for g in curve if g["week"] == origin + 1]
            if w["scheduled_games"] == 1:
                if len(current) != 1 or not equal(current[0], w["predictions"]["points"]):
                    raise ValueError(
                        "Current projection week must equal the saved weekly prediction"
                    )
            elif w["scheduled_games"] != 0 or current or not equal(w["predictions"]["points"], 0):
                raise ValueError("Weekly bye must contribute no game or points")
            total = sum(g["points"] for g in curve)
            if not equal(r["predictions"]["points"], total) or not equal(
                projection["remaining_points"], total
            ):
                raise ValueError("Remaining total must equal the sum of future weekly points")
            if not equal(s["predictions"]["points"], total + projection["actual_points_to_date"]):
                raise ValueError("Season total must equal known actuals plus remaining points")
            if r["scheduled_games"] != len(curve) or projection["remaining_scheduled_games"] != len(
                curve
            ):
                raise ValueError("Remaining denominator must count scheduled future games")
            if s["scheduled_games"] != projection["season_scheduled_games"] or s[
                "scheduled_games"
            ] < len(curve):
                raise ValueError("Season denominator must cover the declared season schedule")
            for row in (r, s):
                count = row["scheduled_games"]
                average = row.get("points_per_scheduled_game")
                if (count and not equal(average, row["predictions"]["points"] / count)) or (
                    not count and average is not None
                ):
                    raise ValueError(
                        "Per-game average must reconcile to its horizon total and games"
                    )
