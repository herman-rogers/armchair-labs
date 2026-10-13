"""Pure weekly lineup aggregation from frozen player and calibration JSON.

Gaussian-copula dependence is an approximation. K/DST scales transferred from
prior-score errors are provisional, and do not claim ESPN interval calibration.
"""

from itertools import combinations

import numpy as np
from scipy.special import ndtr

from engine.espn.observations import weekly_matchups
from engine.metrics.weekly_lineup import number, team_code
from engine.names import normalize

QUANTILES = (0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99)


def relationship(a, b):
    ta, tb = team_code(a.get("team")), team_code(b.get("team"))
    if not ta or not tb:
        return None
    if ta == tb:
        kind = "same_team"
    elif ta == team_code(b.get("opponent")) or tb == team_code(a.get("opponent")):
        kind = "opponents"
    else:
        return None
    return (kind, *sorted([a["position"], b["position"]]))


def correlation_matrix(players, estimates):
    matrix = np.eye(len(players))
    for i, j in combinations(range(len(players)), 2):
        matrix[i, j] = matrix[j, i] = estimates.get(relationship(players[i], players[j]), {}).get(
            "correlation", 0.0
        )
    values, vectors = np.linalg.eigh(matrix)
    matrix = (vectors * np.maximum(values, 1e-8)) @ vectors.T
    scale = np.sqrt(np.diag(matrix))
    return matrix / np.outer(scale, scale)


def joint_draws(players, estimates=None, n=4096, seed=42):
    if not players or any(
        not number(p.get("prediction")) or not number(p.get("std")) for p in players
    ):
        raise ValueError("Every starter needs a finite mean and uncertainty estimate")
    matrix = correlation_matrix(players, estimates or {})
    values, vectors = np.linalg.eigh(matrix)
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, len(players))) @ (vectors * np.sqrt(np.maximum(values, 0))).T
    draws = np.empty_like(z)
    for i, player in enumerate(players):
        samples = player.get("samples")
        qs = [player.get(f"q{int(q * 100):02d}") for q in QUANTILES]
        if samples and len(samples) >= 20 and all(number(v) for v in samples):
            values = np.sort(samples)
            draw = np.interp(ndtr(z[:, i]), (np.arange(len(values)) + 0.5) / len(values), values)
            draws[:, i] = draw + player["prediction"] - values.mean()
        elif all(number(q) for q in qs):
            # Exactly the same finite empirical inverse CDF used in model evaluation.
            qs = np.sort(qs)
            grid = (np.arange(256) + 0.5) / 256
            values = np.interp(grid, QUANTILES, qs)
            draw = np.interp(ndtr(z[:, i]), grid, values)
            draws[:, i] = draw + player["prediction"] - values.mean()
        else:
            draws[:, i] = player["prediction"] + z[:, i] * max(player["std"], 0)
    return draws.sum(axis=1)


def summarize(players, estimates, seed=42):
    draws = joint_draws(players, estimates, seed=seed)
    independent = joint_draws(players, seed=seed)
    return dict(
        prediction_interval_80=np.quantile(draws, [0.1, 0.9]).tolist(),
        independent_interval_80=np.quantile(independent, [0.1, 0.9]).tolist(),
        prediction_std=float(draws.std()),
    )


def forecast_snapshot(snapshot, player_forecasts, identities, schedule, calibration):
    """No model fitting, data access, mutation, or publishing during a request."""
    rows = (
        player_forecasts.get("players", [])
        if isinstance(player_forecasts, dict)
        else player_forecasts
    )
    rows = list(rows.values()) if isinstance(rows, dict) else rows
    rows = [
        r
        for r in rows
        if r.get("season", snapshot.season) == snapshot.season
        and r.get("week", snapshot.week) == snapshot.week
    ]
    by_id = {r["player_id"]: r for r in rows}
    by_name = {}
    for row in rows:
        key = normalize(row.get("player_display_name", "")), row["position"]
        by_name.setdefault(key, []).append(row)
    if calibration.get("covariance_before_season", snapshot.season) > snapshot.season:
        raise ValueError("Correlation calibration contains future-season outcomes")
    if (
        calibration.get("capture_season", snapshot.season),
        calibration.get("special_errors_before_week", snapshot.week),
    ) > (snapshot.season, snapshot.week):
        raise ValueError("K/DST calibration contains outcomes after requested forecast cutoff")
    estimates = {
        (r["relationship"], *sorted([r["position_a"], r["position_b"]])): r
        for r in calibration.get("correlations", [])
    }
    earlier, special = {}, {}
    for game in snapshot.week_lineups:
        for p in game.home_lineup + game.away_lineup:
            if p.position in {"K", "DST", "D/ST"}:
                if game.week == snapshot.week:
                    special[p.espn_id] = p.projected_points
                elif game.week < snapshot.week and number(p.points):
                    earlier.setdefault(p.espn_id, {})[game.week] = p.points
    scheduled = {
        team_code(r[t])
        for r in schedule
        if r["season"] == snapshot.season and r["week"] == snapshot.week and r["game_type"] == "REG"
        for t in ("home_team", "away_team")
    }
    result = weekly_matchups(snapshot, snapshot.week)
    result["source"] = "campaign_weekly_distribution"
    dates = [
        str(r["gameday"])[:10]
        for r in schedule
        if r["season"] == snapshot.season
        and r["week"] == snapshot.week
        and r["game_type"] == "REG"
        and r.get("gameday")
    ]
    result["pregame_deadline"] = f"{min(dates)}T00:00:00+00:00" if dates else None
    for game in result["matchups"]:
        for side in (game["home"], game["away"]):
            forecasts = []
            for p in side["lineup"]:
                position = "DST" if p["position"] == "D/ST" else p["position"]
                row = dict(
                    p,
                    position=position,
                    prediction=None,
                    std=None,
                    basis=None,
                    unavailable_reason=None,
                )
                if position in {"K", "DST"}:
                    prior = [v for _, v in sorted(earlier.get(p["espn_id"], {}).items())[-4:]]
                    projected = special.get(p["espn_id"])
                    if number(projected):
                        row.update(
                            prediction=projected,
                            basis=f"ESPN Week {snapshot.week} projection · K/DST",
                        )
                    elif prior:
                        row.update(
                            prediction=sum(prior) / len(prior),
                            basis=f"Prior {len(prior)} recorded games · K/DST fallback",
                        )
                    scale = calibration.get("special_teams", {}).get(position, {})
                    if number(scale.get("std")) and scale.get("n", 0) >= 8:
                        row.update(
                            std=scale["std"],
                            uncertainty_status="provisional_prior_score_transfer",
                            uncertainty_n=scale["n"],
                        )
                else:
                    pred = by_id.get(identities.get(p["espn_id"]))
                    if pred is None:
                        matches = by_name.get((normalize(p["player_display_name"]), position), [])
                        pred = matches[0] if len(matches) == 1 else None
                    if pred is not None and pred["position"] == position:
                        row.update(
                            {
                                key: pred[key]
                                for key in (
                                    "prediction",
                                    "std",
                                    "samples",
                                    "team",
                                    "opponent",
                                    *[f"q{int(q * 100):02d}" for q in QUANTILES],
                                )
                                if key in pred
                            }
                        )
                        row["basis"] = "Opening Drive v3 weekly distribution"
                if p.get("on_bye") or (
                    row.get("team") and scheduled and team_code(row["team"]) not in scheduled
                ):
                    row.update(prediction=0.0, std=0.0, samples=[0.0] * 20, basis="Scheduled bye")
                if row["prediction"] is None:
                    row["unavailable_reason"] = "No usable weekly forecast matched this starter."
                if number(row["std"]) and number(row["prediction"]):
                    row.update(summarize([row], {}))
                forecasts.append(row)
            starters = [p for p in forecasts if p["started"]]
            covered = [p for p in starters if number(p["prediction"])]
            offense = [p for p in starters if p["position"] in {"QB", "RB", "WR", "TE"}]
            side.update(
                forecasts=[{k: v for k, v in p.items() if k != "samples"} for p in forecasts],
                covered_starters=len(covered),
                starter_count=len(starters),
                model_projection=sum(p["prediction"] for p in covered)
                if starters and len(covered) == len(starters)
                else None,
                uncertainty_status="unavailable",
                prediction_interval_80=None,
                prediction_std=None,
                offense_only_interval_80=None,
            )
            if offense and all(number(p["prediction"]) and number(p["std"]) for p in offense):
                side["offense_only_interval_80"] = summarize(offense, estimates)[
                    "prediction_interval_80"
                ]
            if starters and all(number(p["prediction"]) and number(p["std"]) for p in starters):
                side.update(
                    summarize(starters, estimates),
                    uncertainty_status="provisional_full_lineup"
                    if any(p["position"] in {"K", "DST"} for p in starters)
                    else "research_offense_distribution",
                )
            side.pop("lineup", None)
        h, a = game["home"]["model_projection"], game["away"]["model_projection"]
        game["margin"] = h - a if h is not None and a is not None else None
        game["favorite_team_id"] = (
            None
            if game["margin"] in (None, 0)
            else game["home" if game["margin"] > 0 else "away"]["team_id"]
        )
    result["range_reliability_note"] = (
        "These are nominal 80% outcome ranges, not demonstrated 80% coverage. "
        "In the v3 study, historical synthetic offense-lineup ranges covered "
        "about 77% of outcomes. Full-lineup K/DST uncertainty remains provisional."
    )
    result["limitations"] = [
        result["range_reliability_note"],
        (
            "Offense uses the Opening Drive scoring contract; K/DST uses ESPN "
            "league scoring. Official league offensive scoring can differ."
        ),
        (
            "Full-lineup intervals are provisional: K/DST scales transfer earlier "
            "prior-score errors to ESPN means; K/DST covariance is unestimated."
        ),
        (
            "Gaussian-copula joint draws use regularized earlier-season residual "
            "correlations. Marginal draws are centered to the declared means; "
            "fitted quantile locations can differ."
        ),
        (
            "Snapshot lineups and ESPN projections may be captured after kickoff. "
            "This endpoint does not establish pregame provenance."
        ),
    ]
    return result
