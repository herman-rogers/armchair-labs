"""Explain team forecast bias using the frozen inputs of the team accuracy report.

Run after opening_drive_team_accuracy.py. This performs no fitting or calibration.
"""

import csv
import json
from collections import defaultdict
from hashlib import sha256
from html import escape
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/outputs/opening_drive_team_accuracy_2026"
SKILL = {"QB", "RB", "WR", "TE"}


def scoring_parts(position, stats):
    parts = {"Rushing yards": 0.1 * stats["rushing_yards"], "Rushing TDs": 6 * stats["rushing_tds"]}
    if position == "QB":
        parts.update(
            {
                "Passing yards": 0.04 * stats["passing_yards"],
                "Passing TDs": 4 * stats["passing_tds"],
                "Interceptions": -2 * stats["passing_interceptions"],
            }
        )
    else:
        parts.update(
            {
                "Receptions": stats["receptions"],
                "Receiving yards": 0.1 * stats["receiving_yards"],
                "Receiving TDs": 6 * stats["receiving_tds"],
            }
        )
    return parts


def aggregate(rows):
    values = np.array(rows)
    return dict(
        n=len(rows),
        predicted=float(values[:, 0].mean()),
        actual=float(values[:, 1].mean()),
        underpredicted=int((values[:, 0] < values[:, 1]).sum()),
    )


def main():
    inputs_path, result_path = OUT / "inputs.json", OUT / "results.json"
    inputs, result = (json.loads(p.read_text()) for p in (inputs_path, result_path))
    rows = list(csv.DictReader((OUT / "player_weeks.csv").open()))
    completed = {(r["week"], r["team_id"]) for r in result["team_weeks"] if r["completed"]}
    n_teams = len(completed)
    ranks = {
        int(w): {r["player_id"]: r for r in vintage["rankings"]}
        for w, vintage in inputs["rankings"].items()
    }
    actuals = {(r["player_id"], r["week"]): r for r in inputs["actuals"]}
    position, player, components = defaultdict(list), defaultdict(list), defaultdict(list)
    for r in rows:
        week, team = int(r["week"]), int(r["team_id"])
        if (week, team) not in completed:
            continue
        prediction, actual = float(r["predicted"]), float(r["actual_espn"])
        position[r["position"]].append((prediction, actual))
        player[r["name"]].append((prediction, actual))
        if r["position"] not in SKILL:
            continue
        forecast = ranks[week][r["player_id"]]
        assert forecast["origin_week"] < week
        assert np.isclose(
            forecast["prediction"] / forecast["scheduled_games"], forecast["points_per_game"]
        )
        # Independently confirm the denominator against the pinned schedule.
        code = {"LAR": "LA", "STL": "LA", "SD": "LAC", "OAK": "LV"}
        team_code = code.get(forecast["team"], forecast["team"])
        scheduled = [
            g
            for g in inputs["schedule"]
            if g["season"] == forecast["season"]
            and g["game_type"] == "REG"
            and g["week"] > forecast["origin_week"]
            and team_code
            in (code.get(g["home_team"], g["home_team"]), code.get(g["away_team"], g["away_team"]))
        ]
        assert len(scheduled) == forecast["scheduled_games"]
        observed = actuals.get((r["player_id"], week))
        pred_parts = scoring_parts(r["position"], forecast["predictions"])
        actual_parts = scoring_parts(r["position"], observed) if observed else {}
        assert np.isclose(sum(actual_parts.values()), float(r["actual_model_scoring"]))
        for key, value in pred_parts.items():
            components[key].append((value / forecast["scheduled_games"], actual_parts.get(key, 0)))
    by_position = [
        dict(
            position=k,
            **aggregate(v),
            predicted_per_team=sum(x[0] for x in v) / n_teams,
            actual_per_team=sum(x[1] for x in v) / n_teams,
            gap_per_team=sum(x[1] - x[0] for x in v) / n_teams,
        )
        for k, v in position.items()
    ]
    by_component = [
        dict(
            component=k,
            predicted_per_team=sum(x[0] for x in v) / n_teams,
            actual_per_team=sum(x[1] for x in v) / n_teams,
            gap_per_team=sum(x[1] - x[0] for x in v) / n_teams,
        )
        for k, v in components.items()
    ]
    cohorts, stat_cohorts, unmatched = defaultdict(list), defaultdict(list), []
    for game in inputs["snapshot"]["week_lineups"]:
        week = game["week"]
        for side in ("home", "away"):
            if (week, game[f"{side}_team_id"]) not in completed:
                continue
            for p in game[f"{side}_lineup"]:
                if p["position"] not in SKILL:
                    continue
                pid = inputs["identities"].get(str(p["espn_id"]))
                forecast = ranks[week].get(pid)
                cohort = "Bench / IR" if p["slot"] in {"BE", "IR", "ER", ""} else "Starter"
                if forecast is None:
                    unmatched.append(dict(week=week, name=p["player_display_name"], cohort=cohort))
                    continue
                actual = actuals.get((pid, week))
                if actual is None:
                    assert p["points"] == 0, "Cannot infer zero from an absent nonzero actual"
                scored = sum(scoring_parts(p["position"], actual).values()) if actual else 0
                prediction = 0 if p["on_bye"] else forecast["points_per_game"]
                cohorts[cohort].append((prediction, scored))
                stat_cohorts[(cohort, actual is not None)].append((prediction, scored))
    cohort_rows = [dict(cohort=k, **aggregate(v)) for k, v in cohorts.items()]
    stat_rows = [
        dict(cohort=k[0], has_stat_row=k[1], **aggregate(v)) for k, v in stat_cohorts.items()
    ]
    leaders = sorted(
        [
            dict(name=k, **aggregate(v), total_gap=sum(x[1] - x[0] for x in v))
            for k, v in player.items()
        ],
        key=lambda r: -r["total_gap"],
    )[:15]
    component_pred = sum(r["predicted_per_team"] for r in by_component)
    model = result["metrics"]["model_scoring"]
    full = result["metrics"]["hybrid_full_team"]
    assert np.isclose(sum(r["gap_per_team"] for r in by_position), -full["bias"])
    assert np.isclose(sum(r["actual_per_team"] for r in by_component), model["actual_mean"])
    diagnostics = dict(
        inputs_sha256=sha256(inputs_path.read_bytes()).hexdigest(),
        results_sha256=sha256(result_path.read_bytes()).hexdigest(),
        team_weeks=n_teams,
        underpredicted_teams=sum(
            r["predicted_full"] < r["actual_full"] for r in result["team_weeks"] if r["completed"]
        ),
        position=by_position,
        components=by_component,
        cohorts=cohort_rows,
        cohorts_by_stat_presence=stat_rows,
        unmatched=unmatched,
        biggest_gaps=leaders,
        component_predicted_mean=component_pred,
        direct_predicted_mean=model["predicted_mean"],
        scoring_difference=result["metrics"]["espn_offense"]["actual_mean"] - model["actual_mean"],
    )
    (OUT / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2, allow_nan=False))
    for name, records in [
        ("position", by_position),
        ("components", by_component),
        ("cohorts", cohort_rows),
        ("biggest_gaps", leaders),
    ]:
        with (OUT / f"diagnostics_{name}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    render(diagnostics)
    print(json.dumps(diagnostics, indent=2, allow_nan=False))


def render(data):
    def table(rows, fields):
        def cell(value):
            return f"{value:.2f}" if isinstance(value, float) else escape(str(value))

        return (
            "<table><tr>"
            + "".join(f"<th>{escape(label)}</th>" for _, label in fields)
            + "</tr>"
            + "".join(
                "<tr>" + "".join(f"<td>{cell(r[key])}</td>" for key, _ in fields) + "</tr>"
                for r in rows
            )
            + "</table>"
        )

    html = (
        "<!doctype html><html lang='en'><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>Why Opening Drive runs low</title><style>"
        "body{font:17px system-ui;max-width:1050px;margin:40px auto;padding:0 24px;color:#172033}"
        "p,li{line-height:1.65}table{border-collapse:collapse;width:100%;margin:24px 0}"
        "td,th{padding:10px;text-align:right;border-bottom:1px solid #ddd}"
        "td:first-child,th:first-child{text-align:left}th{background:#eef2ff}"
        "a{color:#2563eb}.note{background:#fff3d6;padding:16px}</style>"
        "<p><a href='index.html'>← Full-team charts</a></p>"
        "<h1>Why Opening Drive runs low</h1>"
        "<p>2026 Weeks 1–4 · frozen inputs from the team comparison · no model changes</p>"
        "<p><strong>The strongest evidence points to applying a season-average forecast to "
        "a weekly selected lineup.</strong> Managers usually start players who are available "
        "and have productive roles. The model predicts total production across the remaining "
        "season, including missed games and lost roles, then divides by all scheduled games. "
        "That rate is not a forecast conditional on this week's availability and workload.</p>"
        "<p class='note'>This is a supported diagnosis, not a causal attribution of the entire "
        "gap. Four early weeks cannot separate selection, availability, role changes, model "
        "shrinkage and unusually strong early performances. These forecasts were reconstructed "
        "after the games.</p>"
        "<h2>Where the 29.1-point team gap comes from</h2>"
        + table(
            data["position"],
            [
                ("position", "Position"),
                ("predicted_per_team", "Predicted / team-week"),
                ("actual_per_team", "ESPN actual / team-week"),
                ("gap_per_team", "Actual minus predicted"),
            ],
        )
        + "<p>Kickers and defenses together miss by just 0.19 points per team-week on average. "
        "RB and WR account for about 79% of the full-team shortfall.</p>"
        "<h2>The starter versus bench comparison</h2>"
        + table(
            data["cohorts"],
            [
                ("cohort", "Cohort"),
                ("n", "Player-weeks"),
                ("predicted", "Predicted / player-week"),
                ("actual", "Actual / player-week"),
            ],
        )
        + "<p>This table uses the same model scoring on both sides. Covered bench/IR players "
        "are nearly unbiased overall, while starters are 4.11 points low per player-week. "
        "57 of 277 bench/IR player-weeks have no NFL stat row and an independently confirmed "
        "ESPN zero; only 3 of 319 starter player-weeks do. Twelve unmatched bench/IR rows are "
        "excluded, not assigned zero predictions. No starting player is excluded.</p>"
        "<p>Availability does not explain everything: among players with recorded stat rows, "
        "bench predictions are 7.91 versus 9.91 actual, while starters are 11.02 versus 15.25. "
        "Current workload, selection of better players and conservative forecasts remain "
        "possible contributors. A stat row alone does not prove full-game participation.</p>"
        "<h2>It is not just missing touchdowns</h2>"
        + table(
            data["components"],
            [
                ("component", "Scoring component"),
                ("predicted_per_team", "Predicted / team-week"),
                ("actual_per_team", "Actual / team-week"),
                ("gap_per_team", "Gap"),
            ],
        )
        + "<p>About 71% of the component shortfall comes from receptions and yardage; "
        "about 29% comes from touchdowns. Component forecasts imply 87.61 offensive points, "
        "almost identical to the direct point forecast of 87.71. Switching to component-derived "
        "points would not fix the gap.</p>"
        "<h2>Checks that rule out simpler explanations</h2><ul>"
        "<li>Every starter's denominator agrees with the pinned NFL schedule: 17 games "
        "preseason and 15 after Week 2. We are not dividing by 18 calendar weeks.</li>"
        "<li>Matching the model scoring makes the gap larger: 32.74 points, versus 28.89 "
        "against ESPN offense. League bonuses and scoring differences are not the explanation.</li>"
        "<li>The deficit appears in all four weeks, so it is not a single bad week.</li>"
        "<li>The model uses squared-error regression and MSE selection, not a low-quantile "
        "objective. It is not trained or validated as a floor.</li></ul>"
        "<h2>Why it looks like a floor</h2>"
        "<p>191 of 319 individual offensive starter forecasts fall below actual production "
        "(60%). After adding the players together, 34 of 40 full-team predictions are low "
        "(85%). Aggregation makes a systematic downward bias look like a reliable lower bound. "
        "That does not establish calibrated downside protection for future weeks.</p>"
        "<h2>Largest player gaps</h2>"
        + table(
            data["biggest_gaps"],
            [
                ("name", "Player"),
                ("n", "Starts"),
                ("predicted", "Predicted / start"),
                ("actual", "ESPN actual / start"),
            ],
        )
        + "<h2>The next experiment</h2>"
        "<p>Keep season-total forecasts for season valuation. For weekly team projections, "
        "test a separate expected-active/workload layer using dated availability, recent snaps, "
        "routes, carries, targets and depth-chart changes, and use exact league scoring. "
        "Evaluate bias on players selected using only pregame evidence, with chronological "
        "historical validation and new prospective captures. Report position and workload "
        "cohorts rather than only all-player errors. Do not fit a blanket +29 adjustment "
        "to these same four weeks and call it validated.</p>"
        "<p><a href='diagnostics.json'>Auditable metrics and input hashes</a> · "
        "<a href='diagnostics_position.csv'>Position CSV</a> · "
        "<a href='diagnostics_components.csv'>Component CSV</a></p></html>"
    )
    (OUT / "diagnosis.html").write_text(html)


if __name__ == "__main__":
    main()
