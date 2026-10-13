"""Compare frozen v2 football components with NFL outcomes, without fitting.

Run: uv run python research/opening_drive_nfl_wins.py
Writes a reproducible local research report; does not change models or serving.
"""

import csv
import json
from collections import Counter, defaultdict
from hashlib import sha256
from html import escape
from pathlib import Path

import numpy as np

from engine.chairstats.store import CampaignStore
from engine.chairstats.views import rankings
from engine.tables import observations
from engine.tables.application import catalog_state, connect, query_scope

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/outputs/opening_drive_nfl_wins_2026"
SEASON, THROUGH = 2026, 4
EXAMPLES = ("BUF", "KC")
STATS = (
    "passing_yards",
    "rushing_yards",
    "passing_tds",
    "rushing_tds",
    "passing_interceptions",
    "attempts",
    "carries",
)
SIGNALS = (*STATS, "total_yards", "offensive_tds", "top_qb_offensive_tds")
LABELS = {
    "passing_yards": "Passing yards",
    "rushing_yards": "Rushing yards",
    "passing_tds": "Passing TDs",
    "rushing_tds": "Rushing TDs",
    "passing_interceptions": "Interceptions (lower favored)",
    "attempts": "Pass attempts",
    "carries": "Carries",
    "total_yards": "Passing + rushing yards",
    "offensive_tds": "Passing + rushing TDs",
    "top_qb_offensive_tds": "TDs, highest-projected QB only",
}


def correlation(x, y):
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def wilson(k, n):
    if not n:
        return None
    z, p = 1.96, k / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [float(center - half), float(center + half)]


def metrics(games, signal):
    # Each game appears once; do not duplicate home and away as independent games.
    direction = -1 if signal == "passing_interceptions" else 1
    edges = [direction * g[f"edge_{signal}"] for g in games]
    margins = [g["margin"] for g in games]
    decisive = [(e, m) for e, m in zip(edges, margins, strict=True) if m != 0]
    picked = [(e, m) for e, m in decisive if abs(e) > 1e-12]
    correct = sum((e > 0) == (m > 0) for e, m in picked)
    return {
        "signal": signal,
        "games": len(games),
        "picks": len(picked),
        "correct": correct,
        "accuracy": correct / len(picked) if picked else None,
        "accuracy_wilson_95_descriptive": wilson(correct, len(picked)),
        "r_win": correlation([e for e, _ in decisive], [int(m > 0) for _, m in decisive]),
        "r_margin": correlation(edges, margins),
        "prediction_ties": len(decisive) - len(picked),
        "actual_ties": len(games) - len(decisive),
    }


def csv_file(name, rows):
    with (OUT / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    store = CampaignStore(ROOT / "data/campaigns")
    vintages = {
        str(origin): rankings(
            store, "opening_drive_v2", horizon=horizon, season=SEASON, origin_week=origin
        )
        for horizon, origin in (("season", 0), ("remaining", 2))
    }
    with query_scope(ROOT / "data"), connect() as db:
        schedule = observations.schedule(db, SEASON)
        games = (
            db.execute(
                "SELECT game_id, season, week, home_team, away_team, home_score, away_score "
                "FROM analytics.current_observed_schedule "
                "WHERE season=? AND game_type='REG' AND week BETWEEN 1 AND ? "
                "ORDER BY week, game_id",
                [SEASON, THROUGH],
            )
            .pl()
            .to_dicts()
        )
        actuals = (
            db.execute(
                "SELECT player_id, team, week, "
                + ", ".join(STATS)
                + " FROM analytics.current_weeks WHERE season=? AND season_type='REG' "
                "AND week BETWEEN 1 AND ?",
                [SEASON, THROUGH],
            )
            .pl()
            .to_dicts()
        )
        catalog = catalog_state()["catalog"]
    inputs = dict(
        vintages=vintages, schedule=schedule, games=games, actuals=actuals, catalog=catalog
    )
    frozen = json.dumps(inputs, indent=2, default=str, allow_nan=False)
    (OUT / "inputs.json").write_text(frozen)
    assert len({g["game_id"] for g in games}) == len(games)
    assert len({(a["player_id"], a["week"]) for a in actuals}) == len(actuals)
    assert all(g["home_score"] is not None and g["away_score"] is not None for g in games)

    def normalize(t):
        return {"LAR": "LA", "STL": "LA", "SD": "LAC", "OAK": "LV"}.get(t, t)

    forecasts, players = {}, []
    for origin, vintage in vintages.items():
        grouped = defaultdict(list)
        rows = vintage["rankings"]
        assert len({r["player_id"] for r in rows}) == len(rows)
        for row in rows:
            team = normalize(row["team"])
            expected_games = sum(
                g["game_type"] == "REG"
                and g["week"] > int(origin)
                and team in (normalize(g["home_team"]), normalize(g["away_team"]))
                for g in schedule
            )
            assert row["origin_week"] == int(origin)
            assert row["scheduled_games"] == expected_games > 0
            rates = {s: row["predictions"].get(s, 0) / expected_games for s in STATS}
            assert all(np.isfinite(v) for v in rates.values())
            record = dict(
                origin_week=int(origin),
                team=team,
                player_id=row["player_id"],
                name=row["player_display_name"],
                position=row["position"],
                **rates,
            )
            players.append(record)
            grouped[team].append(record)
        for team, group in grouped.items():
            total = {s: sum(r[s] for r in group) for s in STATS}
            qbs = [r for r in group if r["position"] == "QB"]
            top = max(qbs, key=lambda r: r["passing_yards"])
            total.update(
                total_yards=total["passing_yards"] + total["rushing_yards"],
                offensive_tds=total["passing_tds"] + total["rushing_tds"],
                top_qb_offensive_tds=top["passing_tds"]
                + top["rushing_tds"]
                + sum(r["rushing_tds"] for r in group if r["position"] != "QB"),
                forecast_players=len(group),
                forecast_qbs=len(qbs),
                top_qb=top["name"],
            )
            forecasts[int(origin), team] = total
    observed = defaultdict(lambda: dict.fromkeys(STATS, 0))
    assignments = {
        int(o): {r["player_id"]: normalize(r["team"]) for r in v["rankings"]}
        for o, v in vintages.items()
    }
    coverage = Counter()
    for row in actuals:
        team, week = normalize(row["team"]), row["week"]
        origin = 0 if week <= 2 else 2
        for stat in STATS:
            assert row[stat] is not None
            observed[week, team][stat] += row[stat]
        if any(row[s] != 0 for s in STATS):
            assigned = assignments[origin].get(row["player_id"])
            coverage["productive_player_weeks"] += 1
            coverage[
                "matched_team"
                if assigned == team
                else "unmodeled"
                if assigned is None
                else "different_saved_team"
            ] += 1
    team_weeks, matchups = [], []
    for game in games:
        week = game["week"]
        origin = 0 if week <= 2 else 2
        assert origin < week
        home, away = normalize(game["home_team"]), normalize(game["away_team"])
        h, a = forecasts[origin, home], forecasts[origin, away]
        match = dict(**game, origin_week=origin, margin=game["home_score"] - game["away_score"])
        match.update({f"edge_{s}": h[s] - a[s] for s in SIGNALS})
        matchups.append(match)
        for side, team, opponent, pred, opp in (
            ("home", home, away, h, a),
            ("away", away, home, a, h),
        ):
            assert (week, team) in observed and team in assignments[origin].values()
            obs = observed[week, team]
            points = game[f"{side}_score"]
            against = game["away_score" if side == "home" else "home_score"]
            edge = pred["offensive_tds"] - opp["offensive_tds"]
            team_weeks.append(
                dict(
                    week=week,
                    team=team,
                    opponent=opponent,
                    home=side == "home",
                    origin_week=origin,
                    points=points,
                    points_against=against,
                    result="W" if points > against else "L" if points < against else "T",
                    td_pick="W" if edge > 0 else "L" if edge < 0 else "T",
                    td_edge=edge,
                    opponent_pred_tds=opp["offensive_tds"],
                    **{f"pred_{s}": pred[s] for s in SIGNALS},
                    **{f"actual_{s}": obs[s] for s in STATS},
                    actual_offensive_tds=obs["passing_tds"] + obs["rushing_tds"],
                    forecast_players=pred["forecast_players"],
                    forecast_qbs=pred["forecast_qbs"],
                )
            )
    overall = [metrics(matchups, s) for s in SIGNALS]
    weekly = [
        dict(week=w, **metrics([g for g in matchups if g["week"] == w], "offensive_tds"))
        for w in range(1, THROUGH + 1)
    ]
    examples = [r for r in team_weeks if r["team"] in EXAMPLES]
    production = [
        {
            "stat": s,
            "r_prediction_actual": correlation(
                [r[f"pred_{s}"] for r in team_weeks], [r[f"actual_{s}"] for r in team_weeks]
            ),
        }
        for s in (*STATS, "offensive_tds")
    ]
    notes = [
        "Retrospective reconstructed 2026 forecasts, not predictions issued before games.",
        "Weeks 1–2 use origin 0; Weeks 3–4 use origin 2. No fitting on these outcomes.",
        "All saved QB/RB/WR/TE candidates are assigned to their saved forecast team; "
        "no filtering by eventual participation, actual starters, or results.",
        "Player horizon totals are divided by scheduled games (17 or 15), then summed. "
        "These are season-average rates, not weekly matchup or availability forecasts.",
        "Offensive TD signal = passing TDs + rushing TDs. Receiving TDs are not added "
        "because that would count the same passing score twice. Yardage is gross passing "
        "plus rushing, without sack deductions. Fantasy points are never used.",
        "Winner rule: favor the team with the higher forecast component; for interceptions, "
        "favor fewer. No home-field, defense, kicking, or opponent adjustment is fitted.",
        "Correlation is Pearson r of the home-minus-away predicted component against "
        "home win (1/0), or actual home-minus-away scoring margin. Each game enters once.",
        "The top-QB sensitivity keeps only each team's QB with highest predicted passing "
        "yards, including his rushing TDs; other QB TDs are omitted. It is not a starter forecast.",
        "Independent player models do not constrain total team opportunities; backups, "
        "stale team assignments and missing candidates can distort team totals.",
        "Four weeks and repeated teams are a small, dependent sample. Wilson intervals "
        "are descriptive binomial approximations, not dependence-adjusted evidence. "
        "Multiple component comparisons are exploratory, not a model selection holdout.",
        "Actual component totals use all recorded player rows, including unmodeled players. "
        "Coverage diagnostics never change forecast eligibility or team assignment.",
    ]
    result = dict(
        season=SEASON,
        through_week=THROUGH,
        manifest_sha256=vintages["0"]["manifest_sha256"],
        inputs_sha256=sha256(frozen.encode()).hexdigest(),
        games=len(matchups),
        metrics=overall,
        weekly=weekly,
        production=production,
        always_home_correct=sum(g["margin"] > 0 for g in matchups),
        non_tied_games=sum(g["margin"] != 0 for g in matchups),
        coverage=dict(coverage),
        examples=examples,
        notes=notes,
    )
    (OUT / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    for name, rows in (
        ("team_weeks", team_weeks),
        ("matchups", matchups),
        ("player_rates", players),
        ("examples", examples),
        ("metrics", overall),
        ("weekly", weekly),
    ):
        csv_file(f"{name}.csv", rows)
    render(result, matchups)
    print(json.dumps({k: result[k] for k in ("games", "metrics", "weekly", "coverage")}, indent=2))


def render(result, games):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for week in range(1, THROUGH + 1):
        group = [g for g in games if g["week"] == week]
        ax.scatter(
            [g["edge_offensive_tds"] for g in group],
            [g["margin"] for g in group],
            label=f"Week {week}",
            alpha=0.8,
        )
    ax.axhline(0, color="#999999", linewidth=0.8)
    ax.axvline(0, color="#999999", linewidth=0.8)
    ax.set(
        xlabel="Predicted offensive TD advantage (home minus away)",
        ylabel="Actual NFL scoring margin (home minus away)",
        title="Opening Drive v2: does offensive TD advantage predict winning?",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "td_advantage.png", dpi=160)
    plt.close(fig)

    def table(rows, columns):
        def cell(v):
            return f"{v:.2f}" if isinstance(v, float) else escape(str(v))

        return (
            "<table><tr>"
            + "".join(f"<th>{escape(label)}</th>" for _, label in columns)
            + "</tr>"
            + "".join(
                "<tr>" + "".join(f"<td>{cell(r[k])}</td>" for k, _ in columns) + "</tr>"
                for r in rows
            )
            + "</table>"
        )

    overview = [
        dict(
            label=LABELS[m["signal"]],
            picks=f"{m['correct']}/{m['picks']}",
            accuracy=f"{m['accuracy']:.1%}",
            r_win=m["r_win"],
            r_margin=m["r_margin"],
        )
        for m in result["metrics"]
    ]
    td = next(m for m in result["metrics"] if m["signal"] == "offensive_tds")
    top = next(m for m in result["metrics"] if m["signal"] == "top_qb_offensive_tds")
    low, high = td["accuracy_wilson_95_descriptive"]
    html = (
        "<!doctype html><html lang='en'><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>Opening Drive v2 and NFL wins</title><style>"
        "body{font:16px system-ui;max-width:1150px;margin:40px auto;"
        "padding:0 24px;color:#172033}"
        "p,li{line-height:1.6}table{border-collapse:collapse;width:100%;margin:20px 0}"
        "td,th{padding:10px;border-bottom:1px solid #ddd;text-align:right}"
        "th{background:#edf2ff}td:first-child,th:first-child{text-align:left}"
        "section{overflow:auto}img{max-width:100%}.note{background:#fff3d6;padding:16px}</style>"
        "<h1>Opening Drive v2 and NFL wins</h1>"
        "<p>2026 Weeks 1–4 · 64 games · NFL outcomes · no fantasy scoring</p>"
        "<p class='note'>Exploratory reconstruction, not a validated NFL win model. "
        "A higher projected offensive TD rate is a simple strength signal, "
        "not an NFL score prediction.</p>"
        f"<p><strong>The offensive TD rule gets {td['correct']}/{td['picks']} winners "
        f"right ({td['accuracy']:.1%}), with win correlation r={td['r_win']:.2f}.</strong> "
        f"Using only the highest-projected QB per team drops accuracy to {top['accuracy']:.1%}. "
        f"The descriptive 95% interval for the original rule is {low:.0%}–{high:.0%}; "
        "it does not adjust for repeated teams or exploring multiple signals. "
        "This is a modest, method-sensitive association that needs historical and prospective "
        "validation. Saved player-team assignments and unconstrained backup production "
        "need particular attention before using this as a weekly win model.</p>"
        + table(
            overview,
            [
                ("label", "Forecast advantage"),
                ("picks", "Correct picks"),
                ("accuracy", "Accuracy"),
                ("r_win", "r with win"),
                ("r_margin", "r with margin"),
            ],
        )
        + f"<p>Always choosing home: {result['always_home_correct']}/"
        f"{result['non_tied_games']}. "
        "Correlations use one home-versus-away comparison per game.</p>"
        "<img src='td_advantage.png' "
        "alt='Predicted touchdown advantage versus actual score margin'>"
        "<h2>Offensive TD picks by week</h2>"
        + table(
            result["weekly"],
            [("week", "Week"), ("correct", "Correct"), ("picks", "Games"), ("r_win", "r with win")],
        )
    )
    for team in EXAMPLES:
        rows = [r for r in result["examples"] if r["team"] == team]
        html += (
            f"<h2>{team}</h2><p>Values are season-average production per scheduled game. "
            "Weeks 1–2 share one forecast; Weeks 3–4 share the updated forecast. "
            "Actual scores are from this team's perspective.</p><section>"
            + table(
                [
                    dict(
                        r,
                        score=f"{r['result']} {r['points']}–{r['points_against']}",
                        pass_pair=f"{r['pred_passing_yards']:.0f} / {r['actual_passing_yards']}",
                        rush_pair=f"{r['pred_rushing_yards']:.0f} / {r['actual_rushing_yards']}",
                        td_pair=f"{r['pred_offensive_tds']:.2f} / {r['actual_offensive_tds']}",
                    )
                    for r in rows
                ],
                [
                    ("week", "Week"),
                    ("opponent", "Opponent"),
                    ("pass_pair", "Pass yards pred / actual"),
                    ("rush_pair", "Rush yards pred / actual"),
                    ("td_pair", "TDs pred / actual"),
                    ("opponent_pred_tds", "Opponent pred TDs"),
                    ("td_pick", "TD pick"),
                    ("score", "Result"),
                ],
            )
            + "</section>"
        )
    html += (
        "<h2>Do predicted components match weekly production?</h2>"
        "<p>Correlations below use 128 team-weeks, with repeated teams; "
        "they describe production accuracy rather than winner selection.</p>"
        + table(
            result["production"],
            [("stat", "Component"), ("r_prediction_actual", "r predicted vs actual")],
        )
        + "<h2>Method and limitations</h2><ul>"
        + "".join(f"<li>{escape(n)}</li>" for n in result["notes"])
        + "</ul>"
        + f"<p>Productive player-week coverage: {escape(str(result['coverage']))}.</p>"
        "<p>Downloads: <a href='team_weeks.csv'>All team-weeks</a> · "
        "<a href='examples.csv'>Two-team examples</a> · <a href='matchups.csv'>Games</a> · "
        "<a href='player_rates.csv'>Player contributions</a> · "
        "<a href='results.json'>Metrics and provenance</a> · "
        "<a href='inputs.json'>Frozen inputs</a></p></html>"
    )
    (OUT / "index.html").write_text(html)


if __name__ == "__main__":
    main()
