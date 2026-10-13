"""Export a retrospective, cutoff-respecting Sweaty Plays team scoring comparison.

Run: uv run python research/opening_drive_team_accuracy.py
Outputs are local research artifacts; no forecast log or model is modified.
"""

import csv
import json
from dataclasses import replace
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from engine.chairstats.store import CampaignStore
from engine.chairstats.views import rankings
from engine.espn.sync import LeagueSnapshot
from engine.metrics.weekly_lineup import number, weekly_forecast
from engine.tables import observations
from engine.tables.application import catalog_state, connect, query_scope

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/outputs/opening_drive_team_accuracy_2026"
OFFENSE = {"QB", "RB", "WR", "TE"}


def write_csv(name, rows):
    with (OUT / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def measures(rows, prediction, actual):
    pairs = [
        (r[prediction], r[actual])
        for r in rows
        if r[prediction] is not None and r[actual] is not None and r["completed"]
    ]
    if not pairs:
        return {"n": 0}
    pred, obs = np.array(pairs).T
    error = pred - obs
    return dict(
        n=len(pairs),
        predicted_mean=float(pred.mean()),
        actual_mean=float(obs.mean()),
        mae=float(np.abs(error).mean()),
        rmse=float(np.sqrt((error**2).mean())),
        bias=float(error.mean()),
        correlation=float(np.corrcoef(pred, obs)[0, 1]),
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    snapshot_bytes = (ROOT / "data/outputs/league_snapshot.json").read_bytes()
    snapshot = LeagueSnapshot.from_dict(json.loads(snapshot_bytes))
    store = CampaignStore(ROOT / "data/campaigns")
    manifest = store.campaign("opening_drive_v2")
    with query_scope(ROOT / "data"), connect() as db:
        identities = observations.espn_ids(db)
        schedule = observations.schedule(db, snapshot.season)
        actuals = (
            db.execute(
                "SELECT player_id, week, passing_yards, passing_tds, passing_interceptions, "
                "rushing_yards, rushing_tds, receptions, receiving_yards, receiving_tds "
                "FROM analytics.current_weeks WHERE season=? AND season_type='REG'",
                [snapshot.season],
            )
            .pl()
            .to_dicts()
        )
        catalog = catalog_state()["catalog"]
    actual_map = {(r["player_id"], r["week"]): r for r in actuals}
    assert len(actual_map) == len(actuals), "Duplicate weekly actuals"
    teams, players, games, vintages, zero_rows = [], [], [], {}, []
    for week in range(1, snapshot.week + 1):
        # Select only an origin strictly before the target week, shared by all positions.
        horizon = "season" if week <= 2 else "remaining"
        models = [m for m in manifest["models"] if m["horizon"] == horizon]
        origins = set.intersection(
            *[
                {
                    f["origin_week"]
                    for f in m["forecasts"]
                    if f["season"] == snapshot.season and f["origin_week"] < week
                }
                for m in models
            ]
        )
        origin = max(origins)
        ranks = rankings(
            store, "opening_drive_v2", horizon=horizon, season=snapshot.season, origin_week=origin
        )
        assert {r["origin_week"] for r in ranks["rankings"]} == {origin}
        vintages[str(week)] = ranks
        forecast = weekly_forecast(replace(snapshot, week=week), ranks, identities, schedule)
        for game in forecast["matchups"]:
            raw = next(
                g
                for g in snapshot.week_lineups
                if g.week == week
                and {g.home_team_id, g.away_team_id}
                == {game["home"]["team_id"], game["away"]["team_id"]}
            )
            pair = []
            for side_name in ("home", "away"):
                side = game[side_name]
                team = next(t for t in snapshot.teams if t.team_id == side["team_id"])
                entry = next(e for e in team.schedule if e.week == week)
                completed = entry.played and week < snapshot.week
                lineup = {p.espn_id: p for p in raw.lineup_for(team.team_id)}
                starters = [p for p in side["forecasts"] if p["started"]]
                # Keep the existing prior-score average when available. For Week 1
                # or a newly rostered K/DST, use ESPN's recorded weekly projection.
                for player in starters:
                    if player["position"] in {"K", "DST", "D/ST"} and player["prediction"] is None:
                        reference = lineup[player["espn_id"]].projected_points
                        if number(reference):
                            player.update(
                                prediction=reference,
                                basis="ESPN weekly projection · no earlier recorded K/DST scores",
                                unavailable_reason=None,
                            )
                offense = [p for p in starters if p["position"] in OFFENSE]
                special = [p for p in starters if p["position"] in {"K", "DST", "D/ST"}]
                special_total = (
                    sum(p["prediction"] for p in special)
                    if all(p["prediction"] is not None for p in special)
                    else None
                )
                assert offense, "Missing offensive lineup"
                covered = all(p["prediction"] is not None for p in offense)
                comparable = 0.0
                for p in starters:
                    observed = lineup[p["espn_id"]]
                    standard = None
                    pid = identities.get(p["espn_id"])
                    if completed and p["position"] in OFFENSE:
                        assert pid is not None, f"Unresolved actual identity: {p}"
                        stats = actual_map.get((pid, week))
                        if stats is None:
                            # The complete-week stat feed omits nonparticipants. Require
                            # independent ESPN zero points before imputing a zero here.
                            assert observed.points == 0, f"Missing nonzero actual: {p}"
                            zero_rows.append(
                                dict(week=week, player_id=pid, name=p["player_display_name"])
                            )
                            standard = 0.0
                        else:
                            assert all(v is not None for v in stats.values())
                            standard = 0.1 * stats["rushing_yards"] + 6 * stats["rushing_tds"]
                            if p["position"] == "QB":
                                standard += (
                                    0.04 * stats["passing_yards"]
                                    + 4 * stats["passing_tds"]
                                    - 2 * stats["passing_interceptions"]
                                )
                            else:
                                standard += (
                                    stats["receptions"]
                                    + 0.1 * stats["receiving_yards"]
                                    + 6 * stats["receiving_tds"]
                                )
                        comparable += standard
                    players.append(
                        dict(
                            week=week,
                            team_id=team.team_id,
                            team_name=team.team_name,
                            espn_id=p["espn_id"],
                            player_id=pid,
                            name=p["player_display_name"],
                            position=p["position"],
                            origin_week=origin,
                            predicted=p["prediction"],
                            actual_model_scoring=standard,
                            actual_espn=observed.points if completed else None,
                            basis=p["basis"],
                            unavailable=p["unavailable_reason"],
                        )
                    )
                actual_full = side["score"] if completed else None
                if completed:
                    assert (
                        abs(sum(lineup[p["espn_id"]].points for p in starters) - actual_full) < 1e-6
                    ), "Lineup does not reconcile to team score"
                    assert abs(actual_full - entry.score) < 1e-6
                row = dict(
                    week=week,
                    team_id=team.team_id,
                    team_name=team.team_name,
                    completed=completed,
                    origin_week=origin,
                    horizon=horizon,
                    predicted_offense=sum(p["prediction"] for p in offense) if covered else None,
                    actual_model_scoring=comparable if completed else None,
                    actual_offense=sum(lineup[p["espn_id"]].points for p in offense)
                    if completed
                    else None,
                    predicted_k_dst=special_total,
                    predicted_full=sum(p["prediction"] for p in starters)
                    if all(p["prediction"] is not None for p in starters)
                    else None,
                    actual_full=actual_full,
                    espn_projected_offense=sum(
                        lineup[p["espn_id"]].projected_points for p in offense
                    )
                    if all(lineup[p["espn_id"]].projected_points is not None for p in offense)
                    else None,
                    covered_offense=sum(p["prediction"] is not None for p in offense),
                    offensive_starters=len(offense),
                )
                teams.append(row)
                pair.append(row)
            home, away = pair
            margin = (
                home["predicted_full"] - away["predicted_full"]
                if all(r["predicted_full"] is not None for r in pair)
                else None
            )
            actual_margin = (
                home["actual_full"] - away["actual_full"]
                if all(r["completed"] for r in pair)
                else None
            )
            games.append(
                dict(
                    week=week,
                    home=home["team_name"],
                    away=away["team_name"],
                    predicted_margin=margin,
                    actual_margin=actual_margin,
                    correct=bool(margin * actual_margin > 0)
                    if margin not in (None, 0) and actual_margin not in (None, 0)
                    else None,
                )
            )
    assert store.campaign("opening_drive_v2")["manifest_sha256"] == manifest["manifest_sha256"]
    summary = []
    for t in snapshot.teams:
        rows = [r for r in teams if r["team_id"] == t.team_id and r["completed"]]
        m = measures(rows, "predicted_offense", "actual_model_scoring")
        full = measures(rows, "predicted_full", "actual_full")
        summary.append(
            dict(
                team_id=t.team_id,
                team_name=t.team_name,
                weeks=m["n"],
                predicted_offense=m["predicted_mean"],
                actual_model_scoring=m["actual_mean"],
                actual_espn_offense=float(np.mean([r["actual_offense"] for r in rows])),
                actual_espn_full=float(np.mean([r["actual_full"] for r in rows])),
                mae=m["mae"],
                bias=m["bias"],
                full_weeks=full["n"],
                predicted_full=full.get("predicted_mean"),
                predicted_k_dst=float(np.mean([r["predicted_k_dst"] for r in rows]))
                if all(r["predicted_k_dst"] is not None for r in rows)
                else None,
                full_mae=full.get("mae"),
                full_bias=full.get("bias"),
            )
        )
    evaluated = [g for g in games if g["correct"] is not None]
    metrics = dict(
        model_scoring=measures(teams, "predicted_offense", "actual_model_scoring"),
        espn_offense=measures(teams, "predicted_offense", "actual_offense"),
        hybrid_full_team=measures(teams, "predicted_full", "actual_full"),
        espn_reference=measures(teams, "espn_projected_offense", "actual_offense"),
        full_team_winners=dict(correct=sum(g["correct"] for g in evaluated), n=len(evaluated)),
    )
    write_csv("team_weeks.csv", teams)
    write_csv("team_summary.csv", summary)
    write_csv("player_weeks.csv", players)
    write_csv("matchups.csv", games)
    notes = [
        "Retrospective reconstruction, not verified pregame performance: v2 artifacts were written "
        "after completed games. Historical lineups are the starters managers actually selected.",
        "Weeks 1–2 use preseason v2; Weeks 3–4 use the after-Week-2 v2 forecast; Week 5 uses "
        "after-Week-4 v2. Forecast origins always precede target weeks. TE is included in v2.",
        "Player horizon points / remaining scheduled games, summed over actual offensive starters. "
        "Byes and empty starting slots receive zero; no opponent adjustment. "
        "These are average-game estimates. One team had seven offensive starters in Week 2.",
        "Model-scoring actuals reproduce the campaign contract: QB 4-point passing TD, "
        "0.04 passing yards, -2 interceptions, 0.1 rush yards, 6 rush TD; other offense "
        "PPR receiving plus rushing. Fumbles, bonuses, 2-point plays and cross-position "
        "production are excluded as specified by the model.",
        "Full-team prediction = Opening Drive offense + kicker estimate + defense estimate. "
        "K/DST use the mean of up to four earlier recorded ESPN scores. When no earlier "
        "score exists (including Week 1), use that player's recorded ESPN weekly projection. "
        "These are simple supplements, not Opening Drive models. No same-week or future "
        "actual score enters a prediction. Player-level sources are in player_weeks.csv.",
        "Full-team actuals are the official ESPN team scores, including K/DST. Opening Drive's "
        "offensive scoring contract still differs from the league's bonuses and other rules; "
        "adding K/DST does not change or calibrate offensive forecasts.",
        "Week 5 is incomplete and excluded from every error metric and winner result. "
        "Missing current actuals are blank, never zero. ESPN projected offense is a provider "
        "reference captured retrospectively, not a verified pregame archive.",
        "Three completed starter rows absent from the complete weekly NFL stat feed have "
        "independent ESPN zero scores and are treated as zero model-scoring production; "
        "their identities are retained in results.json.",
    ]
    result = dict(
        generated_at=datetime.now(UTC).isoformat(),
        season=snapshot.season,
        league=snapshot.league_name,
        captured_at=snapshot.captured_at,
        through_week=max(r["week"] for r in teams if r["completed"]),
        manifest_sha256=manifest["manifest_sha256"],
        snapshot_sha256=sha256(snapshot_bytes).hexdigest(),
        metrics=metrics,
        notes=notes,
        zero_stat_rows=zero_rows,
        teams=summary,
        team_weeks=teams,
    )
    (OUT / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    (OUT / "inputs.json").write_text(
        json.dumps(
            dict(
                snapshot=json.loads(snapshot_bytes),
                rankings=vintages,
                schedule=schedule,
                identities=identities,
                actuals=actuals,
                catalog=catalog,
            ),
            allow_nan=False,
            default=str,
        )
    )
    render(snapshot, teams, summary, result)
    print(json.dumps(result, indent=2, allow_nan=False))


def render(snapshot, rows, summary, result):
    from html import escape

    fig = make_subplots(
        rows=5, cols=2, subplot_titles=[t.team_name for t in snapshot.teams], vertical_spacing=0.065
    )
    for i, t in enumerate(snapshot.teams):
        rs = [r for r in rows if r["team_id"] == t.team_id]
        for key, label, color, dash in [
            ("predicted_full", "Predicted · OD + K/DST", "#2563eb", "dash"),
            ("actual_full", "Actual · ESPN team score", "#059669", "solid"),
        ]:
            fig.add_trace(
                go.Scatter(
                    x=[r["week"] for r in rs],
                    y=[r[key] for r in rs],
                    name=label,
                    legendgroup=key,
                    showlegend=i == 0,
                    mode="lines+markers",
                    line=dict(color=color, dash=dash),
                    hovertemplate="Week %{x}: %{y:.1f} points<extra>%{fullData.name}</extra>",
                ),
                row=i // 2 + 1,
                col=i % 2 + 1,
            )
    fig.update_xaxes(dtick=1, title_text="Week", range=[0.8, snapshot.week + 0.2])
    ymax = max(r[key] or 0 for r in rows for key in ("predicted_full", "actual_full"))
    fig.update_yaxes(range=[0, max(200, ymax * 1.1)], title_text="Full-team points")
    fig.update_layout(
        height=1550,
        template="plotly_white",
        legend=dict(orientation="h", y=1.05),
        margin=dict(t=90),
    )
    graph = fig.to_html(full_html=False, include_plotlyjs=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    chart, axes = plt.subplots(5, 2, figsize=(14, 16), sharex=True, sharey=True)
    for ax, team in zip(axes.flat, snapshot.teams, strict=True):
        rs = [r for r in rows if r["team_id"] == team.team_id and r["completed"]]
        weeks = [r["week"] for r in rs]
        ax.plot(
            weeks,
            [r["predicted_full"] for r in rs],
            "o--",
            color="#2563eb",
            label="Predicted · OD + K/DST",
        )
        ax.plot(
            weeks,
            [r["actual_full"] for r in rs],
            "o-",
            color="#059669",
            label="Actual · ESPN team score",
        )
        ax.set_title(team.team_name, loc="left", fontsize=12)
        ax.set_xticks(weeks)
        ax.set_ylim(0, max(200, ymax * 1.1))
        ax.grid(alpha=0.2)
        ax.set_xlabel("Week")
        ax.set_ylabel("Points")
    chart.suptitle("Opening Drive v2 · Sweaty Plays · Weeks 1–4", fontsize=20, y=0.99)
    chart.legend(
        *axes[0, 0].get_legend_handles_labels(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.97),
        ncol=2,
        frameon=False,
    )
    chart.text(
        0.5,
        0.01,
        "Retrospective reconstruction · Actual weekly starters · "
        "K/DST: prior ESPN averages, with ESPN projections when no history exists",
        ha="center",
        fontsize=10,
    )
    chart.tight_layout(rect=(0, 0.03, 1, 0.94))
    chart.savefig(OUT / "team_comparison.png", dpi=150)
    plt.close(chart)

    def cell(value):
        if value is None:
            return "—"
        return f"{value:.1f}" if isinstance(value, float) else escape(str(value))

    def table(rs, keys):
        return (
            "<table><thead><tr>"
            + "".join(f"<th>{escape(label)}</th>" for _, label in keys)
            + "</tr></thead><tbody>"
            + "".join(
                "<tr>" + "".join(f"<td>{cell(r[k])}</td>" for k, _ in keys) + "</tr>" for r in rs
            )
            + "</tbody></table>"
        )

    metrics = result["metrics"]
    score = metrics["hybrid_full_team"]
    winners = metrics["full_team_winners"]
    html = (
        "<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' "
        "content='width=device-width, initial-scale=1'><title>Opening Drive team accuracy</title>"
        "<style>body{font:16px system-ui;color:#172033;max-width:1250px;"
        "margin:40px auto;padding:0 20px}p,li{line-height:1.6}"
        "table{border-collapse:collapse;width:100%;font-size:14px;margin:24px 0}"
        "td,th{padding:10px;text-align:right;border-bottom:1px solid #ddd}"
        "td:first-child,th:first-child"
        "{text-align:left}th{background:#eef2ff}.notice{padding:16px;background:#fff3d6}"
        "section{overflow:auto}a{color:#2563eb}</style>"
        "<h1>Opening Drive v2 + kickers &amp; defenses</h1>"
        f"<p>{escape(snapshot.league_name)} · {snapshot.season} · "
        f"Completed Weeks 1–{result['through_week']} "
        f"· ESPN captured {escape(snapshot.captured_at)}</p>"
        "<p class='notice'>Retrospective reconstruction. "
        "These are not forecasts saved before the games. "
        "Week 5 predictions are shown for context; its results are excluded.</p>"
        f"<p><strong>{score['mae']:.1f} points mean absolute error</strong> across {score['n']} "
        f"completed team-weeks, including K/DST. Mean bias: {score['bias']:+.1f} points "
        f"(prediction minus actual). Full-team hybrid picks: {winners['correct']}/{winners['n']} "
        "correct in completed non-tied matchups.</p>"
        f"<p><strong>Predicted: {score['predicted_mean']:.1f} points/week · "
        f"Actual: {score['actual_mean']:.1f} points/week.</strong> "
        "Prediction = Opening Drive offense + prior ESPN kicker/defense averages. "
        "When there is no earlier K/DST history, we use ESPN's weekly projection. "
        "Actuals are full ESPN team scores. Hover over the charts for exact values.</p>"
        + graph
        + "<h2>Average points per completed week</h2><section>"
        + table(
            summary,
            [
                ("team_name", "Team"),
                ("predicted_offense", "OD offense"),
                ("predicted_k_dst", "K + D/ST estimate"),
                ("predicted_full", "Full prediction"),
                ("actual_espn_full", "Actual, full team"),
                ("full_mae", "Full-team MAE"),
            ],
        )
        + "</section><h2>Every team and week</h2><p>— = unavailable or incomplete. "
        "Full prediction = OD offense + K/DST estimate. "
        "Actual = the official ESPN score.</p><section>"
        + table(
            rows,
            [
                ("team_name", "Team"),
                ("week", "Week"),
                ("origin_week", "Model cutoff"),
                ("predicted_offense", "OD offense"),
                ("predicted_k_dst", "K + D/ST estimate"),
                ("predicted_full", "Full prediction"),
                ("actual_full", "Actual, full team"),
            ],
        )
        + "</section><h2>Method and limitations</h2><ul>"
        + "".join(f"<li>{escape(n)}</li>" for n in result["notes"])
        + "</ul>"
        + "<p>Downloads: <a href='team_weeks.csv'>Team weeks</a> · "
        "<a href='player_weeks.csv'>Player detail</a> · "
        "<a href='results.json'>Metrics and provenance</a></p>" + "</html>"
    )
    (OUT / "index.html").write_text(html)


if __name__ == "__main__":
    main()
