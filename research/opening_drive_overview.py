"""Build a standalone Python-chart report from the team accuracy artifact.

Run: uv run python research/opening_drive_overview.py
Historical inputs stay frozen; Week 5 uses the current local league snapshot.
No model, source artifact, forecast log, or remote service is modified.
"""

import csv
import io
import json
from collections import defaultdict
from dataclasses import replace
from datetime import UTC, datetime
from hashlib import sha256
from html import escape
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from engine.chairstats.store import CampaignStore
from engine.chairstats.views import rankings
from engine.espn.sync import LeagueSnapshot
from engine.metrics.weekly_lineup import weekly_forecast
from engine.tables import observations
from engine.tables.application import connect, query_scope

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/outputs/opening_drive_team_accuracy_2026"
OUT = ROOT / "data/outputs/opening_drive_overview_2026"
OFFENSE = {"QB", "RB", "WR", "TE"}
BLUE, GREEN, INK = "#4264e8", "#168a74", "#202c43"


def read_json(path):
    return json.loads(path.read_text())


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def mean(values):
    return float(np.mean(values))


def stats(rows, prediction="predicted_full", actual="actual_full"):
    pred = np.array([r[prediction] for r in rows])
    obs = np.array([r[actual] for r in rows])
    return dict(
        n=len(rows),
        predicted_mean=float(pred.mean()),
        actual_mean=float(obs.mean()),
        mae=float(np.abs(pred - obs).mean()),
        rmse=float(np.sqrt(((pred - obs) ** 2).mean())),
        bias=float((pred - obs).mean()),
        underpredicted=int((pred < obs).sum()),
        predicted_sd=float(pred.std(ddof=1)),
        actual_sd=float(obs.std(ddof=1)),
    )


def winner(home, away, key):
    if home[key] is None or away[key] is None:
        return None
    if np.isclose(home[key], away[key], rtol=0, atol=1e-9):
        return "Tie"
    return (home if home[key] > away[key] else away)["team_name"]


def build():
    frozen = read_json(SOURCE / "inputs.json")
    original = read_json(SOURCE / "results.json")
    historical = LeagueSnapshot.from_dict(frozen["snapshot"])
    live_path = ROOT / "data/outputs/league_snapshot.json"
    live_bytes = live_path.read_bytes()
    live = LeagueSnapshot.from_dict(json.loads(live_bytes))
    if (live.season, live.week) != (2026, 5):
        raise ValueError("This report is pinned to 2026 Week 5; supply that league snapshot.")
    if live.league_id != historical.league_id:
        raise ValueError("The current snapshot belongs to a different league.")
    store = CampaignStore(ROOT / "data/campaigns")
    current = rankings(store, "opening_drive_v2", horizon="remaining")
    assert current["manifest_sha256"] == original["manifest_sha256"], "Model changed since audit"
    assert all(
        r["season"] == live.season and r["origin_week"] < live.week for r in current["rankings"]
    ), "Week 5 forecast has a future observation cutoff"
    with query_scope(ROOT / "data"), connect() as db:
        identities = observations.espn_ids(db)
        schedule = observations.schedule(db, live.season)
    with (SOURCE / "player_weeks.csv").open() as stream:
        old_players = {
            (int(r["week"]), int(r["team_id"]), str(r["espn_id"])): r
            for r in csv.DictReader(stream)
        }
    old_teams = {(r["week"], r["team_id"]): r for r in original["team_weeks"]}
    rows, players, matchups, responses = [], [], [], {}
    for week in range(1, 6):
        ranks = current if week == 5 else frozen["rankings"][str(week)]
        assert all(r["origin_week"] < week for r in ranks["rankings"])
        response = weekly_forecast(
            live if week == 5 else replace(historical, week=week),
            ranks,
            identities if week == 5 else frozen["identities"],
            schedule if week == 5 else frozen["schedule"],
        )
        responses[str(week)] = response
        for game in response["matchups"]:
            pair = []
            for side in (game["home"], game["away"]):
                old = old_teams[week, side["team_id"]]
                completed = week < 5 and old["completed"]
                starters = [p for p in side["forecasts"] if p["started"]]
                assert starters and all(p["prediction"] is not None for p in starters)
                offense = sum(p["prediction"] for p in starters if p["position"] in OFFENSE)
                special = sum(p["prediction"] for p in starters if p["position"] not in OFFENSE)
                row = dict(
                    week=week,
                    team_id=side["team_id"],
                    team_name=side["team_name"],
                    completed=completed,
                    origin_week=response["through_week"],
                    horizon=ranks["horizon"],
                    predicted_offense=offense,
                    predicted_k_dst=special,
                    predicted_full=side["model_projection"],
                    actual_full=old["actual_full"] if completed else None,
                    actual_offense=old["actual_offense"] if completed else None,
                    actual_model_scoring=old["actual_model_scoring"] if completed else None,
                )
                assert np.isclose(offense + special, row["predicted_full"])
                if completed:
                    assert np.isclose(offense, old["predicted_offense"])
                    assert np.isclose(side["score"], old["actual_full"])
                for p in starters:
                    old_p = old_players.get((week, side["team_id"], str(p["espn_id"])))
                    assert not completed or old_p is not None
                    players.append(
                        dict(
                            week=week,
                            team_id=side["team_id"],
                            team_name=side["team_name"],
                            espn_id=p["espn_id"],
                            name=p["player_display_name"],
                            position=p["position"],
                            predicted=p["prediction"],
                            basis=p["basis"],
                            actual=float(old_p["actual_espn"]) if completed else None,
                        )
                    )
                if completed:
                    assert np.isclose(
                        sum(
                            p["actual"]
                            for p in players
                            if p["week"] == week and p["team_id"] == side["team_id"]
                        ),
                        row["actual_full"],
                    )
                rows.append(row)
                pair.append(row)
            home, away = pair
            predicted = winner(home, away, "predicted_full")
            actual = winner(home, away, "actual_full")
            correct = predicted == actual if actual and "Tie" not in (actual, predicted) else None
            matchups.append(
                dict(
                    week=week,
                    home=home,
                    away=away,
                    predicted_winner=predicted,
                    actual_winner=actual,
                    correct=correct,
                    predicted_margin=abs(home["predicted_full"] - away["predicted_full"]),
                    actual_margin=abs(home["actual_full"] - away["actual_full"])
                    if actual
                    else None,
                )
            )
    complete = [r for r in rows if r["completed"]]
    evaluated = [g for g in matchups if g["correct"] is not None]
    positions = defaultdict(list)
    for p in players:
        if p["actual"] is not None:
            positions[p["position"]].append(p)
    position = [
        dict(
            position=k,
            predicted=sum(p["predicted"] for p in v) / len(complete),
            actual=sum(p["actual"] for p in v) / len(complete),
            gap=sum(p["actual"] - p["predicted"] for p in v) / len(complete),
        )
        for k, v in positions.items()
    ]
    teams = []
    for team in live.teams:
        history = [r for r in complete if r["team_id"] == team.team_id]
        upcoming = next(r for r in rows if r["week"] == 5 and r["team_id"] == team.team_id)
        teams.append(
            dict(
                team_id=team.team_id,
                team_name=team.team_name,
                **stats(history),
                actual_min=min(r["actual_full"] for r in history),
                actual_max=max(r["actual_full"] for r in history),
                week5=upcoming["predicted_full"],
            )
        )
    report = dict(
        generated_at=datetime.now(UTC).isoformat(),
        league=live.league_name,
        season=live.season,
        forecast_week=5,
        completed_weeks=[1, 2, 3, 4],
        my_team_id=live.my_team_id,
        historical_capture=historical.captured_at,
        forecast_capture=live.captured_at,
        metrics=stats(complete),
        model_scoring=stats(complete, "predicted_offense", "actual_model_scoring"),
        winners=dict(
            correct=sum(g["correct"] for g in evaluated),
            n=len(evaluated),
            ties=sum(g["actual_winner"] == "Tie" for g in matchups),
        ),
        teams=teams,
        team_weeks=rows,
        players=players,
        matchups=matchups,
        positions=position,
        provenance=dict(
            campaign="opening_drive_v2",
            manifest_sha256=current["manifest_sha256"],
            source_inputs_sha256=digest(SOURCE / "inputs.json"),
            source_results_sha256=digest(SOURCE / "results.json"),
            source_player_weeks_sha256=digest(SOURCE / "player_weeks.csv"),
            current_snapshot_sha256=sha256(live_bytes).hexdigest(),
            calculation="engine.metrics.weekly_lineup.weekly_forecast",
        ),
    )
    assert len(complete) == 40 and len(matchups) == 25 and len(teams) == 10
    assert np.isclose(sum(p["gap"] for p in position), -report["metrics"]["bias"])
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = dict(
        historical_inputs=frozen,
        current_snapshot=json.loads(live_bytes),
        current_rankings=current,
        current_identities=identities,
        current_schedule=schedule,
        weekly_forecasts=responses,
    )
    (OUT / "inputs.json").write_text(json.dumps(inputs, allow_nan=False, default=str))
    (OUT / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    for name, records in [("team_weeks", rows), ("team_summary", teams), ("player_weeks", players)]:
        write_csv(name, records)
    write_csv(
        "matchups",
        [
            dict(
                week=g["week"],
                home=g["home"]["team_name"],
                away=g["away"]["team_name"],
                home_predicted=g["home"]["predicted_full"],
                away_predicted=g["away"]["predicted_full"],
                home_actual=g["home"]["actual_full"],
                away_actual=g["away"]["actual_full"],
                predicted_winner=g["predicted_winner"],
                actual_winner=g["actual_winner"],
                correct=g["correct"],
            )
            for g in matchups
        ],
    )
    return report


def write_csv(name, records):
    with (OUT / f"{name}.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def save_chart(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")
    stream = io.StringIO()
    fig.savefig(stream, format="svg", bbox_inches="tight")
    plt.close(fig)
    return stream.getvalue()[stream.getvalue().index("<svg") :]


def charts(data):
    plt.rcParams.update(
        {
            "font.family": ["DejaVu Sans", "sans-serif"],
            "font.size": 10,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": "#647086",
            "ytick.color": "#647086",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.spines.bottom": False,
            "axes.axisbelow": True,
            "axes.grid": True,
            "grid.alpha": 0.15,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
        }
    )
    rows = data["team_weeks"]
    complete = [r for r in rows if r["completed"]]
    fig, ax = plt.subplots(figsize=(11.5, 3.8), layout="constrained")
    weeks = [1, 2, 3, 4]
    avg = [mean([r["actual_full"] for r in complete if r["week"] == w]) for w in weeks]
    sd = [
        float(np.std([r["actual_full"] for r in complete if r["week"] == w], ddof=1)) for w in weeks
    ]
    pred = [mean([r["predicted_full"] for r in rows if r["week"] == w]) for w in range(1, 6)]
    ax.fill_between(
        weeks,
        np.array(avg) - sd,
        np.array(avg) + sd,
        color=GREEN,
        alpha=0.1,
        label="Actual team scores: mean ± 1 sample SD",
    )
    ax.plot(weeks, avg, "o-", color=GREEN, lw=2.5, label="Actual average")
    ax.plot(weeks, pred[:4], "o-", color=BLUE, lw=2.5, label="Predicted average · v2 + K/DST")
    ax.plot([4, 5], pred[3:], "o--", color=BLUE, lw=2)
    ax.axvspan(4.5, 5.4, color=BLUE, alpha=0.04)
    for w, a, p in zip(weeks, avg, pred, strict=False):
        ax.annotate(
            f"{a:.1f}", (w, a), xytext=(0, 9), textcoords="offset points", ha="center", color=GREEN
        )
        ax.annotate(
            f"{p:.1f}", (w, p), xytext=(0, -18), textcoords="offset points", ha="center", color=BLUE
        )
    ax.annotate(
        f"{pred[4]:.1f}\nforecast",
        (5, pred[4]),
        xytext=(0, 12),
        textcoords="offset points",
        ha="center",
        color=BLUE,
        weight="bold",
    )
    ax.set(
        xticks=range(1, 6),
        xticklabels=["Week 1", "Week 2", "Week 3", "Week 4", "Week 5"],
        xlim=(0.7, 5.4),
        ylim=(0, max(np.array(avg) + sd) + 25),
        ylabel="Points per team",
    )
    ax.legend(loc="upper center", ncols=3, frameon=False, fontsize=9)
    league = save_chart(fig, "league_trend")

    ordered = sorted(data["teams"], key=lambda t: t["actual_mean"], reverse=True)
    fig, ax = plt.subplots(figsize=(11.5, 5.5), layout="constrained")
    for y, t in enumerate(ordered):
        ax.plot(
            [t["actual_min"], t["actual_max"]],
            [y, y],
            color=GREEN,
            alpha=0.22,
            lw=9,
            solid_capstyle="round",
            label="Actual min–max, four weeks" if y == 0 else None,
        )
        values = [r["actual_full"] for r in complete if r["team_id"] == t["team_id"]]
        ax.scatter(values, [y] * len(values), color=GREEN, s=18, alpha=0.45)
        ax.scatter(
            t["actual_mean"],
            y,
            color=GREEN,
            s=80,
            edgecolor="white",
            zorder=3,
            label="Actual average" if y == 0 else None,
        )
        ax.scatter(
            t["predicted_mean"],
            y,
            color=BLUE,
            s=75,
            marker="D",
            edgecolor="white",
            zorder=4,
            label="Predicted average" if y == 0 else None,
        )
    ax.set(
        yticks=range(len(ordered)),
        yticklabels=[t["team_name"] for t in ordered],
        xlabel="Full-team points · Weeks 1–4",
        xlim=(0, 220),
    )
    ax.invert_yaxis()
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), ncols=3, frameon=False, fontsize=9)
    spread = save_chart(fig, "team_averages")

    fig, axes = plt.subplots(
        5, 2, figsize=(11.5, 13), sharex=True, sharey=True, layout="constrained"
    )
    for ax, t in zip(axes.flat, data["teams"], strict=True):
        rs = sorted([r for r in rows if r["team_id"] == t["team_id"]], key=lambda r: r["week"])
        ax.axvspan(4.5, 5.35, color=BLUE, alpha=0.05)
        ax.axhline(t["actual_mean"], color=GREEN, alpha=0.28, lw=1)
        ax.axhline(t["predicted_mean"], color=BLUE, alpha=0.28, lw=1)
        ax.plot(weeks, [r["actual_full"] for r in rs[:4]], "o-", color=GREEN, lw=2)
        ax.plot(range(1, 6), [r["predicted_full"] for r in rs], "o--", color=BLUE, lw=2)
        ax.annotate(
            f"{t['week5']:.1f}",
            (5, t["week5"]),
            xytext=(-4, 10),
            textcoords="offset points",
            ha="right",
            color=BLUE,
            weight="bold",
        )
        ax.set_title(t["team_name"], loc="left", fontsize=11, weight="bold")
        ax.set(xticks=range(1, 6), xlim=(0.7, 5.35), ylim=(0, 220), ylabel="Points")
        ax.tick_params(labelbottom=True)
    trajectories = save_chart(fig, "team_trajectories")
    return league, spread, trajectories


def fmt(value):
    return "—" if value is None else f"{value:.1f}"


def table(headers, rows):
    return (
        '<div class="table-scroll"><table><thead><tr>'
        + "".join(f"<th>{escape(h)}</th>" for h in headers)
        + "</tr></thead><tbody>"
        + "".join("<tr>" + "".join(f"<td>{escape(str(c))}</td>" for c in r) + "</tr>" for r in rows)
        + "</tbody></table></div>"
    )


def matchup_card(g, players=None):
    future = g["week"] == 5
    status = (
        "Week 5 forecast"
        if future
        else (
            "Actual tie · ungraded"
            if g["actual_winner"] == "Tie"
            else "No pick · ungraded"
            if g["correct"] is None
            else "Correct pick"
            if g["correct"]
            else "Missed pick"
        )
    )
    tone = "neutral" if future or g["correct"] is None else "good" if g["correct"] else "bad"
    team_ids = f"{g['home']['team_id']},{g['away']['team_id']}"
    body = f'<article class="matchup" data-week="{g["week"]}" data-teams="{team_ids}">'
    body += (
        f'<div class="matchup-top"><span>WEEK {g["week"]}</span>'
        f'<span class="badge {tone}">{status}</span></div>'
    )
    body += (
        '<div class="score-head"><span>Team</span><span>Predicted</span><span>Actual</span></div>'
    )
    for side in (g["home"], g["away"]):
        body += (
            '<div class="score-row"><strong>'
            + escape(side["team_name"])
            + f'</strong><span class="blue">{fmt(side["predicted_full"])}</span>'
            + f'<span class="green">{fmt(side["actual_full"])}</span></div>'
        )
    pick = escape(g["predicted_winner"])
    body += f'<p class="pick"><b>Model pick:</b> {pick} · {g["predicted_margin"]:.1f} pts</p>'
    if not future:
        actual = escape(g["actual_winner"] or "Pending")
        body += f'<p class="result"><b>Actual:</b> {actual} · {fmt(g["actual_margin"])} pts</p>'
    else:
        body += '<p class="result">V2 offense + ESPN K/DST · results pending</p>'
        body += "<details><summary>Starting lineup breakdown</summary>"
        for side in (g["home"], g["away"]):
            body += (
                "<h4>"
                + escape(side["team_name"])
                + f" · {side['predicted_offense']:.1f} offense"
                + f" + {side['predicted_k_dst']:.1f} K/DST</h4>"
            )
            ps = [p for p in players if p["week"] == 5 and p["team_id"] == side["team_id"]]
            body += table(
                ["Player", "Pos", "Pts"],
                [(p["name"], p["position"], fmt(p["predicted"])) for p in ps],
            )
        body += "</details>"
    return body + "</article>"


def render(data):
    league, spread, trajectories = charts(data)
    m, wins = data["metrics"], data["winners"]
    pct = wins["correct"] / wins["n"] * 100
    next_avg = mean([t["week5"] for t in data["teams"]])
    mine = next(t for t in data["teams"] if t["team_id"] == data["my_team_id"])
    my_game = next(
        g
        for g in data["matchups"]
        if g["week"] == 5 and data["my_team_id"] in (g["home"]["team_id"], g["away"]["team_id"])
    )
    opponent = next(
        s for s in (my_game["home"], my_game["away"]) if s["team_id"] != mine["team_id"]
    )
    cards = "".join(
        f'<div class="metric"><span>{label}</span><strong>{value}</strong>'
        f"<small>{note}</small></div>"
        for label, value, note in [
            ("Predicted average", f"{m['predicted_mean']:.1f}", "pts / team-week · v2 + K/DST"),
            ("Actual average", f"{m['actual_mean']:.1f}", "pts / team-week · ESPN scores"),
            ("Average absolute miss", f"{m['mae']:.1f}", "points away from the actual score"),
            (
                "Winner accuracy",
                f"{wins['correct']}/{wins['n']}",
                f"{pct:.0f}% · actual ties excluded",
            ),
        ]
    )
    historical_cards = "".join(matchup_card(g) for g in data["matchups"] if g["week"] < 5)
    future_cards = "".join(
        matchup_card(g, data["players"]) for g in data["matchups"] if g["week"] == 5
    )
    team_options = "".join(
        f'<option value="{t["team_id"]}">{escape(t["team_name"])}</option>' for t in data["teams"]
    )
    team_table = table(
        ["Team", "Pred. avg", "Actual avg", "Actual range", "Abs. miss", "Week 5"],
        [
            (
                t["team_name"],
                fmt(t["predicted_mean"]),
                fmt(t["actual_mean"]),
                f"{t['actual_min']:.0f}–{t['actual_max']:.0f}",
                fmt(t["mae"]),
                fmt(t["week5"]),
            )
            for t in sorted(data["teams"], key=lambda r: r["week5"], reverse=True)
        ],
    )
    position_table = table(
        ["Position", "Predicted / team", "Actual / team", "Actual − predicted"],
        [
            (p["position"], fmt(p["predicted"]), fmt(p["actual"]), f"{p['gap']:+.1f}")
            for p in sorted(data["positions"], key=lambda p: -p["gap"])
        ],
    )
    weekly_table = table(
        ["Week", "Model cutoff", "Predicted avg", "Actual avg", "Abs. miss", "Winner picks"],
        [
            (
                w,
                "Preseason" if w <= 2 else "After Week 2",
                fmt(stats([r for r in data["team_weeks"] if r["week"] == w])["predicted_mean"]),
                fmt(stats([r for r in data["team_weeks"] if r["week"] == w])["actual_mean"]),
                fmt(stats([r for r in data["team_weeks"] if r["week"] == w])["mae"]),
                f"{sum(g['correct'] is True for g in data['matchups'] if g['week'] == w)}/"
                f"{sum(g['correct'] is not None for g in data['matchups'] if g['week'] == w)}",
            )
            for w in [1, 2, 3, 4]
        ],
    )
    rbwr = sum(p["gap"] for p in data["positions"] if p["position"] in {"RB", "WR"})
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Opening Drive · Through Week 4 / Week 5 outlook</title>
<style>
:root{{--ink:#202c43;--muted:#667187;--blue:#4264e8;--green:#168a74;--line:#e3e7ee}}
*{{box-sizing:border-box}}html{{scroll-behavior:smooth}}body{{margin:0;background:#f4f6fa;color:var(--ink);font:16px/1.6 system-ui,-apple-system,sans-serif}}
main{{max-width:1216px;margin:auto;padding:42px 28px 70px}}h1,h2,h3,p{{margin-top:0}}h1{{font-size:clamp(32px,5vw,56px);line-height:1.08;letter-spacing:-2px;max-width:850px;margin-bottom:22px}}h2{{font-size:27px;letter-spacing:-.7px;line-height:1.25}}h3{{font-size:19px}}a{{color:var(--blue)}}header{{padding:24px 0 30px}}.eyebrow{{font-size:12px;font-weight:800;letter-spacing:2px;color:var(--blue);text-transform:uppercase;margin-bottom:20px}}.lead{{font-size:19px;max-width:870px;color:#536078}}.tag{{display:inline-block;border:1px solid #d4dae6;border-radius:6px;font-size:12px;padding:3px 9px;color:var(--muted)}}nav{{display:flex;gap:22px;flex-wrap:wrap;margin:22px 0 0}}nav a{{text-decoration:none;font-size:14px;font-weight:650}}section{{margin-top:32px;scroll-margin-top:20px}}.panel{{background:white;border:1px solid var(--line);border-radius:16px;padding:28px;box-shadow:0 3px 10px #202c4303}}.metrics{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}}.metric{{background:white;border:1px solid var(--line);border-radius:12px;padding:20px}}.metric span,.metric small{{display:block;color:var(--muted)}}.metric span{{font-size:13px;font-weight:650}}.metric strong{{font-size:40px;line-height:1.6;letter-spacing:-1px}}.metric small{{font-size:12px}}.readout{{border-left:4px solid var(--blue);padding:18px 22px;background:#eaf0ff;border-radius:0 10px 10px 0;margin:22px 0}}.readout p{{margin:0}}.caption,.muted{{color:var(--muted);font-size:14px}}.caption{{margin:14px 0 0}}.chart svg{{width:100%;height:auto;display:block}}.chart{{overflow:hidden}}.section-head{{display:flex;justify-content:space-between;gap:20px;align-items:baseline}}.section-head a{{font-size:12px;white-space:nowrap}}.legend{{display:flex;gap:22px;flex-wrap:wrap;font-size:13px;margin:16px 0}}.blue{{color:var(--blue)}}.green{{color:var(--green)}}.matchups{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}}.matchup{{background:white;border:1px solid var(--line);border-radius:12px;padding:20px;min-width:0}}.matchup-top{{display:flex;align-items:center;justify-content:space-between;font-size:11px;color:var(--muted);font-weight:750;letter-spacing:.5px;margin-bottom:18px}}.badge{{padding:3px 8px;border-radius:5px;font-size:10px;letter-spacing:0}}.good{{background:#e4f3ee;color:#11705f}}.bad{{background:#fff0e8;color:#ac4f21}}.neutral{{background:#edf0f9;color:#566488}}.score-head,.score-row{{display:grid;grid-template-columns:minmax(0,1fr) 78px 58px;gap:10px;align-items:center}}.score-head{{font-size:10px;color:var(--muted);text-transform:uppercase}}.score-row{{padding:12px 0;border-bottom:1px solid #edf0f5}}.score-row strong{{font-size:14px;overflow-wrap:anywhere}}.score-row span{{font-size:21px;font-variant-numeric:tabular-nums;font-weight:700;text-align:right}}.score-head span:not(:first-child){{text-align:right}}.pick,.result{{font-size:12px;margin:12px 0 0}}.result{{color:var(--muted);margin-top:4px}}.filters{{display:flex;gap:14px;flex-wrap:wrap;align-items:center;margin:18px 0}}label{{font-size:13px;color:var(--muted)}}select{{margin-left:6px;border:1px solid #d5dce7;background:white;padding:8px;border-radius:6px;color:var(--ink);max-width:100%}}.table-scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:13px;margin:14px 0}}th,td{{padding:12px 10px;text-align:right;border-bottom:1px solid #e9edf3;white-space:nowrap;font-variant-numeric:tabular-nums}}th{{font-size:11px;color:var(--muted);font-weight:650;background:#f8f9fc}}td:first-child,th:first-child{{text-align:left}}details{{margin-top:18px}}summary{{cursor:pointer;font-size:13px;font-weight:650;color:var(--blue)}}details h4{{font-size:13px;margin:20px 0 0}}.columns{{display:grid;grid-template-columns:1fr 1fr;gap:30px}}.fine{{font-size:12px;color:var(--muted)}}footer{{margin-top:32px}}.downloads{{display:flex;gap:20px;flex-wrap:wrap}}[hidden]{{display:none!important}}.forecast-note{{padding:16px 20px;border-radius:10px;background:#eaf0ff;margin-bottom:20px}}.forecast-note p{{margin:0;font-size:14px}}@media(max-width:700px){{main{{padding:20px 14px 40px}}.panel{{padding:18px}}.metrics{{grid-template-columns:1fr 1fr}}.metric{{padding:14px}}.metric strong{{font-size:32px}}.matchups,.columns{{grid-template-columns:1fr}}.section-head{{display:block}}.filters{{display:block}}label{{display:block;margin-bottom:10px}}h2{{font-size:23px}}.chart-wide{{overflow:auto}}.chart-wide svg{{min-width:650px}}}}@media print{{body{{background:white}}main{{max-width:none;padding:0}}nav,.filters{{display:none}}.panel,.matchup{{break-inside:avoid;box-shadow:none}}.matchup[hidden]{{display:block!important}}.metrics{{grid-template-columns:repeat(4,1fr)}}a{{color:inherit}}}}
</style></head><body><main>
<header><div class="eyebrow">Armchair Labs / Opening Drive v2</div>
<h1>Four weeks of results.<br>One clear Week 5 outlook.</h1>
<p class="lead">{escape(data["league"])} · 2026. A look at the model’s scoring estimates, the games it would have picked, and the next set of matchups.</p>
<span class="tag">Weeks 1–4: retrospective reconstruction</span> <span class="tag">Week 5: forecast only</span>
<nav><a href="#scorecard">Performance</a><a href="#variation">Averages & variation</a><a href="#history">Past matchups</a><a href="#week5">Week 5</a><a href="#analysis">Why the gap?</a></nav></header>
<section id="scorecard"><div class="metrics">{cards}</div>
<div class="readout"><p><strong>The estimates are running {abs(m["bias"]):.1f} points low per team-week.</strong> Actual scores exceeded predictions in {m["underpredicted"]} of {m["n"]} team-weeks. The reconstructed picks got {wins["correct"]} of {wins["n"]} non-tied games right; {wins["ties"]} actual ties are ungraded.</p></div>
<p class="fine">These are forecasts reconstructed after the games with observation cutoffs before each target week. They describe retrospective fit, not a verified live prediction record. All headline totals include K/DST.</p></section>
<section class="panel"><div class="section-head"><h2>The model average vs. what happened</h2><a href="league_trend.png">Download chart PNG ↗</a></div>
<div class="chart" role="img" aria-label="Predicted and actual league averages by week, with actual score dispersion and Week 5 forecast">{league}</div>
<p class="caption">The green band is the actual average ± one sample standard deviation across the 10 teams in each completed week. It shows observed score variation, not a model confidence interval. Week 5’s league forecast averages <strong>{next_avg:.1f} points</strong>; its actual results are excluded.</p>
<details><summary>Weekly results and model cutoffs</summary>{weekly_table}</details></section>
<section class="panel" id="variation"><div class="section-head"><h2>Each team’s average and weekly spread</h2><a href="team_averages.png">Download chart PNG ↗</a></div>
<div class="chart chart-wide" role="img" aria-label="Team predicted averages compared with actual averages and observed minimum to maximum scores">{spread}</div>
<p class="caption">Blue diamonds are predicted averages. Green dots are actual averages; small dots are individual weeks. The pale bars span each team’s actual lowest and highest score in Weeks 1–4. Four observations per team are descriptive, not reliable future bounds.</p>
<details><summary>All team averages, errors and Week 5 forecasts</summary>{team_table}</details></section>
<section id="history"><h2>Who was picked — and who actually won?</h2>
<p class="muted">Every completed matchup, with both teams’ predicted and actual points. Winners use unrounded values; displayed points are rounded to one decimal.</p>
<div class="filters"><label>Week<select id="week-filter"><option value="all">All completed weeks</option>{"".join(f'<option value="{w}">Week {w}</option>' for w in [1, 2, 3, 4])}</select></label><label>Team<select id="team-filter"><option value="all">All teams</option>{team_options}</select></label><span class="fine" id="matchup-count" aria-live="polite">20 matchups</span></div>
<div class="matchups" id="historical-matchups">{historical_cards}</div></section>
<section class="panel"><div class="section-head"><h2>The week-by-week story</h2><a href="team_trajectories.png">Download chart PNG ↗</a></div>
<div class="legend"><span class="blue">● Dashed: predicted points</span><span class="green">● Solid: actual points</span><span class="muted">Faint horizontal lines: Weeks 1–4 averages</span></div>
<div class="chart chart-wide" role="img" aria-label="Ten team charts comparing predictions with actual scores through Week 4 and showing Week 5 predictions">{trajectories}</div>
<p class="caption">The shaded final column is Week 5. Offensive forecasts use the saved v2 vintage available at each observation cutoff; K/DST use captured ESPN weekly projections, with earlier scores as fallback.</p></section>
<section id="week5"><h2>Week 5 · the next forecast</h2>
<p class="muted">Selected starters · v2 through Week 4 · ESPN snapshot {escape(data["forecast_capture"])}. Week 5 scores, including any partial results, are withheld throughout this report.</p>
<div class="forecast-note"><p><strong>Your matchup:</strong> {escape(mine["team_name"])} {mine["week5"]:.1f} vs. {escape(opponent["team_name"])} {opponent["predicted_full"]:.1f}. Model pick: <strong>{escape(my_game["predicted_winner"])}</strong> by {my_game["predicted_margin"]:.1f}. These are point estimates, not win probabilities.</p></div>
<div class="matchups">{future_cards}</div>
<p class="caption">No correction for the historical shortfall has been added. Week 5 uses the same calculation as the application overview at this snapshot; later lineup or ESPN projection updates can change it.</p></section>
<section class="panel" id="analysis"><h2>What explains the gap?</h2>
<div class="columns"><div><h3>Most of it is in the offense</h3><p>RB and WR together account for <strong>{rbwr:.1f} points</strong> of the {abs(m["bias"]):.1f}-point average team shortfall. The table compares contributions per team-week under ESPN’s actual league scoring.</p><p>Using the model’s own offensive scoring rules, v2 averages <strong>{data["model_scoring"]["predicted_mean"]:.1f}</strong> against <strong>{data["model_scoring"]["actual_mean"]:.1f}</strong> actual points. Scoring-rule differences do not explain away the low estimates.</p></div>
<div><h3>A season average is a different question</h3><p>V2 predicts horizon totals, then divides by scheduled games. That average includes the possibility of missed time or a reduced role. Managers select a weekly starting lineup based on who is available and useful now.</p><p>That mismatch is a plausible contributor, alongside conservative forecasts and a small early-season sample. These results do not identify a single cause, and a low average is not a calibrated scoring floor.</p></div></div>
{position_table}<p class="caption">Positive gaps mean actual production exceeded the estimate. Values are per team-week, not per player. K/DST are supplements; Opening Drive v2 covers QB/RB/WR/TE.</p></section>
<section class="panel"><h2>How to read this report</h2><div class="columns"><div>
<h3>One consistent calculation</h3><p>Full prediction = v2 offensive horizon points per scheduled game, summed over selected starters, plus captured ESPN K/DST projections. If a K/DST projection is missing, use up to four earlier recorded scores. Byes and empty slots contribute zero.</p>
<p>Weeks 1–2 use preseason v2; Weeks 3–4 use the forecast through Week 2; Week 5 uses v2 through Week 4. Historical lineups and actuals remain pinned to the original artifact.</p></div><div>
<h3>What the numbers can establish</h3><p>Mean absolute error measures the average size of a points miss. Winner accuracy grades only games with a non-tied prediction and result. Actual spread is descriptive; it is not forecast uncertainty.</p>
<p>Historical model files and ESPN references were captured after the games. This is a reconstruction, with no weekly opponent adjustment or validated win probabilities. Four weeks are too few to establish future calibration.</p></div></div>
<details><summary>Source snapshots and the change from the original artifact</summary><p class="fine">Historical ESPN snapshot: {escape(data["historical_capture"])}. Week 5 snapshot: {escape(data["forecast_capture"])}. Built: {escape(data["generated_at"])}.</p><p>The original artifact preferred past-score averages for K/DST. This report recomputes them with the application’s projection-first method. Consequently, historical full-team totals and winner accuracy can differ from the original report. The historical offensive predictions are unchanged, and the original files are preserved.</p><p class="fine">Campaign: opening_drive_v2. Model manifest SHA-256: {data["provenance"]["manifest_sha256"]}. Input hashes are recorded in report.json; inputs.json contains the frozen inputs and calculated weekly forecasts.</p></details></section>
<footer><div class="downloads"><a href="report.json">Metrics & provenance</a><a href="matchups.csv">Matchups CSV</a><a href="team_weeks.csv">Team weeks CSV</a><a href="player_weeks.csv">Player detail CSV</a><a href="inputs.json">Frozen inputs</a></div><p class="fine" style="margin-top:16px">Generated with Python / Matplotlib. Charts are embedded; this report works offline.</p></footer>
</main><script>
const week = document.getElementById('week-filter');
const team = document.getElementById('team-filter');
function filterGames() {{
  let shown = 0;
  document.querySelectorAll('#historical-matchups .matchup').forEach(card => {{
    const visible = (week.value === 'all' || card.dataset.week === week.value)
      && (team.value === 'all' || card.dataset.teams.split(',').includes(team.value));
    card.hidden = !visible; shown += Number(visible);
  }});
  document.getElementById('matchup-count').textContent = `${{shown}} matchup${{shown === 1 ? '' : 's'}}`;
}}
week.addEventListener('change', filterGames); team.addEventListener('change', filterGames);
</script></body></html>"""  # noqa: E501 — embedded HTML retains paragraph and CSS rule boundaries.
    (OUT / "index.html").write_text(html)


if __name__ == "__main__":
    data = build()
    render(data)
    print(
        json.dumps(
            {
                "artifact": str(OUT / "index.html"),
                "metrics": data["metrics"],
                "winner_accuracy": data["winners"],
            },
            indent=2,
        )
    )
