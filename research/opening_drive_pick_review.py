"""Review frozen dashboard v2/v3 picks, without fitting or changing forecasts.

Run: .venv/bin/python research/opening_drive_pick_review.py
Inputs are the snapshot, observation rows and scoring-trends responses frozen in
data/outputs/opening_drive_pick_review_20261013. Week 5 player scores do not
reconcile, so only team-level results are graded for that week.
"""

import csv
import json
from hashlib import sha256
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/outputs/opening_drive_pick_review_20261013"
VERSIONS = ("v2", "v3")


def write_csv(name, rows):
    with (OUT / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metrics(rows, prediction, actual):
    error = np.array([r[prediction] - r[actual] for r in rows])
    return dict(
        n=len(rows),
        mae=float(np.abs(error).mean()),
        rmse=float(np.sqrt(np.mean(error**2))),
        bias=float(error.mean()),
    )


def main():
    inputs = json.loads((OUT / "inputs.json").read_text())
    snapshot = inputs["snapshot"]
    versions = {v: json.loads((OUT / f"{v}.json").read_text()) for v in VERSIONS}
    lookup = {
        v: {(t["team_id"], w["week"]): w for t in d["teams"] for w in t["weeks"]}
        for v, d in versions.items()
    }
    names = {t["team_id"]: t["team_name"] for t in snapshot["teams"]}
    schedule = {(t["team_id"], e["week"]): e for t in snapshot["teams"] for e in t["schedule"]}
    actuals = {(r["player_id"], r["week"]): r for r in inputs["actuals"]}
    assert len(actuals) == len(inputs["actuals"])
    teams, players, games = [], [], []
    for game in snapshot["week_lineups"]:
        week = game["week"]
        if week > min(inputs["cutoff"]["through_week"], snapshot["week"] - 1):
            continue
        pair = []
        for side in ("home", "away"):
            tid = game[f"{side}_team_id"]
            a, b = [lookup[v][tid, week] for v in VERSIONS]
            assert a["actual"] == b["actual"] == schedule[tid, week]["score"]
            assert a["status"] == b["status"] == "Completed"
            observed = {p["espn_id"]: p for p in game[f"{side}_lineup"]}
            pa, pb = [{p["espn_id"]: p for p in r["starters"]} for r in (a, b)]
            assert pa.keys() == pb.keys()
            row = dict(
                week=week,
                team_id=tid,
                team=names[tid],
                actual=a["actual"],
                v2=a["predicted"],
                v3=b["predicted"],
                reconciled=a["reconciled"],
                v2_origin=a["origin_week"],
                v3_origin=b["origin_week"],
            )
            for version, forecast in zip(VERSIONS, (a, b), strict=True):
                assert forecast["origin_week"] < week
                assert all(p["prediction"] is not None for p in forecast["starters"])
                assert np.isclose(sum(p["prediction"] for p in forecast["starters"]), row[version])
            special = [pid for pid, p in pa.items() if p["position"] in {"K", "DST", "D/ST"}]
            assert all(pa[pid]["prediction"] == pb[pid]["prediction"] for pid in special)
            row["special_prediction"] = sum(pa[pid]["prediction"] for pid in special)
            row["special_actual"] = (
                sum(observed[pid]["points"] for pid in special) if row["reconciled"] else None
            )
            for pid, p in pa.items():
                player_id = inputs["identities"].get(str(pid))
                obs = actuals.get((player_id, week))
                actual = observed[pid]["points"] if row["reconciled"] else None
                model_actual = obs["model_points"] if obs else None
                if (
                    row["reconciled"]
                    and model_actual is None
                    and p["position"] in {"QB", "RB", "WR", "TE"}
                ):
                    assert actual == 0, (week, p["name"], actual)
                    model_actual = 0.0
                players.append(
                    dict(
                        week=week,
                        team_id=tid,
                        team=names[tid],
                        player_id=player_id,
                        name=p["name"],
                        position=p["position"],
                        v2=p["prediction"],
                        v3=pb[pid]["prediction"],
                        delta=pb[pid]["prediction"] - p["prediction"],
                        actual_espn=actual,
                        actual_model=model_actual if row["reconciled"] else None,
                        reconciled=row["reconciled"],
                    )
                )
            teams.append(row)
            pair.append(row)
        home, away = pair
        margin = home["actual"] - away["actual"]
        winner = home if margin > 0 else away if margin < 0 else None
        assert (schedule[home["team_id"], week]["outcome"] == "T") == (winner is None)
        row = dict(
            week=week,
            home=home["team"],
            away=away["team"],
            home_id=home["team_id"],
            away_id=away["team_id"],
            home_actual=home["actual"],
            away_actual=away["actual"],
            actual_margin=margin,
            winner=winner["team"] if winner else "Tie",
            reconciled=home["reconciled"] and away["reconciled"],
        )
        for v in VERSIONS:
            m = home[v] - away[v]
            row[f"{v}_margin"] = m
            row[f"{v}_pick"] = home["team"] if m > 0 else away["team"] if m < 0 else None
            row[f"{v}_correct"] = bool(m * margin > 0) if winner and m else None
            row[f"{v}_offense_margin"] = m - home["special_prediction"] + away["special_prediction"]
            if row["reconciled"]:
                row[f"{v}_oracle_special_margin"] = (
                    row[f"{v}_offense_margin"] + home["special_actual"] - away["special_actual"]
                )
            else:
                row[f"{v}_oracle_special_margin"] = None
        row["disagree"] = row["v2_pick"] != row["v3_pick"]
        games.append(row)
    teams.sort(key=lambda r: (r["week"], r["team_id"]))
    games.sort(key=lambda r: (r["week"], r["home_id"]))
    summary = []
    for week in [*sorted({r["week"] for r in teams}), 0]:
        ts = [r for r in teams if not week or r["week"] == week]
        gs = [r for r in games if (not week or r["week"] == week) and r["winner"] != "Tie"]
        for v in VERSIONS:
            summary.append(
                dict(
                    week=week or "All",
                    version=v,
                    correct=sum(r[f"{v}_correct"] for r in gs),
                    games=len(gs),
                    **metrics(ts, v, "actual"),
                    margin_mae=metrics(gs, f"{v}_margin", "actual_margin")["mae"],
                )
            )
    position_summary = []
    for pos in ("QB", "RB", "WR", "TE", "K", "D/ST"):
        rows = [r for r in players if r["reconciled"] and r["position"] == pos]
        if not rows:
            continue
        for v in VERSIONS:
            position_summary.append(
                dict(position=pos, version=v, **metrics(rows, v, "actual_espn"))
            )
    paired = [r for r in games if r["winner"] != "Tie"]
    disagree = [r for r in paired if r["disagree"]]
    centered_rmse = {}
    for v in VERSIONS:
        errors = []
        for week in sorted({r["week"] for r in teams}):
            e = np.array([r[v] - r["actual"] for r in teams if r["week"] == week])
            errors.extend(e - e.mean())
        centered_rmse[v] = float(np.sqrt(np.mean(np.square(errors))))
    import polars as pl

    panel_path = (
        ROOT / "experiments/future_player_lab/runs/opening_drive_v3_20261011_002/panel.parquet"
    )
    panel = pl.read_parquet(panel_path).filter((pl.col("season") == 2026) & (pl.col("week") <= 5))
    result = dict(
        summary=summary,
        positions=position_summary,
        weekly_bias_removed_rmse=centered_rmse,
        injury_reports=dict(
            rows=panel.height,
            known=int(panel["injury_report_known"].sum()),
            panel_sha256=sha256(panel_path.read_bytes()).hexdigest(),
        ),
        disagreements={
            "n": len(disagree),
            **{v: sum(r[f"{v}_correct"] for r in disagree) for v in VERSIONS},
        },
        weeks_1_4={
            v: dict(
                correct=sum(r[f"{v}_correct"] for r in paired if r["week"] <= 4),
                **metrics([r for r in teams if r["week"] <= 4], v, "actual"),
            )
            for v in VERSIONS
        },
        manifest_hashes={v: d["manifest_sha256"] for v, d in versions.items()},
        input_hashes={
            f: sha256((OUT / f).read_bytes()).hexdigest()
            for f in ("inputs.json", "v2.json", "v3.json")
        },
        limitations=[
            "Retrospective dashboard reconstruction, not verified pregame performance.",
            "Two official ties excluded from winner accuracy.",
            "Week 5 team totals are official, but all ten recorded player breakdowns fail "
            "reconciliation; player attribution excludes Week 5.",
            "Oracle K/DST columns are hindsight diagnostics, never usable forecasts.",
        ],
    )
    for name, rows in (
        ("matchups", games),
        ("teams", teams),
        ("players", players),
        ("summary", summary),
        ("positions", position_summary),
    ):
        write_csv(f"{name}.csv", rows)
    (OUT / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    render(summary)
    print(json.dumps(result, indent=2))


def render(summary):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7))
    colors = {"v2": "#475569", "v3": "#167c80"}
    for v in VERSIONS:
        rows = [r for r in summary if r["version"] == v and r["week"] != "All"]
        x = np.array([r["week"] for r in rows])
        bars = axes[0].bar(
            x + (-0.18 if v == "v2" else 0.18),
            [r["correct"] / r["games"] * 100 for r in rows],
            width=0.34,
            label=v,
            color=colors[v],
        )
        axes[0].bar_label(bars, labels=[f"{r['correct']}/{r['games']}" for r in rows], padding=4)
        axes[1].plot(x, [r["mae"] for r in rows], marker="o", label=v, color=colors[v])
    axes[0].set(title="Winner picks: v2 leads overall", ylabel="Correct picks (%)", ylim=(0, 115))
    axes[1].set(
        title="Team scores: v3 has lower error every week", ylabel="Mean absolute error (points)"
    )
    for ax in axes:
        ax.set(xticks=range(1, 6), xlabel="Week")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False)
    fig.suptitle("Opening Drive v2 vs v3 · Sweaty Plays · 2026", fontsize=15)
    fig.text(
        0.5,
        0.01,
        "Reconstructed picks; two ties excluded. Week 5 uses official team totals; "
        "player breakdowns do not reconcile.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(OUT / "comparison.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
