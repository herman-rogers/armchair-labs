#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.3", "scipy==1.18.1", "matplotlib==3.11.2"]
# ///
# ruff: noqa: E501 -- Embedded HTML prose is deliberately kept as complete paragraphs.
"""Disposable current-season league exploration. No model training or app writes.

Run with Python + numpy/scipy/matplotlib:
  uv run experiments/league_strength_explore.py --output /tmp/league-strength

Only actual points are read. Historical prediction rows use strictly earlier
weeks, conditional on the recorded starting lineup. Current-week depth uses
current availability; those tags are NEVER applied to historical rows.
"""

import argparse
import base64
import csv
import hashlib
import html
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
SKILL = {"QB", "RB", "WR", "TE"}
RESERVE = {"BE", "BN", "IR", "ER", ""}
UNAVAILABLE = {"OUT", "INJURY_RESERVE", "IR", "SUSPENSION", "SUSPENDED", "DOUBTFUL"}


def mean(xs):
    return float(np.mean(xs)) if len(xs) else None


def shape(values):
    """Compare the same eight offensive slots; K/DST stay separate."""
    # A recorded empty offensive slot contributes zero; missing player history
    # is handled separately and is never passed here as a zero forecast.
    assert 0 < len(values) <= 8
    values = sorted([*values, *([0.0] * (8 - len(values)))], reverse=True)
    positive = [max(v, 0) for v in values]
    total = sum(positive)
    shares = np.array(positive) / total if total else None
    return dict(
        skill_points=sum(values),
        top2_points=sum(values[:2]),
        remaining_points=sum(values[2:]),
        middle4_points=sum(values[2:6]),
        weakest3_points=sum(values[-3:]),
        median_starter=float(np.median(values)),
        top2_share=float(sum(shares[:2])) if shares is not None else None,
        effective_contributors=float(1 / sum(shares**2)) if shares is not None else None,
    )


def correlation(rows, feature, target):
    pairs = [
        (r[feature], r[target])
        for r in rows
        if r.get(feature) is not None and r.get(target) is not None
    ]
    if len(pairs) < 3 or len({a for a, _ in pairs}) < 2 or len({b for _, b in pairs}) < 2:
        return None
    return float(spearmanr(*zip(*pairs, strict=True)).statistic)


def assign(players, slots, details=False):
    """Max-score legal lineup. Missing coverage is unknown, never a free zero."""
    if len(players) < len(slots):
        return None
    costs = np.array(
        [
            [-p["estimate"] if p["position"] in slot.split("/") else 1e9 for p in players]
            for slot in slots
        ]
    )
    row, col = linear_sum_assignment(costs)
    if len(row) != len(slots) or any(costs[i, j] == 1e9 for i, j in zip(row, col, strict=True)):
        return None
    score = sum(players[j]["estimate"] for j in col)
    if details:
        return score, [(players[j], slots[i]) for i, j in zip(row, col, strict=True)]
    return score


def write_csv(path, rows):
    if rows:
        with path.open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def table(rows, columns):
    def cell(value):
        return (
            "—"
            if value is None
            else f"{value:.2f}"
            if isinstance(value, float)
            else html.escape(str(value))
        )

    return (
        "<table><thead><tr>"
        + "".join(f"<th>{html.escape(label)}</th>" for _, label in columns)
        + "</tr></thead><tbody>"
        + "".join(
            "<tr>" + "".join(f"<td>{cell(r.get(key))}</td>" for key, _ in columns) + "</tr>"
            for r in rows
        )
        + "</tbody></table>"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=ROOT / "data/outputs/league_snapshot.json")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data/research/league_strength_20261003"
    )
    args = parser.parse_args()
    raw = args.snapshot.read_bytes()
    snapshot = json.loads(raw)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    (out / "snapshot.json").write_bytes(raw)
    teams = {t["team_id"]: t for t in snapshot["teams"]}
    current = snapshot["week"]
    through = min(current - 1, snapshot["regular_season_weeks"])
    assert snapshot["season"] == 2026 and through == 3, (
        "This throwaway report is scoped to 2026 Weeks 1–3; use the saved snapshot.json "
        "to reproduce it. Refresh the narrative before exploring later weeks."
    )
    histories = defaultdict(dict)
    historical = []
    current_lineups = {}
    for game in snapshot["week_lineups"]:
        week = game["week"]
        for side in ("home", "away"):
            tid = game[f"{side}_team_id"]
            lineup = game[f"{side}_lineup"]
            if week == current:
                current_lineups[tid] = lineup
            if not 1 <= week <= through:
                continue
            scheduled = next((g for g in teams[tid]["schedule"] if g["week"] == week), None)
            if not scheduled or scheduled["outcome"] not in ("W", "L", "T"):
                continue
            for p in lineup:
                # Non-playing/zero weeks count, but known byes do not.
                if not p["on_bye"] and p["points"] is not None:
                    old = histories[p["espn_id"]].get(week)
                    assert old is None or old == p["points"], "Conflicting recorded player points"
                    histories[p["espn_id"]][week] = p["points"]
            starters = [p for p in lineup if p["slot"] not in RESERVE]
            skills = [p for p in starters if p["position"] in SKILL]
            assert len(skills) <= 8, "Too many offensive starters"
            score = game[f"{side}_score"]
            assert abs(sum(p["points"] for p in starters) - score) < 0.01, (
                "Lineup does not reconcile to team score"
            )
            historical.append(
                dict(
                    team_id=tid,
                    team=teams[tid]["team_name"],
                    week=week,
                    score=score,
                    outcome=scheduled["outcome"],
                    opponent=scheduled["opponent_team_id"],
                    starters=starters,
                    empty_skill_slots=8 - len(skills),
                    **shape([p["points"] for p in skills]),
                )
            )
    assert len(historical) == len(teams) * through, "Incomplete completed-week coverage"
    for row in historical:
        peers = [
            r["score"]
            for r in historical
            if r["week"] == row["week"] and r["team_id"] != row["team_id"]
        ]
        row["all_play_rate"] = sum(
            (row["score"] > v) + 0.5 * (row["score"] == v) for v in peers
        ) / len(peers)

    def estimate(pid, cutoff):
        scores = [v for w, v in histories[pid].items() if w < cutoff]
        return mean(scores), len(scores)

    # Forward comparisons: no same-week score used as an input. These are
    # conditional on actual chosen starters, not reconstructed pregame decisions.
    forward = []
    forward_exclusions = []
    for row in historical:
        if row["week"] == 1:
            continue
        predicted = [(p, *estimate(p["espn_id"], row["week"])) for p in row["starters"]]
        if any(value is None for _, value, _ in predicted):
            forward_exclusions.append(
                dict(
                    team=row["team"],
                    week=row["week"],
                    missing_players=", ".join(
                        p["player_display_name"] for p, value, _ in predicted if value is None
                    ),
                )
            )
            continue
        features = shape([value for p, value, _ in predicted if p["position"] in SKILL])
        total = sum(value for _, value, _ in predicted)
        previous = [
            r for r in historical if r["team_id"] == row["team_id"] and r["week"] < row["week"]
        ]
        forward.append(
            dict(
                team_id=row["team_id"],
                team=row["team"],
                week=row["week"],
                actual=row["score"],
                all_play_rate=row["all_play_rate"],
                prior_team_ppg=mean([r["score"] for r in previous]),
                starter_mean_total=total,
                error=row["score"] - total,
                absolute_error=abs(row["score"] - total),
                **features,
            )
        )
    signals = []
    for week in range(2, through + 1):
        sample = [r for r in forward if r["week"] == week]
        for feature in (
            "prior_team_ppg",
            "starter_mean_total",
            "middle4_points",
            "weakest3_points",
            "median_starter",
            "top2_share",
            "effective_contributors",
        ):
            signals.append(
                dict(
                    week=week,
                    feature=feature,
                    n=len(sample),
                    next_score_rank_correlation=correlation(sample, feature, "actual"),
                    forecast_error_rank_correlation=correlation(sample, feature, "error"),
                    absolute_error_rank_correlation=correlation(sample, feature, "absolute_error"),
                )
            )

    summary = []
    for tid, team in teams.items():
        rows = [r for r in historical if r["team_id"] == tid]
        contributors = defaultdict(float)
        top_names = set()
        for r in rows:
            skill = [p for p in r["starters"] if p["position"] in SKILL]
            top_names.update(
                p["espn_id"] for p in sorted(skill, key=lambda p: p["points"], reverse=True)[:2]
            )
            for p in skill:
                contributors[p["espn_id"]] += p["points"]
        positive = sorted([max(v, 0) for v in contributors.values()], reverse=True)
        summary.append(
            dict(
                team_id=tid,
                team=team["team_name"],
                record="–".join(str(sum(r["outcome"] == v for r in rows)) for v in ("W", "L", "T")),
                actual_ppg=mean([r["score"] for r in rows]),
                score_sd=float(np.std([r["score"] for r in rows], ddof=1))
                if len(rows) > 1
                else None,
                all_play_pct=100 * mean([r["all_play_rate"] for r in rows]),
                skill_ppg=mean([r["skill_points"] for r in rows]),
                middle4_ppg=mean([r["middle4_points"] for r in rows]),
                weakest3_ppg=mean([r["weakest3_points"] for r in rows]),
                weekly_top2_pct=100 * mean([r["top2_share"] for r in rows]),
                cumulative_top2_pct=100 * sum(positive[:2]) / sum(positive),
                unique_weekly_top2_players=len(top_names),
            )
        )
    summary.sort(key=lambda r: -r["actual_ppg"])

    # Current depth: replace one selected offensive starter with one eligible
    # bench player; allow slot reshuffling but retain every other starter.
    slots = [
        slot
        for slot, count in snapshot["roster_slots"].items()
        if set(slot.split("/")) <= SKILL
        for _ in range(count)
    ]
    assert len(slots) == 8
    now, depth, roster_rows = [], [], []
    for tid, team in teams.items():
        weekly = {p["espn_id"]: p for p in current_lineups.get(tid, [])}
        roster = []
        for p in snapshot["players"]:
            if p["owner_team_id"] != tid or p["position"] not in SKILL:
                continue
            value, n = estimate(p["espn_id"], current)
            weekly_row = weekly.get(p["espn_id"])
            reasons = []
            if p["lineup_slot"] in ("IR", "ER"):
                reasons.append("reserve slot")
            if p["injury_status"] in UNAVAILABLE:
                reasons.append(p["injury_status"])
            if weekly_row is None:
                reasons.append("weekly availability not captured")
            elif weekly_row["on_bye"]:
                reasons.append("bye")
            if value is None:
                reasons.append("no earlier observed score")
            row = dict(
                team_id=tid,
                team=team["team_name"],
                player=p["player_display_name"],
                player_id=p["espn_id"],
                position=p["position"],
                slot=p["lineup_slot"],
                injury_status=p["injury_status"],
                estimate=value,
                prior_weeks=n,
                usable=not reasons,
                reason="; ".join(reasons),
            )
            roster.append(row)
            roster_rows.append(row)
        selected = [p for p in roster if p["slot"] not in RESERVE]
        ready = [p for p in roster if p["usable"]]
        complete = len(selected) == len(slots) and all(p["usable"] for p in selected)
        if not complete:
            reasons = "; ".join(
                p["player"] + ": " + p["reason"] for p in selected if not p["usable"]
            )
            now.append(
                dict(
                    team_id=tid,
                    team=team["team_name"],
                    coverage=reasons or "Selected lineup incomplete",
                    best_legal_skill_points=assign(ready, slots),
                )
            )
            continue
        total = sum(p["estimate"] for p in selected)
        assert abs(assign(selected, slots) - total) < 1e-8
        bench = [p for p in ready if p not in selected]
        for starter in selected:
            remaining = [p for p in selected if p["player_id"] != starter["player_id"]]
            options = [(assign(remaining + [b], slots), b) for b in bench]
            options = [(score, b) for score, b in options if score is not None]
            best = max(options, key=lambda x: x[0]) if options else None
            replacement = best[1] if best else None
            plan = assign(remaining + [replacement], slots, details=True) if replacement else None
            moves = (
                "; ".join(
                    f"{p['player']}: {p['slot']} → {slot}"
                    for p, slot in plan[1]
                    if p["slot"] != slot
                )
                if plan
                else None
            )
            depth.append(
                dict(
                    team_id=tid,
                    team=team["team_name"],
                    starter=starter["player"],
                    position=starter["position"],
                    starter_estimate=starter["estimate"],
                    replacement=replacement["player"] if replacement else None,
                    replacement_status=replacement["injury_status"] if replacement else None,
                    replacement_estimate=replacement["estimate"] if replacement else None,
                    slot_moves=moves,
                    point_drop=total - best[0] if best else None,
                )
            )
        team_depth = [r for r in depth if r["team_id"] == tid]
        drops = [r["point_drop"] for r in team_depth if r["point_drop"] is not None]
        # Conservative sensitivity: do not count QUESTIONABLE replacements.
        robust_bench = [p for p in bench if p["injury_status"] != "QUESTIONABLE"]
        robust_drops = []
        for starter in selected:
            remaining = [p for p in selected if p["player_id"] != starter["player_id"]]
            options = [assign(remaining + [b], slots) for b in robust_bench]
            options = [x for x in options if x is not None]
            robust_drops.append(total - max(options) if options else None)
        positive_shares = [max(p["estimate"], 0) / total for p in selected]
        now.append(
            dict(
                team_id=tid,
                team=team["team_name"],
                coverage="Complete",
                **shape([p["estimate"] for p in selected]),
                best_legal_skill_points=assign(ready, slots),
                mean_replacement_drop=mean(drops) if len(drops) == len(slots) else None,
                worst_replacement_drop=max(drops) if len(drops) == len(slots) else None,
                replaceable_starters=len(drops),
                mean_drop_without_questionable=mean(robust_drops)
                if all(v is not None for v in robust_drops)
                else None,
                usable_bench=len(bench),
                excluded_players=sum(not p["usable"] for p in roster),
                # Assumes equally variable proportional shocks and independence;
                # a descriptive concentration measure, NOT estimated team variance.
                concentration_index=sum(v * v for v in positive_shares),
            )
        )

    benchmarks = []
    for week in range(2, through + 1):
        rows = [r for r in forward if r["week"] == week]
        for feature in ("prior_team_ppg", "starter_mean_total"):
            games = [
                (
                    r,
                    next(
                        (
                            o
                            for o in rows
                            if o["team_id"]
                            == next(
                                h["opponent"]
                                for h in historical
                                if h["team_id"] == r["team_id"] and h["week"] == week
                            )
                        ),
                        None,
                    ),
                )
                for r in rows
            ]
            games = [
                (a, b)
                for a, b in games
                if b
                and a["team_id"] < b["team_id"]
                and a["actual"] != b["actual"]
                and a[feature] != b[feature]
            ]
            benchmarks.append(
                dict(
                    week=week,
                    method=feature,
                    n=len(rows),
                    mae=mean([abs(r[feature] - r["actual"]) for r in rows]),
                    correct=sum(
                        (a[feature] > b[feature]) == (a["actual"] > b["actual"]) for a, b in games
                    ),
                    decisive_games=len(games),
                )
            )
    serial_history = [{k: v for k, v in r.items() if k != "starters"} for r in historical]
    for name, rows in [
        ("team_weeks", serial_history),
        ("team_summary", summary),
        ("prior_week_predictions", forward),
        ("lagged_signals", signals),
        ("current_starters", now),
        ("replacement_depth", depth),
        ("current_roster_coverage", roster_rows),
        ("baselines", benchmarks),
    ]:
        # Rows with incomplete coverage have fewer fields.
        keys = list(dict.fromkeys(k for r in rows for k in r))
        write_csv(out / f"{name}.csv", [{k: r.get(k) for k in keys} for r in rows])
    write_csv(out / "forward_exclusions.csv", forward_exclusions)
    result = dict(
        season=snapshot["season"],
        through_week=through,
        captured_at=snapshot["captured_at"],
        snapshot_sha256=hashlib.sha256(raw).hexdigest(),
        source=str(args.snapshot),
        my_team_id=snapshot["my_team_id"],
        summary=summary,
        current=now,
        depth=depth,
        signals=signals,
        baselines=benchmarks,
        forward_exclusions=forward_exclusions,
        limitations=[
            "Only three completed weeks: descriptive exploration, no fitted or validated strength model.",
            "Only this season and this league’s captured player observations; missing scores stay unknown.",
            "Historical comparisons condition on actual starters; earlier-week points only as inputs.",
            "Current means are transparent proxies, not NextGen ROS forecasts or calibrated weekly expectations.",
            "Offensive shape/depth use eight QB/RB/WR/TE starters; actual team results include K/DST.",
            "Scoring concentration does not establish independence, covariance, or a reliable floor.",
            "Depth covers one missing starter at a time; the same backup cannot cover simultaneous absences.",
            "Current OUT/IR/suspension/doubtful/bye excluded; QUESTIONABLE allowed and sensitivity reported.",
            "Observed non-bye zero scores remain zeros; availability and role changes can bias short-run means.",
            "Same-week middle-tier points are part of total points; their correlation is not predictive evidence.",
            "All earlier-score comparisons exclude The Price is Likely Right in Weeks 2 and 3 because one starter lacks earlier captured scores; metrics are not whole-league accuracy.",
        ],
    )
    (out / "analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    plot(out, result, historical)
    build_report(out, result)
    print(
        json.dumps(
            dict(
                output=str(out), summary=summary, current=now, signals=signals, baselines=benchmarks
            ),
            indent=2,
        )
    )


def plot(out, result, historical):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        r["team_id"]: ("#177e89" if r["team_id"] == result["my_team_id"] else "#94a3b8")
        for r in result["summary"]
    }
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), layout="constrained")
    fig.suptitle(
        f"League strength exploration · {result['season']} Weeks 1–{result['through_week']}\nHistorical results and current Week {result['through_week'] + 1} lineup proxies",
        fontsize=18,
    )
    rows = list(reversed(result["summary"]))
    names = [r["team"] for r in rows]
    axes[0, 0].barh(
        names, [r["actual_ppg"] for r in rows], color=[colors[r["team_id"]] for r in rows]
    )
    axes[0, 0].set(title="Actual team points per week · includes K/DST", xlabel="Points")
    for r in result["summary"]:
        rs = sorted(
            [h for h in historical if h["team_id"] == r["team_id"]], key=lambda h: h["week"]
        )
        axes[0, 1].plot(
            [h["week"] for h in rs],
            [h["score"] for h in rs],
            marker="o",
            color=colors[r["team_id"]],
            alpha=1 if r["team_id"] == result["my_team_id"] else 0.55,
            linewidth=3 if r["team_id"] == result["my_team_id"] else 1,
        )
    axes[0, 1].set(
        title="Actual scoring paths · highlighted: Just Say No",
        xlabel="Week",
        ylabel="Points",
        xticks=[1, 2, 3],
    )
    current = sorted(
        [r for r in result["current"] if r["coverage"] == "Complete"],
        key=lambda r: r["skill_points"],
    )
    names = [r["team"] for r in current]
    axes[1, 0].barh(
        names, [r["remaining_points"] for r in current], label="Other six starters", color="#83b6ad"
    )
    axes[1, 0].barh(
        names,
        [r["top2_points"] for r in current],
        left=[r["remaining_points"] for r in current],
        label="Top two starters",
        color="#d39b61",
    )
    axes[1, 0].set(
        title=f"Current starting offense · {len(current)}/10 complete lineups",
        xlabel="Earlier-score mean proxy · excludes K/DST",
    )
    axes[1, 0].legend(loc="lower right")
    for r in current:
        if r["mean_replacement_drop"] is not None:
            label = (
                "You"
                if r["team_id"] == result["my_team_id"]
                else r["team"].replace("Suck my holdie flappy folds", "Flappy folds")
            )
            axes[1, 1].scatter(
                r["skill_points"], r["mean_replacement_drop"], color=colors[r["team_id"]], s=70
            )
            axes[1, 1].annotate(
                label,
                (r["skill_points"], r["mean_replacement_drop"]),
                xytext=(4, 5),
                textcoords="offset points",
                fontsize=8,
            )
    axes[1, 1].set(
        title="Starter strength and usable depth · lower drop is better",
        xlabel="Starting offense point proxy",
        ylabel="Mean drop replacing one starter",
    )
    axes[1, 1].text(
        0.02,
        0.02,
        "Only teams with all 8 replacement scenarios covered",
        transform=axes[1, 1].transAxes,
        fontsize=8,
    )
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", alpha=0.15)
    fig.savefig(out / "team_strength.png", dpi=160)
    fig.savefig(out / "team_strength.svg")
    plt.close(fig)


def build_report(out, r):
    intro = (
        "<h1>League strength exploration</h1><p>2026 season only · completed Weeks 1–3 · current roster snapshot "
        + html.escape(r["captured_at"])
        + "</p>"
    )
    intro += "<p><b>Exploration only.</b> Historical production describes what happened. Current-lineup estimates use earlier observed player scoring averages, not published NextGen ranks. No production models or application code were changed.</p>"
    image = base64.b64encode((out / "team_strength.png").read_bytes()).decode()
    mine = next(x for x in r["current"] if x["team_id"] == r["my_team_id"])
    peer = next(x for x in r["current"] if x["team_id"] == 7)
    findings = "<h2>What this exploration tells us</h2><p>These dimensions provide useful descriptions of strength and fragility. The two forward comparisons do not establish a predictive advantage from balance.</p>"
    if mine["coverage"] == peer["coverage"] == "Complete":
        findings += f"<p>Just Say No’s current offense is more evenly distributed: its top two account for {100 * mine['top2_share']:.1f}% versus Da Foosball Team’s {100 * peer['top2_share']:.1f}%. However, the current middle four are almost equal ({mine['middle4_points']:.1f} vs {peer['middle4_points']:.1f}), and the total still favors Da Foosball ({mine['skill_points']:.1f} vs {peer['skill_points']:.1f}). These are this-season mean proxies, not ROS rank comparisons.</p>"
    findings += "<p>Historical offensive production does not yet establish a stronger middle for Just Say No: its weekly middle-four average was 54.0 versus Da Foosball’s 63.0. Current lineups differ from those historical lineups.</p><p>Depth reveals a different issue: Just Say No has a covered QB backup, while Da Foosball’s Caleb Williams is OUT. Just Say No’s biggest estimated replacement losses are CMC and Adams. Woody Marks has no earlier captured score in this snapshot, so his depth contribution is unknown rather than zero. McBride replacement scenarios shift Juwan to TE and add a bench player at FLEX.</p><p>The strongest next step is to retain these as diagnostics. Do not add an automatic scoring bonus for balance: median-starter strength has a higher next-score correlation than total expected points in Week 2, but lower in Week 3; lower concentration does not consistently predict better scoring.</p>"
    sections = [
        intro,
        findings,
        f'<img alt="League strength charts" src="data:image/png;base64,{image}">',
    ]
    sections += [
        "<h2>Actual team performance</h2><p>All-play compares each score with all nine other teams that week; ties count as half a win. Concentration columns use offensive starters only. These describe results, not predictive skill.</p>",
        table(
            r["summary"],
            [
                ("team", "Team"),
                ("record", "W–L–T"),
                ("actual_ppg", "Actual PPG"),
                ("all_play_pct", "All-play %"),
                ("score_sd", "Score SD"),
                ("middle4_ppg", "Middle four PPG"),
                ("weakest3_ppg", "Bottom three PPG"),
                ("weekly_top2_pct", "Average weekly top two %"),
                ("cumulative_top2_pct", "Same two players’ season %"),
            ],
        ),
    ]
    current = [
        dict(x, top2_pct=100 * x["top2_share"] if x.get("top2_share") is not None else None)
        for x in r["current"]
    ]
    sections += [
        "<h2>Current starting-lineup strength and scoring concentration</h2><p>Eight offensive starters. Players are ordered by earlier-score means: middle four = ranks 3–6; weakest three = ranks 6–8. Best legal lineup uses available current-roster players. This is a descriptive proxy, not a start/sit recommendation.</p>",
        table(
            current,
            [
                ("team", "Team"),
                ("coverage", "Coverage"),
                ("skill_points", "Selected starters"),
                ("best_legal_skill_points", "Best legal lineup"),
                ("middle4_points", "Middle four"),
                ("weakest3_points", "Bottom three"),
                ("top2_pct", "Top two %"),
                ("effective_contributors", "Effective contributors"),
            ],
        ),
    ]
    sections += [
        "<h2>Usable depth</h2><p>Drop in offensive points if one selected starter is removed and replaced by the best legal bench option, retaining all other starters. Negative means the proxy prefers that bench player. Each scenario is independent; backups are not counted twice within a lineup. Unknown or unavailable players cannot fill a slot. QUESTIONABLE players are included, with a separate sensitivity excluding them.</p>",
        table(
            current,
            [
                ("team", "Team"),
                ("usable_bench", "Usable skill backups"),
                ("replaceable_starters", "Covered starter absences / 8"),
                ("mean_replacement_drop", "Mean point drop"),
                ("worst_replacement_drop", "Worst point drop"),
                ("mean_drop_without_questionable", "Mean drop excluding questionable"),
            ],
        ),
    ]
    for tid in [r["my_team_id"], 7]:
        rows = [x for x in r["depth"] if x["team_id"] == tid]
        if rows:
            sections += [
                "<h3>" + html.escape(rows[0]["team"]) + "</h3>",
                table(
                    rows,
                    [
                        ("starter", "Missing starter"),
                        ("starter_estimate", "Point proxy"),
                        ("replacement", "Bench player entering lineup"),
                        ("replacement_status", "Status"),
                        ("replacement_estimate", "Backup proxy"),
                        ("slot_moves", "Slot changes"),
                        ("point_drop", "Point drop"),
                    ],
                ),
            ]
    sections += [
        "<h3>Prior-week comparison exclusions</h3>",
        table(
            r["forward_exclusions"],
            [("week", "Week"), ("team", "Team"), ("missing_players", "No earlier captured score")],
        ),
    ]
    sections += [
        "<h2>Does it predict the next week?</h2><p>Only two forecast origins and ten teams per week. Inputs stop before the target week; lineups are the actual starters recorded afterward. No fitted balance bonus, probability, or statistical significance claim. Correlations range from −1 to +1. Error = actual minus the sum of player means. An association with error is exploratory, not proof of improvement beyond the baseline.</p>",
        table(
            r["signals"],
            [
                ("week", "Target week"),
                ("feature", "Earlier-week feature"),
                ("n", "Teams"),
                ("next_score_rank_correlation", "Correlation with next score"),
                ("forecast_error_rank_correlation", "Correlation with forecast error"),
                ("absolute_error_rank_correlation", "Correlation with absolute error"),
            ],
        ),
        table(
            r["baselines"],
            [
                ("week", "Week"),
                ("method", "Simple baseline"),
                ("mae", "Team point error"),
                ("correct", "Correct picks"),
                ("decisive_games", "Decisive picks"),
            ],
        ),
    ]
    sections += [
        "<h2>Limits</h2><ul>"
        + "".join("<li>" + html.escape(x) + "</li>" for x in r["limitations"])
        + "</ul>"
    ]
    css = "body{font:16px system-ui;color:#22343a;background:#f5f5f0;margin:32px auto;max-width:1400px;padding:0 24px}img{width:100%}table{border-collapse:collapse;width:100%;font-size:13px;margin:20px 0}th,td{padding:9px;text-align:right;border-bottom:1px solid #d5d9d6}th:first-child,td:first-child{text-align:left}th{background:#e4eae7}h2{margin-top:36px}p,li{line-height:1.55}"
    (out / "report.html").write_text(
        '<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>League strength exploration</title><style>'
        + css
        + "</style></head><body>"
        + "".join(sections)
        + "</body></html>"
    )


if __name__ == "__main__":
    main()
