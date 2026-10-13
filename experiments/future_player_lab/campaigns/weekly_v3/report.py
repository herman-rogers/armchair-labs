"""Standalone, reproducible v2/v3 comparison with paired player-week evidence."""
# ruff: noqa: E501 -- Inline standalone HTML, CSS, and JavaScript templates.

from __future__ import annotations

import argparse
import hashlib
import html
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .modeling import score_predictions

ROOT = Path(__file__).resolve().parents[4]


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def table(rows, columns=None):
    if not rows:
        return '<p class="muted">No supported observations.</p>'
    columns = columns or [(k, k.replace("_", " ")) for k in rows[0]]

    def cell(value):
        if value is None or isinstance(value, float) and not np.isfinite(value):
            return "—"
        if isinstance(value, float):
            return f"{value:,.3f}"
        if isinstance(value, (dict, list)):
            return html.escape(json.dumps(clean(value)))
        return html.escape(str(value))

    return (
        '<div class="scroll"><table><thead><tr>'
        + "".join(f"<th>{html.escape(label)}</th>" for _, label in columns)
        + "</tr></thead><tbody>"
        + "".join(
            "<tr>" + "".join(f"<td>{cell(row.get(key))}</td>" for key, _ in columns) + "</tr>"
            for row in rows
        )
        + "</tbody></table></div>"
    )


def paired_comparison(selected):
    eligible = selected[
        selected.season.between(2021, 2025)
        & selected.v2_exact_available.eq(1)
        & selected.target_points.notna()
    ].copy()
    rows, yearly = [], []
    rng = np.random.default_rng(20261011)
    for pos in ["ALL", "QB", "RB", "WR", "TE"]:
        group = eligible if pos == "ALL" else eligible[eligible.position.eq(pos)]
        if group.empty:
            continue
        score = score_predictions(group)
        a = group.target_points.to_numpy()
        v2e = group.v2_weekly_mean.to_numpy() - a
        v3e = group.prediction.to_numpy() - a
        annual = []
        for season, year in group.groupby("season"):
            e2 = year.v2_weekly_mean.to_numpy() - year.target_points.to_numpy()
            e3 = year.prediction.to_numpy() - year.target_points.to_numpy()
            annual.append(float(np.mean(e2**2 - e3**2)))
            yearly.append(
                dict(
                    position=pos,
                    season=int(season),
                    n=len(year),
                    v2_rmse=float(np.sqrt(np.mean(e2**2))),
                    v3_rmse=float(np.sqrt(np.mean(e3**2))),
                    mse_reduction_percent=float(100 * (1 - np.mean(e3**2) / np.mean(e2**2))),
                )
            )
        bootstrap = rng.choice(annual, size=(10000, len(annual))).mean(axis=1)
        rows.append(
            dict(
                position=pos,
                n=len(group),
                v2_rmse=float(np.sqrt(np.mean(v2e**2))),
                v3_rmse=score["rmse"],
                v2_mae=float(np.abs(v2e).mean()),
                v3_mae=score["mae"],
                mse_reduction_percent=float(100 * (1 - np.mean(v3e**2) / np.mean(v2e**2))),
                v3_coverage80=score.get("coverage_80"),
                v3_width80=score.get("width_80"),
                mean_annual_mse_gain=float(np.mean(annual)),
                annual_block_gain_low=float(np.quantile(bootstrap, 0.025)),
                annual_block_gain_high=float(np.quantile(bootstrap, 0.975)),
            )
        )
    return rows, yearly


def build_report(run_dir, output_dir):
    run, output = Path(run_dir), Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    selected = pd.read_parquet(run / "selected_predictions.parquet")
    comparisons, annual = paired_comparison(selected)
    paired = selected[
        selected.season.between(2021, 2025)
        & selected.v2_exact_available.eq(1)
        & selected.target_points.notna()
    ].copy()
    paired["v2_score_group"] = pd.cut(
        paired.v2_weekly_mean,
        [-np.inf, 5, 10, np.inf],
        labels=["below 5", "5 to 10", "above 10"],
        right=False,
    )
    utility = []
    for (position, group_name), group in paired.groupby(
        ["position", "v2_score_group"], observed=True
    ):
        e2 = group.v2_weekly_mean - group.target_points
        e3 = group.prediction - group.target_points
        utility.append(
            dict(
                position=position,
                v2_score_group=str(group_name),
                n=len(group),
                v2_mae=float(e2.abs().mean()),
                v3_mae=float(e3.abs().mean()),
                v2_rmse=float(np.sqrt((e2**2).mean())),
                v3_rmse=float(np.sqrt((e3**2).mean())),
            )
        )
    means = pd.read_csv(run / "mean_scores.csv")
    distributions = pd.read_csv(run / "distribution_scores.csv")
    cohort = pd.read_csv(run / "cohort_scores.csv")
    policy = read_json(run / "selection.json", {})
    dataset = read_json(run / "dataset.json", {})
    metadata = read_json(run / "candidate_metadata.json", {})
    lineup = read_json(run / "lineups/results.json", {})
    audit = read_json(run / "audit.json", {"status": "pending"})
    replay = read_json(run / "replay.json", [])
    protocol = read_json(
        ROOT / "experiments/future_player_lab/campaigns/configs/opening_drive_v3.json", {}
    )
    recent = selected[selected.season.eq(2026)]
    current = recent[recent.week.eq(recent.week.max())].sort_values("prediction", ascending=False)
    fields = [
        c
        for c in [
            "player_id",
            "player_name",
            "position",
            "team",
            "week",
            "prediction",
            "v2_weekly_mean",
            "std",
            "q10",
            "q50",
            "q90",
            "player_history_count",
            "mean_method",
            "distribution_method",
        ]
        if c in current
    ]
    players = clean(current[fields].to_dict("records"))
    scores = []
    for kind, frame in [("mean", means), ("distribution", distributions)]:
        evaluated = frame[frame.season.between(2021, 2025)]
        cols = [
            c
            for c in ["rmse", "mae", "crps_quantile", "coverage_80", "width_80", "brier_over_20"]
            if c in frame
        ]
        for (pos, method), group in evaluated.groupby(["position", "method"]):
            scores.append(
                dict(
                    kind=kind,
                    position=pos,
                    method=method,
                    n=int(group.n.sum()),
                    **{c: float(group[c].mean()) for c in cols},
                )
            )
    receipt = dict(
        generated_at=datetime.now(UTC).isoformat(),
        comparisons=comparisons,
        annual=annual,
        paired_by_v2_scoring_level=utility,
        policies=policy,
        candidates=scores,
        cohorts=clean(cohort.to_dict("records")),
        dataset=dataset,
        experiment=metadata,
        lineups=lineup,
        audit=audit,
        replay=replay,
        publication=read_json(run / "publication.json", {}),
        sealed_replay=read_json(run / "sealed_replay.json", {}),
        refresh_validation=read_json(run / "rescore_e2e.json", {}),
        published_bundle_audit=read_json(run / "published_bundle_audit.json", {}),
        current_players=players,
        sources=protocol.get("research_sources", []),
        input_hashes={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                run / "selected_predictions.parquet",
                run / "mean_scores.csv",
                run / "distribution_scores.csv",
                run / "dataset.json",
            ]
        },
    )
    (output / "comparison.json").write_text(json.dumps(clean(receipt), indent=2, allow_nan=False))
    pd.DataFrame(comparisons).to_csv(output / "paired_v2_v3.csv", index=False)
    pd.DataFrame(scores).to_csv(output / "candidate_scores.csv", index=False)
    current[fields].to_csv(output / "current_players.csv", index=False)
    selected.to_parquet(output / "selected_predictions.parquet", index=False)
    plots = []
    positions = [r for r in comparisons if r["position"] != "ALL"]
    fig = go.Figure()
    for name, key, color in [
        ("v2 horizon allocation", "v2_rmse", "#8998ae"),
        ("v3 weekly", "v3_rmse", "#087e8b"),
    ]:
        fig.add_bar(
            name=name,
            x=[r["position"] for r in positions],
            y=[r[key] for r in positions],
            marker_color=color,
        )
    fig.update_layout(
        title="Weekly point error on identical eligible observations",
        yaxis_title="RMSE · points",
        barmode="group",
    )
    plots.append(fig)
    fig = go.Figure()
    for pos in ["QB", "RB", "WR", "TE"]:
        rows = [r for r in annual if r["position"] == pos]
        fig.add_scatter(
            name=pos,
            x=[r["season"] for r in rows],
            y=[r["mse_reduction_percent"] for r in rows],
            mode="lines+markers",
        )
    fig.add_hline(y=0, line_dash="dot")
    fig.update_layout(
        title="Does improvement repeat across seasons?",
        yaxis_title="MSE reduction vs v2 · percent",
        xaxis=dict(dtick=1),
    )
    plots.append(fig)
    fig = make_subplots(
        rows=1, cols=2, subplot_titles=["80% interval coverage", "80% interval width"]
    )
    for pos in ["QB", "RB", "WR", "TE"]:
        rows = [r for r in scores if r["kind"] == "distribution" and r["position"] == pos]
        fig.add_scatter(
            name=pos,
            x=[r["method"] for r in rows],
            y=[r.get("coverage_80") for r in rows],
            mode="markers",
            row=1,
            col=1,
        )
        fig.add_scatter(
            name=pos,
            showlegend=False,
            x=[r["method"] for r in rows],
            y=[r.get("width_80") for r in rows],
            mode="markers",
            row=1,
            col=2,
        )
    fig.add_hline(y=0.8, line_dash="dot", row=1, col=1)
    fig.update_layout(
        title="Uncertainty methods: coverage and width must be read together", height=470
    )
    plots.append(fig)
    chart_html = []
    for index, fig in enumerate(plots):
        fig.update_layout(
            template="plotly_white",
            font=dict(family="system-ui"),
            margin=dict(t=65, b=90, l=55, r=30),
        )
        chart_html.append(
            fig.to_html(
                full_html=False,
                include_plotlyjs=index == 0,
                config={"displaylogo": False, "responsive": True},
            )
        )
    sources = "".join(
        f'<li><a href="{html.escape(s["url"])}">{html.escape(s["title"])}</a> — {html.escape(s["use"])}</li>'
        for s in protocol.get("research_sources", [])
    )
    notes = metadata.get("limitations", [])
    lineups_summary = lineup.get("league", {})

    def details(title, value):
        return f"<details><summary>{html.escape(title)}</summary><pre>{html.escape(json.dumps(clean(value), indent=2))}</pre></details>"

    payload = json.dumps(players, allow_nan=False).replace("<", "\\u003c")
    page = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Opening Drive v3 compared with v2</title><style>
:root{color-scheme:light;font-family:Inter,system-ui,sans-serif;color:#152738;background:#f0f3f5}body{margin:0}main{max-width:1240px;margin:auto;padding:38px 24px}h1{font-size:42px;letter-spacing:-1.5px;margin:12px 0}h2{margin-top:0}p{line-height:1.65;max-width:1000px}.eyebrow{color:#087e8b;text-transform:uppercase;letter-spacing:2px;font-size:12px;font-weight:750}.intro{font-size:19px;color:#445466}section{background:white;border:1px solid #dce3e7;border-radius:12px;padding:25px;margin:24px 0}.notice{border-left:4px solid #cb8a28;background:#fff8e9;padding:15px 20px}.scroll{overflow:auto}table{width:100%;border-collapse:collapse;font-size:13px}th,td{text-align:right;padding:11px 12px;border-bottom:1px solid #e5e9ed;white-space:nowrap}th:first-child,td:first-child{text-align:left}th{background:#f6f8fa;position:sticky;top:0}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}a{color:#087e8b}nav{display:flex;flex-wrap:wrap;gap:15px}select,input{padding:10px;border:1px solid #becad3;border-radius:6px;font:inherit;margin-right:10px}summary{cursor:pointer;padding:12px;font-weight:650}.muted{color:#657387}.controls{margin:15px 0}li{margin:9px 0;line-height:1.5}@media(max-width:650px){h1{font-size:30px}main{padding:20px 10px}section{padding:16px}}
</style></head><body><main><div class="eyebrow">Armchair Labs · research and model comparison</div><h1>Opening Drive v3 compared with v2</h1>
<p class="intro">Weekly expectations, individual variability, conditional ranges, and full-lineup aggregation. Original v2 models are preserved; v3 adds separately trained weekly models.</p>
<nav><a href="#comparison">v2 vs v3</a><a href="#uncertainty">Uncertainty research</a><a href="#players">Current players</a><a href="#lineups">Lineups and K/DST</a><a href="#evidence">Audit and downloads</a></nav>
<p class="notice">Historical development evidence, not an untouched prospective test. Current 2026 forecasts reconstruct earlier cutoffs from later captures. Week 5 is already underway. Wider intervals alone do not establish a better forecast.</p>
"""
    overall = next(r for r in comparisons if r["position"] == "ALL")
    coverage_text = ", ".join(
        f"{r['position']} {100 * r['v3_coverage80']:.1f}%"
        for r in comparisons
        if r["position"] != "ALL"
    )
    synthetic_coverage = lineup.get("synthetic", {}).get("v3_joint", {}).get("coverage80")
    team_coverage_text = (
        f"{100 * synthetic_coverage:.1f}%" if synthetic_coverage is not None else "pending"
    )
    page += f'<section><h2>What the evidence supports</h2><p><strong>Weekly point forecasts improve across all four positions.</strong> Across {overall["n"]:,} paired player-weeks, RMSE falls from {overall["v2_rmse"]:.2f} to {overall["v3_rmse"]:.2f} points and MAE from {overall["v2_mae"]:.2f} to {overall["v3_mae"]:.2f}. This comparison includes both refreshed weekly information and new models; it does not isolate the algorithm alone.</p><p>Train shared models on historical players, then condition each forecast on that player’s current information. The evidence favors scoring-level residual pools for RB/WR/TE and quantile trees for QB. A separate personal-volatility estimate did not beat the scoring-level pools. Variance describes dispersion around the mean: it is not an extra amount of points to add to the mean.</p><p class="notice"><strong>Ranges are research estimates.</strong> Nominal 80% player coverage: {coverage_text}. Some high-projection cohorts fare substantially worse; see the workload table below. Artificial-lineup coverage is {team_coverage_text}. K/DST uncertainty remains provisional. These ranges should not be read as verified 80% guarantees.</p></section>'
    page += '<section id="comparison"><h2>Paired weekly comparison</h2><p>2021–2025, identical player-weeks with an exact frozen v2 forecast. v2 predicts horizon points allocated per scheduled game. v3 predicts the next week. Positive MSE reduction favors v3.</p>'
    page += table(
        comparisons,
        [
            ("position", "Position"),
            ("n", "Player-weeks"),
            ("v2_rmse", "v2 RMSE"),
            ("v3_rmse", "v3 RMSE"),
            ("v2_mae", "v2 MAE"),
            ("v3_mae", "v3 MAE"),
            ("mse_reduction_percent", "MSE reduction %"),
            ("v3_coverage80", "80% coverage"),
            ("v3_width80", "80% width"),
        ],
    )
    page += chart_html[0] + chart_html[1]
    page += "<details><summary>Paired point errors by the original v2 scoring level</summary><p>This separates low-projection depth players from higher-projection players using the original baseline, so the new model does not define its own comparison cohort. Artificial lineups below provide a second check on fantasy-relevant starters.</p>"
    page += (
        table(
            utility,
            [
                ("position", "Position"),
                ("v2_score_group", "v2 weekly projection"),
                ("n", "Player-weeks"),
                ("v2_mae", "v2 MAE"),
                ("v3_mae", "v3 MAE"),
                ("v2_rmse", "v2 RMSE"),
                ("v3_rmse", "v3 RMSE"),
            ],
        )
        + "</details>"
    )
    page += '<p class="muted">Uncertainty of the mean improvement: season-block bootstrap, five seasons; descriptive only and does not undo prior reuse of these years or comparison of many candidates.</p>'
    page += (
        table(
            comparisons,
            [
                ("position", "Position"),
                ("mean_annual_mse_gain", "Annual mean MSE gain"),
                ("annual_block_gain_low", "95% lower"),
                ("annual_block_gain_high", "95% upper"),
            ],
        )
        + "</section>"
    )
    page += '<section id="uncertainty"><h2>What predicts the spread?</h2><p>The experiment compares shared residual distributions, scoring-level pools, personal residual variance shrunk toward a population baseline, conditional scale trees with and without personal variability inputs, and quantile trees. Each player receives an individual distribution, while training shares information across historical players.</p>'
    page += "<p>“Test whether the spread changes predictably” means forecasting uncertainty before a game, then checking whether the weeks assigned wider ranges actually have larger errors. The spread-group diagnostic below compares predicted standard deviation with realized, centered errors and reports bias separately. A player having a volatile past is not sufficient evidence that a separate personal variance model improves the next forecast.</p>"
    page += chart_html[2]
    page += "<p>Candidate scores below average each evaluation season equally. The selected policies were chosen using 2018–2020, rather than choosing the best row after seeing these evaluation results. Lower RMSE, MAE and distribution scores are better. CRPS is approximated by integrating quantile losses; 1% and 99% tails are bounded.</p>"
    page += table([dict(position=p, **s) for p, s in policy.items()])
    selected_cohorts = pd.concat(
        [
            cohort[
                cohort.position.eq(pos)
                & cohort.method.eq(spec["distribution_method"])
                & cohort.cohort.isin(
                    ["projected_under_5", "projected_5_to_10", "projected_10_plus"]
                )
            ]
            for pos, spec in policy.items()
        ]
    )
    page += "<h3>Range reliability by projected workload</h3><p>Coverage should approach 0.80 in each group. Overall coverage can conceal poor ranges for high-projection players. These diagnostics use all eligible historical evaluation rows, while the headline comparison requires exact v2 forecasts.</p>"
    page += table(
        clean(selected_cohorts.to_dict("records")),
        [
            ("position", "Position"),
            ("cohort", "Projected points"),
            ("n", "Player-weeks"),
            ("coverage_80", "80% coverage"),
            ("width_80", "Width"),
            ("bias", "Mean error"),
        ],
    )
    evidence = []
    for pos in ["QB", "RB", "WR", "TE"]:
        dist = {
            r["method"]: r for r in scores if r["kind"] == "distribution" and r["position"] == pos
        }
        row = {"position": pos}
        for key, left, right in [
            ("personal_gain", "mean_bin", "player_shrinkage"),
            ("conditional_gain", "mean_bin", "conditional_scale"),
            ("personal_feature_gain", "conditional_no_variability", "conditional_scale"),
        ]:
            row[key] = (
                dist[left]["crps_quantile"] - dist[right]["crps_quantile"]
                if left in dist and right in dist
                else None
            )
        evidence.append(row)
    page += "<h3>Personal history versus changing weekly context</h3><p>Positive differences below mean a lower distribution error for the more individualized method. These are evaluation diagnostics, not a second policy-selection step.</p>"
    page += table(
        evidence,
        [
            ("position", "Position"),
            ("personal_gain", "Personal shrinkage vs scoring-level pool"),
            ("conditional_gain", "Conditional scale vs scoring-level pool"),
            ("personal_feature_gain", "Adding personal variability to scale tree"),
        ],
    )
    for pos in ["QB", "RB", "WR", "TE"]:
        page += f"<details><summary>{pos} candidate comparison</summary>"
        page += (
            table(
                [r for r in scores if r["position"] == pos],
                [
                    ("kind", "Component"),
                    ("method", "Method"),
                    ("rmse", "RMSE"),
                    ("mae", "MAE"),
                    ("crps_quantile", "Approx. CRPS"),
                    ("coverage_80", "80% coverage"),
                    ("width_80", "Width"),
                    ("brier_over_20", "Brier above 20"),
                ],
            )
            + "</details>"
        )
    spread_path = run / "spread_calibration.csv"
    if spread_path.exists():
        spread = pd.read_csv(spread_path)
        shown = pd.concat(
            [
                spread[spread.position.eq(pos) & spread.method.eq(spec["distribution_method"])]
                for pos, spec in policy.items()
            ]
        )
        page += "<details><summary>Does predicted spread track realized spread?</summary><p>Player-weeks are grouped by predicted standard deviation. A useful spread model should distinguish narrow from wide outcome variation without confusing systematic bias with variance.</p>"
        page += (
            table(
                clean(shown.to_dict("records")),
                [
                    ("position", "Position"),
                    ("spread_bin", "Spread group"),
                    ("n", "Rows"),
                    ("predicted_std", "Predicted SD"),
                    ("observed_residual_std", "Observed residual SD"),
                    ("observed_bias", "Residual bias"),
                    ("coverage_80", "80% coverage"),
                ],
            )
            + "</details>"
        )
    page += (
        details("Workload and player-history cohorts", clean(cohort.to_dict("records")))
        + "</section>"
    )
    page += '<section id="players"><h2>Current player forecasts</h2><p>Mean, median, and 10th–90th percentile interval have different meanings. The interval describes outcome variation, not a confidence interval on average player ability. Forecasts use the saved observation cutoff.</p><div class="controls"><label>Position <select id="position"><option>ALL</option><option>QB</option><option>RB</option><option>WR</option><option>TE</option></select></label><label>Player <input id="search" type="search" placeholder="Search name"></label><span id="count"></span></div><div id="players-table"></div></section>'
    page += '<section id="lineups"><h2>Lineups and K/DST</h2><p>Team means add. Team intervals come from joint draws, with an independent-draw control. Historical artificial lineups and observed 2026 league lineups are separate experiments. The comparable mixed target combines the original offensive scoring contract with ESPN K/DST scores; official ESPN totals are retained separately.</p>'
    synthetic = lineup.get("synthetic", {})
    support = synthetic.get("coverage", {})
    page += f"<p>The primary artificial-lineup comparison requires exact v2 forecasts for every starter: {support.get('exact_v2_lineups', 'pending')} paired lineups. An additional {support.get('lineups_with_v2_proxy', 'pending')} lineups contain a fallback v2 estimate and are retained only as a separately labeled sensitivity analysis.</p>"
    page += table(
        [
            dict(method=k, **synthetic[k])
            for k in ("v2_mean", "v3_independent", "v3_joint")
            if k in synthetic
        ],
        [
            ("method", "Synthetic lineup method"),
            ("n", "Team-weeks"),
            ("mae", "MAE"),
            ("rmse", "RMSE"),
            ("crps", "CRPS"),
            ("coverage80", "80% coverage"),
            ("width80", "Width"),
        ],
    )
    page += "<p>Joint dependence modeling did not show a convincing improvement over independent draws: the season-bootstrap interval for its CRPS difference includes zero. The observed 2026 comparison below is small and retrospective, and interval support is narrower than point-forecast support. No archived league forecast was verified as published before kickoff.</p>"
    page += table(
        [dict(method=k, **v) for k, v in lineups_summary.get("metrics", {}).items()],
        [
            ("method", "Observed league comparison"),
            ("n", "Paired team-weeks"),
            ("mae", "MAE"),
            ("rmse", "RMSE"),
            ("interval_n", "Interval cases"),
            ("coverage80", "80% coverage"),
        ],
    )
    page += details(
        "Historical artificial-lineup comparison",
        lineup.get("synthetic", lineup.get("synthetic_summary", {})),
    )
    page += details("Observed league lineup comparison and support", lineups_summary)
    current_teams = run / "lineups/current_lineups.parquet"
    if current_teams.exists():
        rows = clean(pd.read_parquet(current_teams).to_dict("records"))
        page += "<h3>Current full-lineup predictions</h3><p>Both versions use the same K/DST means. Ranges remain provisional; coverage counts show missing offensive forecasts.</p>"
        page += table(
            rows,
            [
                ("team_name", "Team"),
                ("week", "Week"),
                ("covered_starters", "Covered"),
                ("starter_count", "Starters"),
                ("v2_prediction", "v2 points"),
                ("prediction", "v3 points"),
                ("q10", "P10"),
                ("q90", "P90"),
            ],
        )
        pd.DataFrame(rows).to_csv(output / "current_lineups.csv", index=False)
    kicker = lineup.get("kickers", {})
    if kicker:
        page += "<h3>Historical kicker uncertainty</h3><p>The same prior-four-game mean is used for all methods. This experiment uses repository kicker scoring, which differs from the observed ESPN blocked-field-goal penalty. It does not validate uncertainty around ESPN projections.</p>"
        page += table(
            [dict(method=name, **values) for name, values in kicker.get("heldout", {}).items()],
            [
                ("method", "Method"),
                ("n", "Games"),
                ("mae", "MAE"),
                ("coverage80", "80% coverage"),
                ("width80", "Width"),
                ("interval_score80", "Interval score"),
                ("crps", "CRPS"),
            ],
        )
        page += details("Kicker policy selection and scoring evidence", kicker)
    page += details("Full aggregation evidence", lineup) + "</section>"
    page += '<section id="evidence"><h2>Reproduction and evidence</h2><p><a href="comparison.json">Complete report JSON</a> · <a href="paired_v2_v3.csv">Paired comparison CSV</a> · <a href="candidate_scores.csv">Candidate metrics CSV</a> · <a href="current_players.csv">Current player CSV</a> · <a href="selected_predictions.parquet">Saved prediction panel</a></p>'
    page += (
        details("Independent audit", audit)
        + details("Saved model replay", replay)
        + details(
            "Sealed campaign publication",
            read_json(run / "publication.json", {"status": "pending"}),
        )
        + details(
            "Offline sealed campaign replay",
            read_json(run / "sealed_replay.json", {"status": "pending"}),
        )
        + details(
            "Independent preservation check of published bundle",
            read_json(run / "published_bundle_audit.json", {"status": "pending"}),
        )
        + details(
            "Weekly refresh validation in an isolated copy",
            read_json(run / "rescore_e2e.json", {"status": "pending"}),
        )
        + details(
            "Browser verification",
            read_json(output / "browser_validation.json", {"status": "pending"}),
        )
        + details("Dataset and cutoff contract", dataset)
        + details("Candidate definitions and selection", metadata)
    )
    page += (
        "<h3>Limits and research basis</h3><ul>"
        + "".join(f"<li>{html.escape(x)}</li>" for x in notes)
        + "</ul><ul>"
        + sources
        + "</ul></section>"
    )
    page += "<section><h2>Scope covered and next research questions</h2><p>This run covers nine mean candidates, six uncertainty methods, two v2 uncertainty controls, full-history/recent/decayed training, player-history and workload cohorts, teammate/opponent dependence, stacked lineups, two lineup selection designs, and a separate kicker study. It preserves the original scoring contract and distinguishes exact-v2 comparisons from proxy controls.</p><p>It does not exhaust every possible estimator. NGBoost, quantile forests and GPBoost inform the research discussion but were not fitted here. The clearest remaining work is past-only interval recalibration, an explicit availability/active-points mixture, a wider shrinkage-prior sensitivity study, and genuinely prospective forecasts. Historical DST reconstruction and ESPN-projection uncertainty need better archived inputs. These are disclosed limitations of the completed v3 experiment.</p></section>"
    page += (
        "<script>const players="
        + payload
        + """;
const esc=v=>String(v??'—').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function render(){const pos=document.getElementById('position').value,q=document.getElementById('search').value.toLowerCase();const rows=players.filter(p=>(pos==='ALL'||p.position===pos)&&(p.player_name??p.player_id).toLowerCase().includes(q));document.getElementById('count').textContent=rows.length+' players';const cols=[['player_name','Player'],['position','Position'],['team','Team'],['week','Week'],['prediction','v3 mean'],['v2_weekly_mean','v2 mean'],['std','Std dev'],['q10','P10'],['q50','Median'],['q90','P90']];document.getElementById('players-table').innerHTML='<div class="scroll"><table><thead><tr>'+cols.map(c=>'<th>'+c[1]+'</th>').join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+cols.map(([k])=>'<td>'+esc(typeof r[k]==='number'&&k!=='week'?r[k].toFixed(2):r[k])+'</td>').join('')+'</tr>').join('')+'</tbody></table></div>';}
document.getElementById('position').addEventListener('change',render);document.getElementById('search').addEventListener('input',render);render();</script></main></body></html>"""
    )
    (output / "index.html").write_text(page)
    return output / "index.html"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "data/outputs/opening_drive_v3_comparison_2026"
    )
    args = parser.parse_args()
    print(build_report(args.run_dir, args.output_dir))
