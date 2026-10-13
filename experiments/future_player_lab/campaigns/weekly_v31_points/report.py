"""Interactive, static review of the v3.1 point-forecast experiment."""
# ruff: noqa: E501

from __future__ import annotations

import json

import pandas as pd

NAMES = {
    "v3": "Original v3",
    "v31": "V3.1 intervals / unchanged v3 mean",
    "v3_residual_intervals": "V3 mean / matched residual intervals",
    "bias_global": "Position bias correction",
    "select_rmse": "Prior RMSE selection",
    "select_guarded": "Prior RMSE selection + subgroup guard",
    "blend_equal": "Equal model blend",
    "blend_global": "Historical MSE blend",
    "blend_convex": "Covariance-aware convex blend",
}


def label(method):
    return NAMES.get(method, method.replace("_", " "))


def render(directory, output=None):
    scores = pd.read_csv(directory / "scores.csv")
    comparisons = pd.read_csv(directory / "comparisons.csv")
    rankings = pd.read_csv(directory / "rankings.csv")
    ablations = pd.read_csv(directory / "ablations.csv")
    pooled = scores[scores.season.eq("all")]
    primary = pooled[pooled.cohort.eq("all")]
    summary = []
    for position, group in primary.groupby("position"):
        baseline = group[group.method.eq("v31")].iloc[0]
        chosen = group[group.method.eq("select_guarded")].iloc[0]
        rank = rankings[rankings.position.eq(position)].groupby("method").selected_points.mean()
        summary.append(
            dict(
                position=position,
                v3_rmse=baseline.rmse,
                selected_rmse=chosen.rmse,
                rmse_change_pct=100 * (chosen.rmse / baseline.rmse - 1),
                v3_mae=baseline.mae,
                selected_mae=chosen.mae,
                weekly_top_k_points_change=rank["select_guarded"] - rank["v31"],
                coverage80=chosen.covered_80,
            )
        )
    summary = pd.DataFrame(summary)
    improved = int((summary.rmse_change_pct < 0).sum())
    ab = ablations[ablations.cohort.eq("all")]
    uncertainty_wins = int((ab.delta_rmse < 0).sum())
    estimates = comparisons[comparisons.method.eq("select_guarded") & comparisons.cohort.eq("all")]
    selections = json.loads((directory / "selection.json").read_text())
    final = pd.DataFrame(
        [
            dict(
                position=r["position"],
                policy="guarded" if r["guarded"] else "RMSE",
                method=r["method"],
                latest_validation=r["max_validation_year"],
            )
            for r in selections
            if r["year"] == 2026
        ]
    )
    names = {m: label(m) for m in scores.method.unique()}
    page = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Opening Drive v3.1 · Can uncertainty improve point predictions?</title>
<style>body{{font:16px/1.55 system-ui;background:#f4f7fb;color:#192b40;margin:0}}main{{max-width:1250px;margin:auto;padding:28px}}h1{{line-height:1.18;font-size:clamp(28px,4vw,44px)}}h2{{font-size:23px}}section{{background:white;padding:24px;margin:24px 0;border-radius:12px;border:1px solid #d8e1ec}}.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{text-align:right;padding:9px;border-bottom:1px solid #dde5ec;white-space:nowrap}}th:first-child,td:first-child{{text-align:left}}th{{background:#edf2f7}}label{{display:inline-block;margin:8px 18px 8px 0}}select{{padding:8px;max-width:290px}}svg{{width:100%;height:auto}}a{{color:#175eaa}}.tag{{font-weight:700;color:#245c83}}.note{{color:#53667d}}code{{font-size:13px;overflow-wrap:anywhere}}@media(max-width:600px){{main{{padding:12px}}section{{padding:15px}}}}</style>
<main><p class="tag">OPENING DRIVE · V3.1 POINT-FORECAST RESEARCH</p>
<h1>Can uncertainty feedback improve predicted points?</h1>
<p>This experiment actually changes point predictions. It compares {scores.method.nunique() - 2} fixed candidates and two chronological selectors on 2021–2025 player-weeks. Existing v3.1 interval-only forecasts and prospective captures are preserved.</p>
<section><h2>What the completed experiment says</h2>
<p>The designated policy, selected from earlier seasons with a subgroup error guard, improves pooled point RMSE in <b>{improved} of 4 positions</b>. Adding uncertainty inputs improves RMSE in <b>{uncertainty_wins} of {len(ab)}</b> matched position/model comparisons. These are different questions: a new architecture can help even when the extra uncertainty features do not.</p>
<p>RMSE is in fantasy points and lower is better. Weekly top-k points measures ranking decisions on the same eligible player pool; higher is better. It is not a simulation of actual roster choices.</p>
<div class="scroll">{summary.to_html(index=False, float_format=lambda x: f"{x:.4f}", border=0)}</div>
<p class="note">This is historical development evidence, not an untouched test or proof of a prospective gain. The guarded selector was designated before the initial run. Revision 002 fixes floating-point cancellation in a zero-valued feature and adds two matched half-strength controls after inspecting 001; the initial run is retained. Its guard is checked on earlier seasons and cannot guarantee future subgroup performance.</p></section>
<section><h2>Point accuracy explorer</h2><p>All cohorts use the original v3 predicted-points boundaries, so candidates are compared on identical rows. “Observed appearance” groups are retrospective diagnostics and are never used to choose the model before a game.</p>
<label>Position <select id="position"><option>QB</option><option>RB</option><option>WR</option><option>TE</option></select></label>
<label>Cohort <select id="cohort"></select></label>
<div id="plot"></div><div id="table" class="scroll"></div>
<p class="note">The three point controls have identical RMSE/MAE. V3 uses original intervals, v3.1 uses the existing expanded intervals, and the matched residual-interval control uses the same recalibration recipe as the changed means. Interval quality therefore has a fair control independent of point changes.</p></section>
<section><h2>Does the uncertainty information itself help?</h2><p>Each pair uses the same training years, architecture and regularization. Only the extra forecast spread, asymmetry and earlier interval-correction features differ. Negative deltas favor adding uncertainty. Ordinary historical variability features remain available in both arms.</p>
<div class="scroll">{ab.to_html(index=False, float_format=lambda x: f"{x:.4f}", border=0)}</div></section>
<section><h2>How stable are the point gains across seasons?</h2><p>Paired whole-season bootstrap intervals describe the equal-year RMSE difference versus v3/v3.1. There are only five seasons; these are descriptive intervals, with no multiple-comparison correction.</p>
<div class="scroll">{estimates.to_html(index=False, float_format=lambda x: f"{x:.4f}", border=0)}</div></section>
<section><h2>What was tested</h2><ul>
<li>Position, projected-workload and player bias corrections; three group and three player shrinkage priors.</li>
<li>Regularized linear and shallow-tree corrections, with and without uncertainty inputs; half-strength tree correction.</li>
<li>Direct point models, with and without uncertainty inputs.</li>
<li>A two-part mixture: probability of box-score appearance × expected points if appearing. Zero-point and negative-point appearances are retained. This is a participation proxy, not an official active-roster model.</li>
<li>Seven existing point models blended equally, using historical error, accounting for correlated errors, or using conditional error predictions. Inverse-error weights are a heuristic; their performance is tested rather than assumed.</li>
<li>Two earlier-season selectors. The designated guard protects prior RMSE for projections ≥5 and QBs at 1–5, allowing at most 1% degradation where support is at least 100.</li></ul>
<p>Each fold fits only earlier seasons. Earlier out-of-time forecasts train the correction and blend models. Corrected means receive interval estimates from earlier errors of that same candidate, not from its fitted training residuals. No arbitrary conversion of interval width into variance is used.</p>
<p>Mean squared prediction error is decomposed as error variance plus squared average bias in the exported scores. Reducing a width is not, by itself, evidence of reducing point error.</p></section>
<section><h2>Saved 2026 research choices</h2><div class="scroll">{final.to_html(index=False, border=0)}</div>
<p>Final fitted models are saved for replay and a separately recorded future comparison. They are not deployed. Research selection does not replace already captured v3.1 forecasts, and this study does not rebuild full predictive distributions or joint lineup simulations.</p></section>
<section><h2>Evidence and reproduction</h2><p>
<a href="scores.csv">All point and interval scores</a> · <a href="comparisons.csv">Paired season comparisons</a> ·
<a href="rankings.csv">Weekly ranking outcomes</a> · <a href="ablations.csv">Matched ablations</a> ·
<a href="mixture_scores.csv">Participation probability scores</a> · <a href="examples.csv">Changed prediction examples</a> ·
<a href="selection.json">Every selection decision</a> · <a href="protocol.json">Fixed experiment protocol</a> · <a href="replay.json">Exact replay checks</a></p>
<p>Run: <code>python -m experiments.future_player_lab.campaigns.weekly_v31_points.run</code> with fresh output paths.</p>
<p>Method references: <a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingRegressor.html">stacking and out-of-sample training</a> and <a href="https://lightgbm.readthedocs.io/en/latest/Parameters.html">LightGBM objectives and regularization</a>.</p>
<p class="note">The original v3 policies were chosen on 2018–2020, and evaluation years have been reused in prior development. The study evaluates this bounded set of methods; it does not exhaust every estimator. Archived injury/availability inputs remain incomplete. K/DST are excluded. See the protocol for exact limits.</p></section>
<script>const data={pooled.to_json(orient="records")};const names={json.dumps(names)};
const p=document.getElementById('position'),c=document.getElementById('cohort');
function cohorts(){{let old=c.value;let choices=[...new Set(data.filter(r=>r.position===p.value).map(r=>r.cohort))];c.innerHTML=choices.map(v=>`<option value="${{v}}">${{v}}</option>`).join('');c.value=choices.includes(old)?old:'all';render();}}
function render(){{let rows=data.filter(r=>r.position===p.value&&r.cohort===c.value);const W=960,H=380,P=55;let xmin=Math.min(...rows.map(r=>r.mae))-.03,xmax=Math.max(...rows.map(r=>r.mae))+.03,ymin=Math.min(...rows.map(r=>r.rmse))-.03,ymax=Math.max(...rows.map(r=>r.rmse))+.03;let x=v=>P+(v-xmin)/(xmax-xmin)*(W-2*P),y=v=>H-P-(v-ymin)/(ymax-ymin)*(H-2*P);let svg=`<svg viewBox="0 0 ${{W}} ${{H}}" role="img" aria-label="Point RMSE versus MAE; lower and left is better"><path d="M${{P}},${{P}}V${{H-P}}H${{W-P}}" stroke="#75859a" fill="none"/><text x="${{W/2}}" y="${{H-8}}">Mean absolute error (points)</text><text x="10" y="25">RMSE (points)</text>`;for(let i=0;i<=4;i++){{let a=xmin+i*(xmax-xmin)/4,b=ymin+i*(ymax-ymin)/4;svg+=`<text x="${{x(a)}}" y="${{H-P+22}}" text-anchor="middle" font-size="12">${{a.toFixed(2)}}</text><text x="${{P-8}}" y="${{y(b)+4}}" text-anchor="end" font-size="12">${{b.toFixed(2)}}</text>`;}}rows.forEach(r=>{{let color=r.method==='v31'?'#182e48':r.method==='select_guarded'?'#d15722':'#3983b6';svg+=`<circle cx="${{x(r.mae)}}" cy="${{y(r.rmse)}}" r="${{r.method==='select_guarded'?8:5}}" fill="${{color}}" opacity=".8"><title>${{names[r.method]}}: RMSE ${{r.rmse.toFixed(4)}}, MAE ${{r.mae.toFixed(4)}}</title></circle>`;}});document.getElementById('plot').innerHTML=svg+'</svg><p>Orange: designated selector · dark: v3.1 baseline · blue: other candidates. Hover for details.</p>';document.getElementById('table').innerHTML='<table><thead><tr><th>Method</th><th>Rows</th><th>RMSE ↓</th><th>MAE ↓</th><th>Bias</th><th>80% coverage</th><th>80% width</th><th>Interval score ↓</th></tr></thead><tbody>'+rows.sort((a,b)=>a.rmse-b.rmse).map(r=>`<tr><td>${{names[r.method]}}</td><td>${{r.n}}</td><td>${{r.rmse.toFixed(4)}}</td><td>${{r.mae.toFixed(4)}}</td><td>${{r.bias.toFixed(3)}}</td><td>${{(100*r.covered_80).toFixed(1)}}%</td><td>${{r.width_80.toFixed(2)}}</td><td>${{r.interval_score_80.toFixed(2)}}</td></tr>`).join('')+'</tbody></table>';}}
p.addEventListener('change',cohorts);c.addEventListener('change',render);cohorts();</script></main></html>"""
    (output or directory).joinpath("index.html").write_text(page)
    return summary


if __name__ == "__main__":
    import argparse
    import shutil
    from pathlib import Path

    from ..weekly_v31.artifacts import digest, now, verify, write
    from .run import DEFAULT_REPORT, DEFAULT_RUN

    parser = argparse.ArgumentParser(description="Render sealed point-research results")
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    verify(args.run)
    if args.output.resolve().is_relative_to(args.run.resolve()):
        raise ValueError("Presentation must be outside the sealed run")
    args.output.mkdir(parents=True, exist_ok=True)
    for source in [
        *args.run.glob("*.csv"),
        *[args.run / name for name in ("protocol.json", "selection.json", "replay.json")],
    ]:
        shutil.copy2(source, args.output / source.name)
    render(args.run, args.output)
    write(
        args.output / "report_receipt.json",
        dict(
            rendered_at=now().isoformat(),
            run=str(args.run),
            run_manifest_sha256=digest(args.run / "manifest.json"),
            renderer_sha256=digest(Path(__file__)),
            html_sha256=digest(args.output / "index.html"),
            note="Presentation regenerated from unchanged sealed research tables",
        ),
    )
