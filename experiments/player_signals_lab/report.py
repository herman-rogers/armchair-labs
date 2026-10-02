"""Local, static results; no application routes or publication hooks."""

from __future__ import annotations

import html


def markdown_report(summaries, intervals, manifest):
    lines = [
        f"# Player signals lab — {manifest['run_id']}",
        "",
        "Exploratory preseason experiments. No production model, registry, catalog, or forecast "
        "was updated. Historical seasons have been researched before; this is not a fresh holdout.",
        "",
        f"Gold: `{manifest['gold']['version']}`. "
        f"Evaluation: {manifest['evaluation_start']}–{manifest['evaluation_end']}. "
        f"Tasks: {len(manifest['tasks'])}. "
        "Full configuration and source snapshots are in this run.",
        "",
        "## Matched comparisons",
        "",
        "Each challenger is compared with the named control on identical player-season rows. "
        "Positive gains mean less error. MSE is primary for these squared-loss mean models; "
        "MAE is secondary. Seasons receive equal weight. CIs are nominal paired season bootstrap "
        "intervals, unadjusted for multiple tests and repeated research. No promotion threshold.",
        "",
        "| Position / target | Candidate | Control | N / years | MAE gain | MSE gain % | "
        "MSE 95% CI above zero? |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for r in summaries:
        if r["slice"] != "all" or r["model"] in ("reference", "nextgen_boost"):
            continue
        percentage = f"{r['mse_gain_pct']:.2f}" if r["mse_gain_pct"] is not None else "—"
        lines.append(
            f"| {r['position']} {r['target']} | {r['model']} | {r['control']} | "
            f"{r['n']} / {r['years']} | {r['mae_gain']:.3f} | {percentage} | "
            f"{'yes' if r['mse_ci_low'] > 0 else 'no'} |"
        )
    lines += [
        "",
        "## Interval diagnostics",
        "",
        "These intervals use only earlier out-of-fold residuals (up to five prior seasons). "
        "The first three calibration seasons are withheld. Coverage is empirical, without "
        "an exchangeability guarantee. Lower interval score is better; wide intervals can "
        "cover well without being useful. All models and population slices are in JSON.",
        "",
        "| Position / target | Model | Method | Nominal | Coverage | Width | Interval score |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for r in intervals:
        if (
            r["slice"] == "all"
            and r["model"] in ("raw_rates", "market_control")
            and r["level"] == 0.8
        ):
            lines.append(
                f"| {r['position']} {r['target']} | {r['model']} | {r['method']} | "
                f"80% | {100 * r['covered']:.1f}% | {r['width']:.2f} | "
                f"{r['interval_score']:.2f} |"
            )
    lines += [
        "",
        "## Boundaries",
        "",
        "- Market controls use repaired cutoff-safe ECR where present; missing values stay "
        "missing. Inspect `market_covered` before claiming information beyond ECR. This is "
        "not a replication of the full 224-feature individual-stat screen.",
        "- Participation is passing-play participation, not verified routes. Coverage "
        "fractions use the source's 18-week grid; 0.5 means at least nine observed weeks.",
        "- QB roles are outcome strata defined by games with at least 15 attempts "
        "(0, 1–7, 8+). They do not identify the opening starter, health, or job retention.",
        "- Efficiency labels are conditional on positive observed opportunities. Missing "
        "efficiency is not zero; count/points tasks retain zero-production candidates.",
        "- Gaussian rate pooling is approximate and univariate. Opportunity counts do not "
        "equal independent observations, and this model does not isolate intrinsic ability.",
        "- These are historical preseason forecasts, not current rest-of-season advice. "
        "Provider reconstruction dates may differ from original publication vintages.",
        "- Role mixtures predict whole-season totals conditional on role state; they do "
        "not multiply independent expected volume and efficiency. Team-total coherence, "
        "dated news, college development and decision utility remain separate research.",
        "",
        "## Files",
        "",
        "`predictions.parquet`: every held-out forecast, control, interval, cohort flag and "
        "role probability. `summaries.json`: all predeclared slices and annual paired losses. "
        "`intervals.json`: coverage, width, proper interval scores, and mean/median MAE on "
        "identical calibrated rows. `folds.json`: train/test counts, feature recipes, pooling "
        "parameters and time bounds. `manifest.json`: hashes, versions, status "
        "and isolation audit.",
        "",
    ]
    return "\n".join(lines)


def html_report(summaries, manifest):
    rows = []
    for r in summaries:
        text = " ".join(str(r[c]) for c in ("position", "target", "model", "control", "slice"))
        values = [
            r["position"],
            r["target"],
            r["model"],
            r["control"],
            r["slice"],
            f"{r['n']} / {r['years']}",
            f"{r['mae_gain']:.3f}",
            f"{r['mse_gain_pct']:.2f}" if r["mse_gain_pct"] is not None else "—",
            f"[{r['mse_ci_low']:.3f}, {r['mse_ci_high']:.3f}]",
        ]
        rows.append(
            f'<tr data-search="{html.escape(text.lower(), quote=True)}">'
            + "".join(f"<td>{html.escape(str(v))}</td>" for v in values)
            + "</tr>"
        )
    return r"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Player signals experiment lab</title><style>
body{font:15px system-ui;margin:32px;color:#182734;background:#fafbfc}
input{font:inherit;padding:10px;width:min(600px,90%);margin:15px 0}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:9px;text-align:left;
border-bottom:1px solid #ddd}th{position:sticky;top:0;background:#e6eef3}
tr:hover{background:#edf4f7}
.scroll{overflow:auto}a{color:#0759a6}</style>
<h1>Player signals experiment lab</h1>
<p>RUN_ID · Historical preseason research · Nothing promoted</p>
<p>Matched controls, equal-season losses. Positive gains mean lower error. MSE is primary;
nominal 95% season bootstrap intervals are exploratory and unadjusted for multiple testing.</p>
<p><a href="report.md">Full report and limitations</a> · <a href="summaries.json">All results</a> ·
<a href="intervals.json">Interval calibration</a> · <a href="manifest.json">Provenance</a></p>
<label for="filter">Filter by position, target, model, or population</label><br>
<input id="filter" type="search" placeholder="e.g. QB role_mixture all" value="all">
<p id="count" aria-live="polite"></p><div class="scroll"><table><thead><tr>
<th>Position</th><th>Target</th><th>Model</th><th>Control</th><th>Slice</th><th>N / years</th>
<th>MAE gain</th><th>MSE gain %</th><th>MSE gain CI</th></tr></thead>
<tbody>ROWS</tbody></table></div>
<script>const input=document.querySelector('#filter');
function filter(){const terms=input.value.toLowerCase().trim().split(/\s+/);let count=0;
for(const row of document.querySelectorAll('tbody tr')){
row.hidden=!terms.every(term=>row.dataset.search.split(' ').some(word=>word.includes(term)));
if(!row.hidden)count++;}document.querySelector('#count').textContent=count+' comparisons';}
input.addEventListener('input',filter);filter();</script></html>""".replace(
        "RUN_ID", html.escape(manifest["run_id"])
    ).replace("ROWS", "\n".join(rows))
