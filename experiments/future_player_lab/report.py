"""Portable, filterable experiment readout with explicit coverage and limitations."""
# ruff: noqa: E501 -- Embedded HTML and report paragraphs retain natural wrapping.

from __future__ import annotations

import json

import numpy as np
import polars as pl

from .evaluate import matched_scores, paired_interval


def generate(run, config, *, destination=None):
    output = destination or run
    frames = [pl.read_parquet(p) for p in sorted((run / "tasks").glob("*/metrics.parquet"))]
    if not frames:
        raise ValueError("No completed tasks to report")
    metrics = pl.concat(frames, how="diagonal_relaxed")
    metrics.write_parquet(output / "metrics.parquet")
    forecasts = {
        p.parent.name: pl.read_parquet(p) for p in (run / "tasks").glob("*/predictions.parquet")
    }
    groups = metrics.partition_by(["position", "horizon", "target", "model"], as_dict=True)
    records = []
    for (pos, horizon, target, model), block in groups.items():
        valid = block.filter(pl.col("mse").is_not_null())
        if not valid.height:
            continue
        paired = matched_scores(forecasts[f"{pos}_{horizon}"], model, "persistence", target)
        diffs = np.asarray([r["delta_mse"] for r in paired])
        interval = (
            paired_interval(diffs, config["bootstrap_draws"], config["seed"])
            if len(diffs)
            else [None, None]
        )
        records.append(
            {
                "position": pos,
                "horizon": horizon,
                "target": target,
                "model": model,
                "years": valid.height,
                "n": int(valid["n"].sum()),
                "matched_n": sum(r["n"] for r in paired),
                "matched_years": len(paired),
                "rmse": float(np.sqrt(valid["mse"].mean())),
                "mae": valid["mae"].mean(),
                "spearman": valid["spearman"].mean(),
                "ndcg24": valid["ndcg24"].mean(),
                "delta_mse": float(diffs.mean()) if len(diffs) else None,
                "delta_low": interval[0],
                "delta_high": interval[1],
                "coverage80": valid["coverage80"].mean() if "coverage80" in valid else None,
            }
        )
    records.sort(key=lambda r: (r["position"], r["horizon"], r["target"], r["rmse"]))
    pl.DataFrame(records).write_parquet(output / "summary.parquet")
    (output / "summary.json").write_text(json.dumps(records, indent=2, allow_nan=False))
    policies = [r for r in records if r["model"] == "selected_mse" and r["target"] == "points"]
    lines = [
        "# Future player lab",
        "",
        "Research only. Historical reconstruction; no production promotion.",
        "",
        "## Chronologically selected points forecasts",
        "",
        "| Position | Horizon | Years | RMSE | MAE | Δ MSE vs persistence |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in policies:
        lines.append(
            f"| {r['position']} | {r['horizon']} | {r['years']} | {r['rmse']:.2f} | {r['mae']:.2f} | {r['delta_mse']:.2f} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "Only selected_mse, selected_rank and selected_ensemble are adaptive policies chosen before each test season. "
        "Fixed candidate results are descriptive comparisons, not independently selected winners.",
        "",
        "Five outputs are fitted: league points, position-specific workload, yards, touchdowns and observed scoring-record weeks. "
        "Workload means attempts for QB, carries + targets for RB, and targets for WR/TE. "
        "It is not snaps, medical availability or a claim of causal talent.",
        "",
        "MSE/MAE are equally weighted across seasons; ranking is computed within forecast origin. "
        "Absolute metrics use each model's available predictions; Δ MSE uses exactly matched rows. "
        "Δ MSE intervals resample seasons, are unadjusted for multiple comparisons, and can be unstable with few years. "
        "80% intervals use only prior policy forecast errors; coverage is empirical, with no exchangeability guarantee.",
        "",
        "Historical raw observations start in 2001; the verified preseason candidate population starts in 2004. "
        "In-season candidates retain that preseason population, including zero future production. "
        "Midseason newcomers outside that population are not evaluated. Missing targets remain unknown labels. "
        "No row is excluded based on future workload.",
        "",
        "Historical data are reconstructed from current source vintages. This is not an untouched holdout, "
        "and complete 2026 outcomes are unavailable. Preseason static evidence is carried forward for in-season forecasts; "
        "weekly observations update the history but there is no claim of a complete live roster/news feed.",
        "",
        "See config.json, candidates.json, manifest.json, panel_audit.json, task choices, diagnostics and predictions for provenance. "
        "The source snapshot is stored in source/. Failed candidates are recorded and excluded from selection, never silently replaced.",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines))
    payload = json.dumps(records, allow_nan=False).replace("<", "\\u003c")
    page = """<!doctype html><meta charset="utf-8"><title>Future player lab</title>
<style>body{font:15px system-ui;background:#101827;color:#e5edf8;margin:32px}h1{font-size:28px}input,select{font:inherit;padding:9px;margin:5px;background:#202d41;color:white;border:1px solid #5f718a;border-radius:5px}table{border-collapse:collapse;width:100%}th,td{text-align:right;padding:9px;border-bottom:1px solid #354359}th{position:sticky;top:0;background:#202d41;cursor:pointer}td:first-child{text-align:left}.note{max-width:1000px;color:#bccce1;line-height:1.5}a{color:#8ed8ff}</style>
<h1>Future player lab</h1><p class="note">Historical forecast experiments. Select a target and horizon to compare equivalent tasks. Adaptive policies choose their models using earlier seasons. Click a column to sort. Lower RMSE/MAE is better; higher ranking scores are better. Δ MSE compares with persistence. These results do not promote a production model.</p>
<p><a href="report.md">Readout</a> · <a href="manifest.json">Run provenance</a> · <a href="panel_audit.json">Data coverage</a></p>
<select id="position"></select><select id="horizon"></select><select id="target"></select><input id="query" placeholder="Filter model"><label><input type="checkbox" id="policy" checked>Adaptive policies and baseline</label><table><thead id="head"></thead><tbody id="body"></tbody></table>
<script>const data=PAYLOAD;const keys=['model','years','rmse','mae','spearman','ndcg24','delta_mse','delta_low','delta_high','coverage80'];let order='rmse',dir=1;
for(const key of ['position','horizon','target']){const e=document.getElementById(key);for(const v of [...new Set(data.map(x=>x[key]))])e.add(new Option(v,v));e.onchange=render}document.getElementById('target').value='points';for(const id of ['query','policy'])document.getElementById(id).oninput=render;
document.getElementById('head').innerHTML='<tr>'+keys.map(k=>'<th>'+k+'</th>').join('')+'</tr>';document.querySelectorAll('th').forEach((th,i)=>th.onclick=()=>{dir=order===keys[i]?-dir:1;order=keys[i];render()});
function render(){let rows=data.filter(r=>['position','horizon','target'].every(k=>r[k]===document.getElementById(k).value)&&r.model.includes(document.getElementById('query').value)&&(!document.getElementById('policy').checked||r.model.startsWith('selected_')||r.model==='persistence'));rows.sort((a,b)=>dir*(typeof a[order]==='string'?a[order].localeCompare(b[order]):(a[order]??Infinity)-(b[order]??Infinity)));const body=document.getElementById('body');body.replaceChildren();for(const r of rows){const tr=document.createElement('tr');for(const k of keys){const td=document.createElement('td');td.textContent=r[k]==null?'—':typeof r[k]==='number'?r[k].toFixed(k==='years'?0:3):r[k];tr.append(td)}body.append(tr)}}render();</script>"""
    page = page.replace(
        "Δ MSE compares with persistence.",
        "Absolute scores use available predictions (n); Δ MSE compares identical rows "
        "with persistence (matched_n). Missing historical workload can make these counts differ.",
    )
    page = page.replace("['model','years','rmse'", "['model','years','n','matched_n','rmse'")
    (output / "index.html").write_text(page.replace("PAYLOAD", payload))
    return records
