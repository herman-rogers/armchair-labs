#!/usr/bin/env python3
# ruff: noqa: E402, E501
"""Build a standalone, searchable comparison report from verified, completed lab trials."""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import polars as pl

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB.parents[1] / "src"))

from followup_closure import artifact
from followup_common import Inputs
from safety import install_write_guard, new_run, sha256, write_json


def supplemental(inputs, root):
    policies, population_atoms, wis = [], [], []
    for position in ("QB", "RB", "WR", "TE"):
        frame = pl.read_parquet(
            artifact(inputs, "calibration_001", f"parts/{position}_quantiles.parquet")
        )
        source = frame.filter(
            (pl.col("method") == "quantile_uncalibrated") & (pl.col("level") == 0.8)
        )
        for year in sorted(source["season"].unique()):
            values = source.filter(pl.col("season") == year).to_dicts()
            k = min(len(values), dict(QB=12, RB=24, WR=30, TE=12)[position])
            for policy, field in (
                ("mean", "prediction"),
                ("conditional_median", "median"),
                ("conditional_lower80", "lower"),
                ("conditional_lower80_mean_tiebreak", "lower"),
            ):
                # Zero lower quantiles create many ties. Expose both deterministic tie rules.
                def ordering(row, field=field, policy=policy):
                    return (
                        -round(row[field], 8),
                        -row["prediction"] if policy.endswith("mean_tiebreak") else 0,
                        row["player_id"],
                    )

                chosen = sorted(values, key=ordering)[:k]
                policies.append(
                    dict(
                        position=position,
                        season=year,
                        policy=policy,
                        n_selected=k,
                        actual_points=sum(r["actual"] for r in chosen),
                        player_ids=[r["player_id"] for r in chosen],
                    )
                )
        for (population,), group in source.group_by("population"):
            population_atoms.append(
                dict(
                    position=position,
                    population=population,
                    n=group.height,
                    zero_outcome_fraction=group.filter(pl.col("actual") == 0).height / group.height,
                    zero_lower_bound_fraction=group.filter(pl.col("lower").abs() < 1e-8).height
                    / group.height,
                )
            )
    base = {(r["position"], r["season"]): r for r in policies if r["policy"] == "mean"}
    sensitivity = []
    for position in ("QB", "RB", "WR", "TE"):
        for policy in (
            "conditional_median",
            "conditional_lower80",
            "conditional_lower80_mean_tiebreak",
        ):
            values = [r for r in policies if r["position"] == position and r["policy"] == policy]
            gains = np.array(
                [r["actual_points"] - base[position, r["season"]]["actual_points"] for r in values]
            )
            rng = np.random.default_rng(20260923)
            boot = rng.choice(gains, (4000, len(gains))).mean(axis=1)
            sensitivity.append(
                dict(
                    position=position,
                    policy=policy,
                    seasons=len(values),
                    mean_points_gain=float(gains.mean()),
                    ci_low=float(np.quantile(boot, 0.025)),
                    ci_high=float(np.quantile(boot, 0.975)),
                    changed_seasons=sum(
                        set(r["player_ids"]) != set(base[position, r["season"]]["player_ids"])
                        for r in values
                    ),
                )
            )
    frame = pl.read_parquet(
        artifact(inputs, "calibration_001", "weighted_interval_scores.parquet")
    ).filter((pl.col("target") == "season_points") & (pl.col("base") == "market_control"))
    baseline = frame.filter(pl.col("method") == "global_scaled").select(
        "player_id", "season", "position", pl.col("wis").alias("control_wis")
    )
    frame = frame.join(baseline, on=["player_id", "season", "position"], validate="m:1")
    for (position, method), group in frame.group_by("position", "method"):
        annual = group.group_by("season").agg(pl.col("wis").mean(), pl.col("control_wis").mean())
        score, control = annual["wis"].mean(), annual["control_wis"].mean()
        wis.append(
            dict(
                position=position,
                method=method,
                wis=score,
                control_wis=control,
                gain_pct=100 * (control - score) / control,
            )
        )
    write_json(
        root / "supplemental_diagnostics.json",
        dict(
            conditional_policy_comparisons=sensitivity,
            conditional_policy_rows=policies,
            quantile_population_atoms=population_atoms,
            weighted_interval_scores=wis,
            interpretation="Descriptive follow-ups using saved OOF forecasts; no policy selection, executable costs, feasible roster replay, profit claim, or untouched confirmation. Original global mean/median/lower policies choose identical sets; conditional quantile policies can differ.",
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--audit-run", default="audit_002")
    args = parser.parse_args()
    root = new_run(LAB, args.run_id)
    install_write_guard(root)
    inputs = Inputs()
    inputs.bind(Path(__file__))
    rows, manifests = [], {}
    files = {
        "calibration_001": ["calibration_summaries.json"],
        "signals_001": ["signals_summaries.json"],
        "structure_001": [
            "team_volume_summaries.json",
            "team_allocation_summaries.json",
            "rookie_summaries.json",
            "availability_summaries.json",
            "weekly_roles_summaries.json",
        ],
        "distributions_001": ["joint_point_summaries.json", "joint_score_summaries.json"],
        args.audit_run: ["age_recency_summaries.json", "signed_calibration_summaries.json"],
    }
    seen = set()
    for run, names in files.items():
        manifests[run] = json.loads((LAB / "runs" / run / "manifest.json").read_text())
        for name in names:
            for r in json.loads(artifact(inputs, run, name).read_text()):
                base = dict(
                    run=run,
                    position=r["position"],
                    target=r.get("target", "receiving_yards"),
                    model=r.get("model", r.get("method")),
                    slice=r["slice"],
                    n=r["n"],
                    seasons=r.get("years", r.get("seasons")),
                    holm_p=r.get("holm_p"),
                    level=r.get("level"),
                    coverage=None,
                    width=None,
                    fallback=None,
                )
                if "score_gain_pct" in r:
                    base.update(
                        category="Signed CQR" if run == args.audit_run else "Calibration",
                        control="global_scaled",
                        base=r["base"],
                        metric="interval score",
                        gain_pct=r["score_gain_pct"],
                        ci_low=r["ci_low"],
                        ci_high=r["ci_high"],
                        coverage=r["coverage"],
                        width=r["width"],
                        fallback=r["fallback_fraction"],
                    )
                elif "relative_gain" in r:
                    base.update(
                        category="Joint distributions",
                        control="direct_residual",
                        base="market_control",
                        metric=r["metric"],
                        gain_pct=100 * r["relative_gain"],
                        ci_low=r["gain_ci_low"],
                        ci_high=r["gain_ci_high"],
                    )
                else:
                    base.update(
                        category="Age/recency" if run == args.audit_run else "Point forecasts",
                        control=r["control"],
                        base="market_control",
                        metric="Brier"
                        if r["target"] in ("meaningful_role", "next_block_majority_starter")
                        else "MSE",
                        gain_pct=r["mse_gain_pct"],
                        ci_low=r["mse_ci_low"],
                        ci_high=r["mse_ci_high"],
                    )
                key = tuple(
                    (k, base[k])
                    for k in (
                        "run",
                        "category",
                        "position",
                        "target",
                        "model",
                        "control",
                        "base",
                        "slice",
                        "metric",
                        "level",
                    )
                )
                if key not in seen:
                    seen.add(key)
                    rows.append(base)
    write_json(root / "comparisons.json", rows)
    supplemental(inputs, root)
    with (root / "comparisons.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = json.dumps(rows).replace("<", "\\u003c")
    links = " · ".join(f'<a href="../{name}/manifest.json">{name}</a>' for name in manifests)
    html = TEMPLATE.replace("__PAYLOAD__", payload).replace("__LINKS__", links)
    (root / "index.html").write_text(html)
    (root / "source.py").write_bytes(Path(__file__).read_bytes())
    inputs.verify()
    write_json(
        root / "manifest.json",
        dict(
            status="complete",
            research_only=True,
            protected_inputs_unchanged=True,
            inputs=inputs.hashes,
            runs=list(manifests),
            comparison_rows=len(rows),
            outputs={p.name: sha256(p) for p in root.iterdir() if p.is_file()},
        ),
    )
    print(f"Report: {root / 'index.html'} ({len(rows)} comparisons)")


TEMPLATE = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Player signals lab — follow-up experiments</title>
<style>
body{font:15px system-ui,sans-serif;margin:30px;color:#17263a;background:#f7f9fc}main{max-width:1500px;margin:auto}h1{font-size:28px}h2{font-size:20px}p{max-width:1050px;line-height:1.6}a{color:#1761a0}select,input{padding:8px;border:1px solid #a4b3c4;border-radius:4px;background:white;margin:5px}label{white-space:nowrap}section{background:white;padding:20px;border:1px solid #dce3eb;border-radius:8px;margin:20px 0}.scroll{overflow:auto;max-height:700px}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:9px;text-align:left;border-bottom:1px solid #e0e6ec;white-space:nowrap}th{position:sticky;top:0;background:#eaf0f7;cursor:pointer}tr:hover{background:#f2f6fc}.barrow{display:grid;grid-template-columns:115px 1fr 150px;gap:15px;align-items:center;margin:18px 0}.bar{height:23px;background:#eef2f6;position:relative}.fill{height:100%;background:#527db0}.line{position:absolute;top:-5px;bottom:-5px;width:2px;background:#8c2727}.note{color:#556579;font-size:13px}.positive{color:#16634f}.negative{color:#a13333}#count{font-weight:600}
</style><main><h1>Player signals lab: follow-up experiments</h1>
<p>Historical research only. All forecasts use earlier seasons; calibration uses earlier out-of-fold residuals. Positive gain means lower loss against the displayed matched control. The historical years have been reused: adjusted tests and bootstrap intervals do not make this prospective evidence.</p>
<p><a href="../../followup_readout.md">Readout and recommendations</a> · <a href="../../research_queue.md">Research-path coverage</a> · <a href="comparisons.csv">Download comparisons</a></p>
<section><h2>Coverage by player population</h2>
<p>Season fantasy points with the market control. The red line is nominal coverage; each blue bar is observed coverage, averaged equally across seasons. Score improvements can coexist with poor subgroup coverage.</p>
<label>Position <select id="chartPosition"></select></label><label>Method <select id="chartMethod"></select></label><label>Nominal coverage <select id="chartLevel"><option value="0.8">80%</option><option value="0.9">90%</option></select></label>
<div id="chart"></div><p class="note">Original calibration: 2010–2025. Signed CQR: matched 2013–2025 follow-up, proposed after the first results. Sparse populations may use global calibration. These are prediction intervals, not confidence intervals for model parameters.</p></section>
<section><h2>All comparisons</h2><label>Category <select id="category"></select></label><label>Position <select id="position"></select></label><label>Slice <select id="slice"></select></label><input id="search" type="search" aria-label="Search comparisons" placeholder="Search target, model, control…"><p id="count"></p>
<p class="note">95% gain intervals are in the original loss units; gain percentages use the control loss as denominator. Holm adjustment applies to each declared stage's all-population primary comparisons. Blank p-values denote descriptive/secondary comparisons. Joint-distribution coverage is available in the saved score summaries.</p>
<div class="scroll"><table><thead><tr id="headers"></tr></thead><tbody id="body"></tbody></table></div></section>
<p class="note">Verified source runs: __LINKS__. Everything is local and disposable; this page loads no external scripts or data.</p></main>
<script>
const data=__PAYLOAD__;
const $=id=>document.getElementById(id);
function options(id,values,all=false){$(id).replaceChildren();for(const value of (all?['all values',...values]:values)){const o=document.createElement('option');o.value=value;o.textContent=value;$(id).append(o);}}
for(const key of ['category','position','slice'])options(key,[...new Set(data.map(r=>r[key]))].sort(),true);
$('slice').value='all';
const chartRows=data.filter(r=>(r.category==='Calibration'||r.model.startsWith('signed_'))&&r.target==='season_points'&&r.base==='market_control');
options('chartPosition',['QB','RB','WR','TE']);options('chartMethod',[...new Set(chartRows.map(r=>r.model))].sort());$('chartMethod').value='volume_scaled';
function chart(){const rows=chartRows.filter(r=>r.position===$('chartPosition').value&&r.model===$('chartMethod').value&&r.level===Number($('chartLevel').value)&&['rookie','returner','market_only'].includes(r.slice));$('chart').replaceChildren();for(const r of rows){const row=document.createElement('div');row.className='barrow';const label=document.createElement('span');label.textContent=r.slice;const bar=document.createElement('div');bar.className='bar';const fill=document.createElement('div');fill.className='fill';fill.style.width=(100*r.coverage)+'%';const line=document.createElement('div');line.className='line';line.style.left=(100*r.level)+'%';bar.append(fill,line);const value=document.createElement('span');value.textContent=(100*r.coverage).toFixed(1)+'% · n='+r.n;row.append(label,bar,value);row.title='Mean width '+r.width.toFixed(1)+'; global fallback '+(100*r.fallback).toFixed(1)+'%';$('chart').append(row);}}
const cols=[['category','Category'],['position','Pos'],['target','Target'],['model','Method'],['control','Control'],['base','Base'],['slice','Slice'],['metric','Loss'],['level','Level'],['n','Rows'],['seasons','Years'],['gain_pct','Gain %'],['ci_low','95% gain low'],['ci_high','95% gain high'],['holm_p','Holm p'],['coverage','Coverage'],['width','Width']];let sortKey='gain_pct',direction=-1;
for(const [key,label] of cols){const th=document.createElement('th');th.textContent=label;th.onclick=()=>{direction=sortKey===key?-direction:-1;sortKey=key;render();};$('headers').append(th);}
function render(){const search=$('search').value.toLowerCase();let rows=data.filter(r=>['category','position','slice'].every(k=>$(k).value==='all values'||r[k]===$(k).value)&&JSON.stringify(r).toLowerCase().includes(search));rows.sort((a,b)=>{let x=a[sortKey],y=b[sortKey];if(x===null)return 1;if(y===null)return -1;return direction*(typeof x==='number'?x-y:String(x).localeCompare(String(y)));});$('count').textContent=rows.length+' comparisons';$('body').replaceChildren();for(const r of rows){const tr=document.createElement('tr');for(const [key] of cols){const td=document.createElement('td');let v=r[key];td.textContent=v===null?'—':typeof v==='number'?(['n','seasons'].includes(key)?v.toLocaleString():['level','coverage'].includes(key)?(v*100).toFixed(1)+'%':key==='holm_p'?v.toFixed(4):v.toFixed(2)):v;if(key==='gain_pct')td.className=v>0?'positive':v<0?'negative':'';tr.append(td);}$('body').append(tr);}}
for(const id of ['category','position','slice','search'])$(id).addEventListener('input',render);for(const id of ['chartPosition','chartMethod','chartLevel'])$(id).addEventListener('input',chart);chart();render();
</script></html>"""


if __name__ == "__main__":
    main()
