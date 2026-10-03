"""Portable, searchable evidence and draft-consistency tables. No serving mutation."""

# ruff: noqa: E501
# The standalone HTML/JavaScript template retains its own line layout.

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl
from individual_stat_registry import build_registry

from engine.data.releases import write_json

PAGE = r"""<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Player consistency and individual-stat evidence</title>
<style>
:root{color-scheme:light}body{font:15px/1.5 system-ui,sans-serif;color:#18333e;background:#f5f7f6;margin:0;padding:28px}main{max-width:1550px;margin:auto}h1{font-size:30px;line-height:1.2;margin-bottom:10px}p{max-width:1050px}label{display:inline-flex;flex-direction:column;gap:4px;font-weight:600}input,select,button{font:inherit;padding:8px;border:1px solid #96a6ad;border-radius:5px;background:white;color:#18333e}input{min-width:250px}.controls{display:flex;gap:12px;flex-wrap:wrap;margin:20px 0}.notice{border-left:4px solid #2a7475;padding:10px 16px;background:#e8f1ef}.scroll{overflow:auto;max-height:70vh;border:1px solid #c2cdcf;background:white}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}td,th{padding:10px 12px;border-bottom:1px solid #dce3e3;text-align:right;vertical-align:top}th{position:sticky;top:0;background:#e4eded;white-space:nowrap}th button{border:0;background:transparent;padding:0;font-weight:650}td:first-child,th:first-child{text-align:left}td:first-child{min-width:220px;overflow-wrap:anywhere}tr:hover{background:#f0f5f4}small{font-size:12px;color:#526770}.positive{color:#146849}.negative{color:#9d342e}details{max-width:650px;text-align:left;min-width:270px}summary{cursor:pointer}#more{margin:15px 0}a{color:#086a80}footer{padding:20px 0;color:#526770} .hidden{display:none}
</style><main>
<h1>Player consistency and individual-stat evidence</h1>
<p>See whether a player's average reflects steady scoring or a few huge weeks, then inspect what each statistic adds to a next-season forecast. The 2026 table is descriptive; historical model tests use completed seasons only.</p>
<p class="notice">Research release <strong>__VERSION__</strong>. Corrected gold, full available careers, and all earlier training seasons. These results do not approve a new ranking or establish an edge at market prices.</p>
<div class="controls">
<label>View<select id="view"><option value="draft">2026 draft consistency</option><option value="individual">Individual-stat results</option><option value="registry">All stats and formulas</option><option value="combinations">Combination checks</option></select></label>
<label>Search<input id="search" placeholder="Player, statistic, family, or formula"></label>
<label>Position<select id="position"><option value="">All</option><option>QB</option><option>RB</option><option>WR</option><option>TE</option></select></label>
<label id="periodLabel">History period<select id="period"><option value="prior">Prior season (2025)</option><option value="recent3">Prior three seasons</option><option value="career">Full available career</option></select></label>
<label id="sampleLabel">Minimum scoring weeks<input id="sample" type="number" min="0" value="0" style="min-width:80px;width:80px"></label>
<label id="laneLabel">Baseline<select id="lane"><option value="">All</option><option value="own">Own data</option><option value="market">Own data + ECR</option></select></label>
<label id="actionLabel">Test<select id="action"><option value="">All</option><option value="add">Add one stat</option><option value="remove_family">Remove from family</option><option value="remove_profile">Remove from full profile</option><option value="remove_baseline">Remove baseline member</option></select></label>
<label id="passLabel">Screen<select id="pass"><option value="">All results</option><option value="yes">Passes preregistered screen</option></select></label>
</div><p id="explain"></p><p id="count" role="status" aria-live="polite"></p>
<div class="scroll"><table><thead id="head"></thead><tbody id="body"></tbody></table></div><button id="more">Show 200 more</button>
<footer>Unknown values appear as —. Click column headings to sort. <a href="registry.csv">Registry CSV</a> · <a href="individual_results.csv">Individual results CSV</a> · <a href="combination_results.csv">Combination results CSV</a> · <a href="draft_consistency_2026.csv">2026 consistency CSV</a></footer>
</main><script id="payload" type="application/json">__DATA__</script>
<script>
const data=JSON.parse(document.getElementById('payload').textContent);
const $=id=>document.getElementById(id);let sortKey='',descending=true,limit=200;
const registry=new Map(data.registry.map(r=>[r.stat,r]));
function render(){
const view=$('view').value,period=$('period').value,query=$('search').value.toLowerCase(),pos=$('position').value;
for(const id of ['periodLabel','sampleLabel'])$(id).classList.toggle('hidden',view!=='draft');
for(const id of ['laneLabel','actionLabel','passLabel'])$(id).classList.toggle('hidden',view!=='individual');
let rows=data[view],columns;
if(view==='draft'){
 const p='distribution_'+period+'_';
 columns=[['player_display_name','Player'],['position','Pos'],['market_ecr','ECR'],[p+'observations','Observed weeks'],[p+'mean','Mean points'],[p+'median','Median points'],[p+'std','Weekly SD'],[p+'cv','SD / mean'],[p+'q25','25th percentile'],[p+'top2_positive_share','Best two / positive points'],[p+'mean_without_top2','Mean without best two']];
 rows=rows.filter(r=>r[p+'observations']>=Number($('sample').value));
 $('explain').textContent='Observed weekly league points include recorded zeros and negatives. SD is population standard deviation (at least two observations). Best-two share uses positive points only; the trimmed mean needs at least three observations. Compare players within a position and similar sample sizes. Calendar-era changes and career role changes can also create variation. These are historical descriptions, not next-season forecast intervals.';
}else if(view==='individual'){
 columns=[['stat','Statistic'],['position','Pos'],['lane','Baseline'],['action','Test'],['context','Combination'],['seasons','Seasons'],['first_season','First test'],['last_season','Last test'],['mean_mae_gain','MAE gain (pts)'],['ci_low','95% low'],['ci_high','95% high'],['mean_mse_gain','MSE gain'],['q_value','BH q'],['passes_screen','Passes screen']];
 rows=rows.filter(r=>(!$('lane').value||r.lane===$('lane').value)&&(!$('action').value||r.action===$('action').value)&&(!$('pass').value||r.passes_screen));
 $('explain').textContent='Positive MAE gain means the tested change reduced error. For “add,” it favors adding the stat; for “remove,” it favors removing it from that specific combination. Linear ridge screen, alpha 100, training-only transforms. 95% intervals resample seasons; BH q covers all primary comparisons. Passing requires >=1 point improvement, >=5 seasons, positive lower interval and leave-one-season-out means, q<=.05, and nonnegative MSE gain. It remains a research candidate. Correlated substitutes and duplicate columns affect attribution.';
}else if(view==='registry'){
 columns=[['stat','Statistic'],['family','Family'],['status','Input disposition'],['finite_rows','Finite player-seasons'],['finite_years','Finite years'],['first_year','First forecast year'],['last_year','Last forecast year'],['exact_alias_of','Exact alias'],['definition','Definition / source / reason']];
 $('explain').textContent='Every numeric gold input, original metric-catalog entry, numeric legacy enriched column, profile predictor, named reconstruction and recorded retired formula receives a disposition. Unavailable or unreconstructed does not mean ineffective. All serving dispositions remain research only. Coverage is descriptive; admission to each model fold uses earlier years only.';
}else{
 columns=[['recipe','Recipe'],['position','Pos'],['architecture','Model'],['baseline','Compared with'],['population','Population'],['era','Test era'],['seasons','Seasons'],['mean_mae_gain','MAE gain (pts)'],['ci_low','95% low'],['ci_high','95% high'],['mean_mse_gain','MSE gain']];
 $('explain').textContent='Fixed ridge and histogram-booster recipes. Controls compare against the full profile; basic/career/profile comparisons use the basic baseline. Market comparisons cover ECR-observed candidates in 2014–2025 after three earlier covered years. Test-era filters never shorten the training history. Intervals are descriptive, unadjusted for multiple combination comparisons. Lower error is not proven draft profit.';
}
rows=rows.filter(r=>(!pos||!r.position||r.position===pos)&&(!query||JSON.stringify(r).toLowerCase().includes(query)));
let key=sortKey|| (view==='draft'?'distribution_'+period+'_mean':view==='registry'?'stat':'mean_mae_gain');
rows=[...rows].sort((a,b)=>{let x=a[key],y=b[key];if(x==null)return y==null?0:1;if(y==null)return -1;return (typeof x==='string'?x.localeCompare(y):Number(x)-Number(y))*(descending?-1:1)});
$('head').replaceChildren();const tr=document.createElement('tr');for(const [key,label]of columns){const th=document.createElement('th'),b=document.createElement('button');b.textContent=label;b.onclick=()=>{descending=sortKey===key?!descending:true;sortKey=key;render()};th.append(b);tr.append(th)}$('head').append(tr);
$('body').replaceChildren();for(const row of rows.slice(0,limit)){const tr=document.createElement('tr');for(const [key]of columns){const td=document.createElement('td'),value=row[key];
if(key==='definition'){const d=document.createElement('details'),s=document.createElement('summary'),p=document.createElement('p');s.textContent='Formula and disposition';p.textContent=row.definition+' Source: '+row.definition_ref+' Reason: '+row.reason;d.append(s,p);td.append(d)}else{td.textContent=value==null?'—':typeof value==='boolean'?(value?'Yes':'No'):typeof value==='number'?(key.endsWith('top2_positive_share')?(100*value).toFixed(1)+'%':Number.isInteger(value)?String(value):value.toFixed(key==='q_value'?4:2)):value;
if(key==='mean_mae_gain')td.className=value>0?'positive':'negative';if(key==='stat'&&registry.has(value))td.title=registry.get(value).definition;}tr.append(td)}$('body').append(tr)}
$('count').textContent=rows.length.toLocaleString()+' matching rows · displaying '+Math.min(limit,rows.length).toLocaleString();$('more').hidden=rows.length<=limit;
}
for(const id of ['view','search','position','period','sample','lane','action','pass'])$(id).addEventListener(id==='search'?'input':'change',()=>{limit=200;if(id==='view'||id==='period')sortKey='';render()});$('more').onclick=()=>{limit+=200;render()};render();
</script></html>"""


def build(root):
    registry = build_registry(pl.read_parquet(root / "features.parquet"), root)
    results = pl.read_parquet(root / "individual_results.parquet")
    counts = dict(results.group_by("stat").len().iter_rows())
    for row in registry:
        row["screened_comparisons"] = counts.get(row["stat"], 0)
        row["screen_coverage"] = (
            "evaluated"
            if counts.get(row["stat"], 0)
            else "no eligible fold: check input disposition, earlier coverage and training variation"
        )
    write_json(root / "registry.json", registry)
    pl.DataFrame(registry).write_parquet(root / "registry.parquet")
    pl.DataFrame(registry).write_csv(root / "registry.csv")
    data = dict(registry=registry)
    for key, filename in [
        ("individual", "individual_results"),
        ("combinations", "combination_results"),
        ("draft", "draft_consistency_2026"),
    ]:
        data[key] = pl.read_parquet(root / (filename + ".parquet")).to_dicts()
    payload = json.dumps(data, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    (root / "index.html").write_text(
        PAGE.replace("__VERSION__", root.name).replace("__DATA__", payload)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    build(parser.parse_args().root)
