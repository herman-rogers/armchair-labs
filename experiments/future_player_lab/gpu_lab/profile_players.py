"""Searchable RB histories from verified saved profiles and pinned observed outcomes."""

from __future__ import annotations

# Embedded standalone HTML keeps its own line layout.
# ruff: noqa: E501
import argparse
import json
import shutil
from pathlib import Path

import polars as pl

from .profile_predict import VARIANTS, verify_profile_library
from .run import RUNS, digest, json_write

STATS = ("carries", "targets", "receptions", "rushing_yards", "receiving_yards")
KEY = ["horizon", "player_id", "year", "origin"]


def browser_data(profile_root):
    root = Path(profile_root)
    manifest = verify_profile_library(root)
    histories, cohorts, players, sources = {}, {}, {}, []
    for horizon in ("season", "remaining"):
        source = next(s for s in manifest["input_sources"] if s["horizon"] == horizon)
        metadata_path = Path(source["dataset_manifest"])
        if digest(metadata_path) != source["dataset_manifest_sha256"]:
            raise ValueError("Profile input manifest changed")
        metadata = json.loads(metadata_path.read_text())
        panel_path = metadata_path.parent / "panel.parquet"
        if digest(panel_path) != metadata["files"]["panel.parquet"]:
            raise ValueError("Pinned observed player history changed")
        sources.append(
            {
                "horizon": horizon,
                "dataset_manifest_sha256": source["dataset_manifest_sha256"],
                "panel_sha256": metadata["files"]["panel.parquet"],
            }
        )
        panel = pl.read_parquet(
            panel_path,
            columns=[*KEY, "player_display_name", "team", "age", *["y_" + s for s in STATS]],
        )
        if panel.select(KEY).is_duplicated().any():
            raise ValueError("Player history keys must be unique")
        for variant in VARIANTS:
            path = root / variant / horizon
            assignments = pl.read_parquet(path / "profile_assignments.parquet").filter(
                pl.col("k") == 5
            )
            if assignments.select(KEY).is_duplicated().any():
                raise ValueError("Full-data profile assignments are duplicated")
            panel = panel.join(
                assignments.select(*KEY, pl.col("profile").alias(variant)),
                on=KEY,
                how="left",
                validate="1:1",
            )
            if panel[variant].null_count():
                raise ValueError("Saved profiles do not cover all pinned player histories")
            profile = json.loads((path / "profiles.json").read_text())
            cohorts.setdefault(variant, {})[horizon] = next(
                r["profiles"] for r in profile["descriptive"] if r["k"] == 5
            )
        histories[horizon] = {}
        for row in panel.sort("year", "player_id").iter_rows(named=True):
            pid = row["player_id"]
            if row["player_display_name"]:
                players[pid] = row["player_display_name"]
            histories[horizon].setdefault(pid, []).append(
                {
                    "year": row["year"],
                    "origin": row["origin"],
                    "team": row["team"],
                    "age": row["age"],
                    "profiles": {v: row[v] for v in VARIANTS},
                    "observed": {stat: row["y_" + stat] for stat in STATS},
                }
            )
    return {
        "players": sorted(
            [{"id": pid, "name": name} for pid, name in players.items()],
            key=lambda row: (row["name"].casefold(), row["id"]),
        ),
        "histories": histories,
        "cohorts": cohorts,
        "stats": STATS,
        "profile_manifest_sha256": digest(root / "manifest.json"),
        "source_outcomes": sources,
    }


def _render(data):
    payload = json.dumps(data, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    page = """<!doctype html><html><head><meta charset="utf-8"><title>Running back player histories</title>
<style>body{font:16px system-ui;max-width:1300px;margin:30px auto;background:#f7f9fc;color:#172839}
p{line-height:1.5}input,select{padding:8px;margin:6px;max-width:390px}table{border-collapse:collapse;width:100%;background:white}
td,th{padding:8px;border-bottom:1px solid #ccd;text-align:left;font-size:14px}tr[data-year]{cursor:pointer}
tr.active{background:#e0edff}h1{font-size:29px}h2{font-size:22px}.muted{color:#5c6b7b}a{color:#1d4ed8}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:22px}@media(max-width:800px){.grid{display:block}body{margin:12px}}</style>
</head><body><h1>Explore running back histories</h1>
<p>Search an RB and inspect how their observed history relates to profiles learned from all available inputs.
Full-data profiles are descriptive. Their IDs belong to one method and horizon fit, and are not intrinsic
talent labels. The separate chronological reports use profiles refitted before each forecast year.</p>
<div><label>Search <input id="search" placeholder="Player name" aria-label="Search player names"></label>
<label>Player <select id="player" aria-label="Player"></select></label></div>
<div><label>Horizon <select id="horizon"><option value="season">Full season</option>
<option value="remaining">Remaining after Week 2</option></select></label>
<label>Profile method <select id="variant"><option value="standard_kmeans">Standard PCA + KMeans</option>
<option value="standard_gmm">Standard PCA + Gaussian mixture</option>
<option value="balanced_kmeans">Source-balanced PCA + KMeans</option>
<option value="balanced_gmm">Source-balanced PCA + Gaussian mixture</option></select></label></div>
<p id="heading"></p><table id="history"></table>
<p class="muted">Outcome columns are recorded totals for the displayed season or remaining period.
They are historical outcomes, not model predictions or inputs to these unsupervised clusters.
Absent future production is recorded as zero; a recorded unknown value remains unknown.
Team is the input panel's last-observed or dated draft team and may differ from the team of later production.</p>
<h2 id="cohortHeading"></h2><div class="grid"><div><h3>Examples nearest the full-data cluster center</h3>
<div id="examples"></div></div><div><h3>Characteristic inputs</h3><div id="features"></div></div></div>
<p class="muted">Characteristic differences are standardized associations with profile membership.
Coverage, calendar era and repeated source fields can influence clusters. They do not establish causation.</p>
<p><a href="../index.html">Compare all profile methods</a> · <a href="../conditional/index.html">Chronological forecast cohorts</a></p>
<script>const data=__DATA__;const $=id=>document.getElementById(id);let selectedYear=null;
function esc(v){return String(v).replace(/[&<>\"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',"'":'&#39;'}[c]));}
function number(v){return v===null||v===undefined?'—':Number.isInteger(v)?String(v):v.toFixed(1);}
function search(){const q=$('search').value.toLowerCase(),old=$('player').value;
const people=data.players.filter(p=>p.name.toLowerCase().includes(q));$('player').innerHTML='';
people.forEach(p=>$('player').add(new Option(p.name+' · '+p.id,p.id)));
if(people.some(p=>p.id===old))$('player').value=old;selectedYear=null;draw();}
function draw(){const player=data.players.find(p=>p.id===$('player').value),h=$('horizon').value,v=$('variant').value;
if(!player){$('heading').textContent='No player matches this search.';$('history').innerHTML='';
$('cohortHeading').textContent='';$('examples').innerHTML='';$('features').innerHTML='';return;}
const history=[...(data.histories[h][player.id]||[])].sort((a,b)=>b.year-a.year);
if(!history.some(r=>r.year===selectedYear))selectedYear=history[0]?.year;
$('heading').textContent=player.name+' · '+history.length+' recorded player-seasons · click a year for cohort details';
const columns=['year','team','age','profile',...data.stats];
$('history').innerHTML='<tr>'+columns.map(c=>'<th>'+esc(c.replaceAll('_',' '))+'</th>').join('')+'</tr>'+
history.map(r=>'<tr data-year="'+r.year+'" class="'+(r.year===selectedYear?'active':'')+'">'+
[r.year,r.team||'—',number(r.age),r.profiles[v],...data.stats.map(s=>number(r.observed[s]))].map(x=>'<td>'+esc(x)+'</td>').join('')+'</tr>').join('');
for(const row of document.querySelectorAll('#history tr[data-year]'))row.onclick=()=>{selectedYear=Number(row.dataset.year);draw();};
const current=history.find(r=>r.year===selectedYear);if(!current){$('cohortHeading').textContent='No candidate history for this horizon.';
$('examples').innerHTML='';$('features').innerHTML='';return;}
const cohort=data.cohorts[v][h].find(c=>c.profile===current.profiles[v]);
$('cohortHeading').textContent=selectedYear+' · local full-data profile '+cohort.profile+' · '+cohort.rows+
' player-seasons / '+cohort.unique_players+' players';
$('examples').innerHTML='<ul>'+cohort.exemplars.map(e=>'<li>'+esc((e.player_display_name||e.player_id)+' · '+e.year+' · '+(e.team||''))+'</li>').join('')+'</ul>';
$('features').innerHTML='<table><tr><th>Input</th><th>Profile mean</th><th>Difference (SD)</th></tr>'+
cohort.characteristics.map(c=>'<tr><td>'+esc(c.feature)+'</td><td>'+number(c.mean)+'</td><td>'+
(c.standardized_difference>=0?'+':'')+c.standardized_difference.toFixed(2)+'</td></tr>').join('')+'</table>';}
$('search').oninput=search;for(const id of ['player','horizon','variant'])$(id).onchange=()=>{selectedYear=null;draw();};
search();</script></body></html>"""
    return page.replace("__DATA__", payload)


def build(out, profile_root):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    data = browser_data(profile_root)
    (out / "data.json").write_text(json.dumps(data, allow_nan=False) + "\n")
    (out / "index.html").write_text(_render(data))
    shutil.copyfile(__file__, out / "browser_source.py")
    json_write(
        out / "manifest.json",
        {
            "status": "complete",
            "players": len(data["players"]),
            "descriptive_player_seasons": {
                h: sum(len(rows) for rows in histories.values())
                for h, histories in data["histories"].items()
            },
            "profile_manifest_sha256": data["profile_manifest_sha256"],
            "source_outcomes": data["source_outcomes"],
            "new_fits": 0,
            "files": {p.name: digest(p) for p in out.iterdir() if p.is_file()},
        },
    )
    return out / "index.html"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="rb_gpu_20261003_001")
    parser.add_argument("--profile-root", type=Path, default=RUNS / "rb_gpu_profiles_20261003_002")
    args = parser.parse_args()
    print(build(RUNS / args.run_id / "profiles/players", args.profile_root))
