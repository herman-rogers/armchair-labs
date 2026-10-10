"""A self-contained local ranking report with no application deployment."""

import html
import json


def expert_panel(rankings, comparison):
    matched = rankings.filter(rankings["expert_rank"].is_not_null())
    n = comparison["matched"]
    dots = []
    for r in matched.iter_rows(named=True):
        x = 48 + (r["model_matched_rank"] - 1) / max(n - 1, 1) * 370
        y = 30 + (r["expert_matched_rank"] - 1) / max(n - 1, 1) * 260
        label = html.escape(
            f"{r['player_display_name']}: model #{r['rank']}, experts #{r['expert_rank']:g}"
        )
        color = "#087b77" if r["matched_rank_gap"] >= 0 else "#8253a3"
        dots.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}" '
            f'opacity=".75" tabindex="0" aria-label="{label}"><title>{label}</title></circle>'
        )
    overlap = comparison["top_overlap"]
    missing_top = overlap["24"]["model_names_without_expert_rank"]
    missing_text = html.escape(", ".join(missing_top)) if missing_top else "None"
    unmatched = ", ".join(f"{r['name']} (#{r['rank']})" for r in comparison["unmatched_experts"])
    rho = f"{comparison['spearman']:.3f}" if comparison["spearman"] is not None else "Undefined"
    url = html.escape(comparison.get("source_url") or "https://www.fantasypros.com/", quote=True)
    return f'''<section class="card" id="expert-comparison">
<h2>Model vs. expert consensus</h2>
<p><a href="{url}">FantasyPros rest-of-season PPR</a> ·
{html.escape(comparison["source_date"])} · {comparison["total_experts"]} included experts ·
{n} matched of {comparison["model_count"]} modeled and
{comparison["expert_count"]} expert-listed RBs.</p>
<p class="notice">{html.escape(comparison["date_warning"])}</p>
<div class="metrics">
<div><b>{rho}</b><span>Rank correlation<br>1.0 = identical ordering</span></div>
<div><b>{comparison["mean_absolute_matched_rank_gap"]:.1f}</b>
<span>Mean absolute rank gap<br>among the {n} matched RBs</span></div>
<div><b>{overlap["12"]["shared"]}/{overlap["12"]["model_count"]}</b>
<span>Shared top 12<br>original full lists</span></div>
<div><b>{overlap["24"]["shared"]}/{overlap["24"]["model_count"]}</b>
<span>Shared top 24<br>original full lists</span></div></div>
<div class="comparison-grid"><div>
<svg viewBox="0 0 470 345" role="img" aria-label="Model and expert ranks among matched RBs">
<title>Matched-player ranks; points near the diagonal indicate agreement</title>
<path d="M48 30 V290 H418" fill="none" stroke="#738496"/>
<path d="M48 30 L418 290" stroke="#97a6b5" stroke-dasharray="5 5"/>
<text x="48" y="312" text-anchor="middle">1</text>
<text x="418" y="312" text-anchor="middle">{n}</text>
<text x="34" y="35" text-anchor="end">1</text>
<text x="34" y="294" text-anchor="end">{n}</text>
<text x="230" y="336" text-anchor="middle">Model rank within matched players</text>
<text x="16" y="165" text-anchor="middle" transform="rotate(-90 16 165)">Expert rank</text>
{"".join(dots)}</svg></div><div>
<p>Near the diagonal = agreement. Hover or focus a dot to see the player and original ranks.</p>
<p>Correlation, mean gap and the chart rerank the same {n} players in both lists.
The table shows each original rank; its gaps also reflect differences in list coverage.</p>
<p><strong>Gap = expert rank − model rank.</strong><br>
<span class="higher">Positive: model ranks higher.</span><br>
<span class="lower">Negative: experts rank higher.</span></p>
<p>Model top-24 players absent from the expert list: <strong>{missing_text}</strong>.
An absent rank stays blank; it is not assigned last place.</p>
<details><summary>Expert-listed players absent or unresolved in the model</summary>
<p>{html.escape(unmatched) or "None"}</p></details></div></div></section>'''


def write_report(output, rankings, summary):
    comparison = summary.get("expert_comparison")
    rows = []
    for row in rankings.iter_rows(named=True):
        cells = [
            str(row["rank"]),
            html.escape(row["player_display_name"] or row["player_id"]),
            html.escape(row["team"] or "Unknown"),
            f"{row['predicted_ppr']:.1f}",
            f"{row['ppr_per_scheduled_game']:.1f}"
            if row["ppr_per_scheduled_game"] is not None
            else "—",
            f"{row['ppr_selected']:.1f}",
            f"{row['ppr_ensemble_diverse']:.1f}",
            str(row["history_games"]),
            str(row["scheduled_games"] or "Unknown"),
        ]
        attrs = f'data-rank="{row["rank"]}"'
        if comparison:
            er, gap = row["expert_rank"], row["rank_gap"]
            expert_text = f"{er:g}" if er is not None else "—"
            if row["expert_team_differs"]:
                expert_text += (
                    ' <span title="Expert snapshot team: '
                    + html.escape(row["expert_team"] or "Unknown", quote=True)
                    + '; differs from model cutoff team">*</span>'
                )
            gap_class = "higher" if gap is not None and gap > 0 else "lower" if gap else ""
            gap_text = "—" if gap is None else (f'<span class="{gap_class}">{gap:+g}</span>')
            cells[3:3] = [expert_text, gap_text]
            attrs += f' data-expert="{er if er is not None else 999999}"'
            attrs += f' data-gap="{abs(gap) if gap is not None else -1}"'
            attrs += f' data-matched="{1 if er is not None else 0}"'
        rows.append(f"<tr {attrs}>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    cutoff = summary["origin"]
    withheld = [a["source"] for a in summary["supplements"]["sources"] if a["status"] != "used"]
    title = f"{summary['season']} {summary['position']} model preview · after Week {cutoff}"
    document = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title><style>
body{font:16px/1.5 system-ui,sans-serif;background:#f5f7fa;color:#172334;margin:0}
main{max-width:1120px;margin:auto;padding:32px 20px}h1{line-height:1.2}
.card{background:white;border:1px solid #d8e0e8;border-radius:10px;padding:20px;margin:16px 0}
.note{color:#4d5e70}input{font:inherit;padding:10px;width:min(380px,90%);
border:1px solid #aab8c5;border-radius:6px}
.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14px}
th,td{padding:10px;text-align:right;border-bottom:1px solid #e3e8ee;white-space:nowrap}
th{background:#edf2f7}td:nth-child(2),th:nth-child(2){text-align:left}
tbody tr:hover{background:#edf5fc}
a{color:#16588b}summary{cursor:pointer}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}
.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:20px 0}
.metrics>div{background:#edf2f7;border-radius:8px;padding:14px}
.metrics b{font-size:27px;display:block}.metrics span{font-size:13px;color:#4d5e70}
.notice{padding:12px;background:#fff3d7;border-radius:6px}
.comparison-grid{display:grid;grid-template-columns:1fr 1fr;gap:24px}
svg{width:100%;font:12px system-ui}circle:hover,circle:focus{r:7;opacity:1;stroke:#172334}
.higher{color:#087b77}.lower{color:#8253a3}.controls{display:flex;gap:12px;flex-wrap:wrap}
select{font:inherit;padding:9px;border:1px solid #aab8c5;border-radius:6px;background:white}
@media(max-width:650px){.metrics{grid-template-columns:1fr 1fr}.comparison-grid{display:block}}
</style><main><h1>__TITLE__</h1>
<p>Saved tree-model predictions for Weeks __START__–18, ranked by expected remaining-season PPR.</p>
__EXPERT_PANEL__
<div class="card"><strong>Reconstructed Week __CUTOFF__ view · research preview</strong>
<p>These predictions use observations through Week __CUTOFF__, with models trained through 2025.
They were generated on __ISSUED__ using later source captures. They are not a forecast issued
at the original cutoff or a ranking updated for subsequent injuries, transactions or games.</p>
<p>The primary forecast is the saved equal-weight top-three ensemble. “Single” and “Diverse”
are the saved alternative policies; differences show model sensitivity, not confidence intervals.
No model was retrained or selected using 2026 results.</p>
<p class="note">PPR here includes rushing/receiving yards, receptions and touchdowns;
it excludes fumbles, passing and bonuses. Per-game values divide by scheduled remaining games,
not an estimate of games the player will actually play. Teams are last observed by the cutoff.
All candidates are retained, including players without a 2026 box-score appearance.</p></div>
<div class="card"><label for="search">Find a player or team</label><br>
<div class="controls">
<input id="search" type="search" placeholder="Player or team" autocomplete="off">
__EXPERT_CONTROLS__</div>
<p><a href="rankings.csv">Download rankings CSV</a> · <a href="rankings.json">JSON</a> ·
<a href="coverage.csv">Feature coverage</a> · <a href="manifest.json">Run provenance</a></p>
__EXPERT_NOTE__
<div class="scroll"><table id="rankings"><thead><tr><th>Model rank</th><th>Player</th><th>Team</th>
__EXPERT_HEADINGS__<th>ROS PPR</th>
<th>PPR / game</th><th>Single PPR</th><th>Diverse PPR</th><th>Observed games</th><th>Games left</th>
</tr></thead><tbody>__ROWS__</tbody></table></div></div>
<div class="card"><strong>Inputs and checks</strong>
<p>Raw box scores, offensive snaps, play-by-play, team statistics, NGS and rosters are transformed
using the training pipeline's aggregations. Current-source exclusions: __WITHHELD__.
Current FTN participation remains withheld under the original publication-delay rule.</p>
<p>Historical feature replay: __REPLAY__. Historical 2021–2025 research found remaining-season
PPR RMSE 46.45 versus 51.67 for adjusted persistence, across the broad RB candidate population.
That is prior development evidence, not measured accuracy for this 2026 ranking.</p>
<details><summary>Run details</summary><pre>__DETAILS__</pre></details></div></main>
<script>
const body=document.querySelector('#rankings tbody'), rows=Array.from(body.rows);
function update(){
const q=document.getElementById('search').value.toLowerCase();
const filter=document.getElementById('filter')?.value||'all';
const sort=document.getElementById('sort')?.value||'model';
rows.sort((a,b)=>{
let d=sort==='expert'?Number(a.dataset.expert)-Number(b.dataset.expert):
sort==='gap'?Number(b.dataset.gap)-Number(a.dataset.gap):0;
return d||Number(a.dataset.rank)-Number(b.dataset.rank);});
rows.forEach(r=>{const ok=filter==='all'||(filter==='matched'&&r.dataset.matched==='1')||
(filter==='top24'&&(Number(r.dataset.rank)<=24||Number(r.dataset.expert)<=24));
r.hidden=!ok||!r.textContent.toLowerCase().includes(q);body.appendChild(r);});}
document.getElementById('search').addEventListener('input',update);
document.getElementById('filter')?.addEventListener('change',update);
document.getElementById('sort')?.addEventListener('change',update);
</script></html>"""
    replacements = {
        "TITLE": html.escape(title),
        "START": str(cutoff + 1),
        "CUTOFF": str(cutoff),
        "ISSUED": html.escape(summary["issued_at"]),
        "ROWS": "\n".join(rows),
        "WITHHELD": html.escape(", ".join(withheld) or "none"),
        "REPLAY": html.escape(summary["feature_replay"]["description"]),
        "DETAILS": html.escape(json.dumps(summary, indent=2)),
        "EXPERT_PANEL": expert_panel(rankings, comparison) if comparison else "",
        "EXPERT_HEADINGS": "<th>Expert rank</th><th>Rank gap</th>" if comparison else "",
        "EXPERT_NOTE": (
            '<p class="note">* Expert and model snapshots list different teams. '
            "Matches use unique normalized name and position; missing ranks stay blank.</p>"
        )
        if comparison
        else "",
        "EXPERT_CONTROLS": """<select id="filter" aria-label="Player filter">
<option value="all">All RBs</option><option value="matched">Matched RBs only</option>
<option value="top24">Top 24 in either list</option></select>
<select id="sort" aria-label="Sort rankings"><option value="model">Sort: model rank</option>
<option value="expert">Sort: expert rank</option>
<option value="gap">Sort: largest published rank gap</option></select>"""
        if comparison
        else "",
    }
    for key, value in replacements.items():
        document = document.replace(f"__{key}__", value)
    (output / "index.html").write_text(document)
