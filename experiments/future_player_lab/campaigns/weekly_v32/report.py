"""Review source usability, point accuracy, participation and workload mechanisms."""
# ruff: noqa: E501

from __future__ import annotations

import pandas as pd


def render(directory):
    scores = pd.read_csv(directory / "scores.csv")
    coverage = pd.read_csv(directory / "coverage.csv")
    components = pd.read_csv(directory / "component_scores.csv")
    comparisons = pd.read_csv(directory / "comparisons.csv")
    timestamps = pd.read_csv(directory / "source_timestamps.csv")
    pool = scores[scores.season.eq("all")]
    summary = []
    for (horizon, position), group in pool[pool.cohort.eq("all")].groupby(["horizon", "position"]):
        base = group[group.method.eq("v3")].iloc[0]
        for method in ("v3_out_rule", "mixture_team_blend", "redistribution_half", "selected"):
            row = group[group.method.eq(method)].iloc[0]
            summary.append(
                dict(
                    horizon=horizon,
                    position=position,
                    method=method,
                    v3_rmse=base.rmse,
                    rmse=row.rmse,
                    change_pct=100 * (row.rmse / base.rmse - 1),
                    mae=row.mae,
                )
            )
    summary = pd.DataFrame(summary)

    def table(f):
        return (
            '<div class="scroll">'
            + f.to_html(index=False, border=0, float_format=lambda x: f"{x:.4f}")
            + "</div>"
        )

    page = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Opening Drive v3.2 · Injury, participation and workload</title>
<style>body{{margin:0;background:#f4f7fa;color:#193249;font:16px/1.55 system-ui}}main{{max-width:1250px;margin:auto;padding:28px}}section{{background:white;border:1px solid #d5e0e8;padding:24px;margin:24px 0;border-radius:12px}}h1{{font-size:clamp(28px,4vw,42px);line-height:1.15}}h2{{font-size:23px}}.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:9px;white-space:nowrap;border-bottom:1px solid #d9e2ea;text-align:right}}td:first-child,th:first-child{{text-align:left}}th{{background:#eaf1f6}}select{{padding:8px;max-width:100%}}label{{display:inline-block;margin:8px 14px 8px 0}}a{{color:#155f9b}}code{{overflow-wrap:anywhere}}svg{{width:100%;height:auto}}.note{{color:#576b7c}}@media(max-width:600px){{main{{padding:12px}}section{{padding:15px}}}}</style>
<main><p>OPENING DRIVE · V3.2 · BASELINE V3</p><h1>Who will play, how much work will they get, and who benefits from an absence?</h1>
<p>V3.1 is paused and retained. This independent v3-based study tests {scores.method.nunique() - 1} fixed candidates and an earlier-season selector on 2021–2025. It separates offensive participation, conditional workload and teammate redistribution.</p>
<section><h2>The data gap is real—and depends on forecast timing</h2>
<p>The original frozen v3 panel has 3,469 2026 rows for Weeks 1–5, with zero known dated injury reports. The underlying table has 2025/2026 reports, but their modification timestamps are missing. This study excludes those undated rows. Separately corroborated, timestamped 2025 article reports can be used only after their recorded update time.</p>
<p><b>Early week</b> uses the original v3 cutoff. <b>Pregame 60m</b> updates injury evidence one hour before the player's game while retaining the early-week baseline and other pregame features. Later-arm gains include the value of later information; they are not a same-information comparison with v3.</p>
<p>V3's original injury join could carry an earlier week's report forward within the season. The new join requires the same player, pregame team, season and game week, a usable timestamp, and age ≤7 days. Unknown is never encoded as healthy. Historical report timestamps reconstruct what could have been known; they are not proof that our system captured the report then.</p>
<h3>Original source timestamps</h3>{table(timestamps[timestamps.season.ge(2019)])}
<h3>Usable reports and workload labels in the study</h3>{table(coverage[coverage.season.ge(2021)])}
<p class="note">Study counts exclude scheduled byes and retain only rows with saved v3 means, hence fewer than the 3,469 original 2026 rows. No fresh provider capture is backfilled into those historical rows. An all-team immutable provider collector is supplied for observations going forward.</p></section>
<section><h2>Point-prediction results</h2><p>Lower RMSE is better. Negative percentage changes improve on v3. The selector chooses separately by position using the preceding two validation seasons; it can choose v3. The Out-only rule isolates the effect of setting a fresh same-game reported-Out player to zero.</p>{table(summary)}</section>
<section><h2>Compare every method on the same rows</h2><label>Timing <select id="horizon"><option>early_week</option><option>pregame_60m</option></select></label><label>Position <select id="position"><option>QB</option><option>RB</option><option>WR</option><option>TE</option></select></label><label>Cohort <select id="cohort"></select></label><div id="plot"></div><div class="scroll" id="table"></div>
<p class="note">Report-known and teammate-out cohorts depend on information available at that horizon. Compare methods within a horizon/cohort on identical rows; do not compare different cohort compositions as if they were matched.</p></section>
<section><h2>Participation and workload are evaluated separately</h2>
<p>“Played” here means evidence of offensive snaps, offensive opportunities or scoring. It does not mean officially active on the game-day roster, nor does absence establish an injury. Unsupported labels remain unknown. Prior four scheduled games supply snaps, snap share and usage, with no current-game outcomes entering features.</p>
<p>The mixture computes P(offensive participation) × expected points conditional on participation. The workload model predicts conditional attempts, carries and targets, passes them through a linear conditional points model, and applies the participation probability once. Lower Brier/log loss and lower workload RMSE are better.</p>
{table(components[components.season.eq("all") & components.stage.isin(["no_injury", "team", "redistribution_half"])])}</section>
<section><h2>How teammate redistribution was tested</h2><p>The learned model sees teammates' dated Out/Questionable/practice status weighted by prior usage, excluding the player's own status. Two additional rules redistribute 50% or 100% of a predicted team workload shortfall to teammates with participation probability at least 50%. Allocation follows their baseline expected usage and updated participation probability.</p>
<p>Each transfer is capped at the shortfall relative to the no-injury expected-workload budget. Unassigned volume is recorded, and zero-probability players receive none. The budget covers the v3 eligible player pool, not an assertion about complete team volume. A team-level conserved transfer can still worsen player predictions; its benefit must be demonstrated.</p>
<p><a href="redistribution_budgets.csv">Inspect every workload budget and transfer</a> · <a href="examples.csv">Examples of changed forecasts with an out teammate</a></p></section>
<section><h2>Season stability</h2><p>Descriptive paired whole-season bootstrap intervals, five evaluation seasons, no multiple-testing correction. These years have been reused for research.</p>{table(comparisons[comparisons.method.eq("selected") & comparisons.cohort.eq("all")])}</section>
<section><h2>What is ready and what still needs evidence</h2><p>The research models, source audit, historical predictions and repeatable provider capture are implemented. Models are saved through a 2026 research fit. No API model or existing v3/v3.1 forecast is replaced.</p>
<p>The critical remaining input gap is repeated, dated practice reports and official inactive/active evidence collected before the relevant forecast. A current provider tag—even one labeled Active—is not that evidence. Historical weekly final reports often arrive too late for an early-week forecast. The collector preserves observed time separately from a story's date and never treats missing injury rows as clearance.</p>
<p>Historical snap/box features assume earlier-week results were available by the next forecast week; they are not original publication-vintage archives. Complete practice trajectories, official active-roster labels and within-game injury exits were not reconstructed. K/DST and joint lineup simulation are outside this study.</p>
<p>Upstream context: <a href="https://nflreadr.nflverse.com/articles/nflverse_data_schedule.html">nflverse availability schedule</a> documents an injury-feed gap after 2024. Local undated later rows are therefore not assumed to be usable historical reports.</p></section>
<section><h2>Artifacts and reproduction</h2><p><a href="scores.csv">Point scores</a> · <a href="rankings.csv">Weekly ranking outcomes</a> · <a href="component_scores.csv">Participation/workload scores</a> · <a href="coverage.csv">Coverage</a> · <a href="protocol.json">Protocol</a> · <a href="selection.json">Every selection decision</a></p><p><code>python -m experiments.future_player_lab.campaigns.weekly_v32.run</code><br><code>python -m experiments.future_player_lab.campaigns.weekly_v32.capture --week 6</code></p></section>
<script>const data={pool.to_json(orient="records")};const h=document.getElementById('horizon'),p=document.getElementById('position'),c=document.getElementById('cohort');
function cohorts(){{let old=c.value;let a=[...new Set(data.filter(r=>r.horizon===h.value&&r.position===p.value).map(r=>r.cohort))];c.innerHTML=a.map(v=>`<option>${{v}}</option>`).join('');c.value=a.includes(old)?old:'all';render();}}
function render(){{let rows=data.filter(r=>r.horizon===h.value&&r.position===p.value&&r.cohort===c.value);const W=950,H=330,P=50;let min=Math.min(...rows.map(r=>r.rmse))-.01,max=Math.max(...rows.map(r=>r.rmse))+.01;let x=v=>P+(v-min)/(max-min)*(W-2*P);let svg=`<svg viewBox="0 0 ${{W}} ${{H}}" role="img" aria-label="Point RMSE by method"><path d="M${{P}},${{H-45}}H${{W-P}}" stroke="#668099"/>`;rows.sort((a,b)=>a.rmse-b.rmse).forEach((r,i)=>{{let y=20+i*15;svg+=`<circle cx="${{x(r.rmse)}}" cy="${{y}}" r="5" fill="${{r.method==='selected'?'#cf6428':'#22779a'}}"><title>${{r.method}}: RMSE ${{r.rmse.toFixed(4)}}</title></circle>`;}});svg+=`<text x="${{P}}" y="${{H-15}}">${{min.toFixed(2)}} ← lower RMSE is better → ${{max.toFixed(2)}}</text></svg>`;document.getElementById('plot').innerHTML=svg;document.getElementById('table').innerHTML='<table><thead><tr><th>Method</th><th>Rows</th><th>RMSE</th><th>MAE</th><th>Bias</th></tr></thead><tbody>'+rows.map(r=>`<tr><td>${{r.method}}</td><td>${{r.n}}</td><td>${{r.rmse.toFixed(4)}}</td><td>${{r.mae.toFixed(4)}}</td><td>${{r.bias.toFixed(4)}}</td></tr>`).join('')+'</tbody></table>';}}
h.addEventListener('change',cohorts);p.addEventListener('change',cohorts);c.addEventListener('change',render);cohorts();</script></main></html>"""
    (directory / "index.html").write_text(page)
