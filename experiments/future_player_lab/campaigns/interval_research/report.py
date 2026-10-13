"""Local interactive review of a completed width experiment."""
# ruff: noqa: E501

from __future__ import annotations

import json


def interpretation(scores):
    table = scores[scores.season.eq("all") & scores.level.eq(80)]

    def row(position, method, cohort="all"):
        return table[
            table.position.eq(position) & table.method.eq(method) & table.cohort.eq(cohort)
        ].iloc[0]

    old = row("QB", "v31", "1_to_5")
    guarded = row("QB", "selective_guarded", "1_to_5")
    fine = row("QB", "tail_scaled_fine", "1_to_5")
    lineup = row("LINEUP", "v31")
    half = row("LINEUP", "half_v31")
    return f"""<section><h2>What this investigation found</h2>
<p><b>There is modest room to improve player precision.</b> The earlier-season selection policy
with a coverage requirement improves the pooled 80% interval score in all four positions,
but the improvements are small. It retains the large mid-projection QB and high-projection TE
coverage gains. This does not establish that every subgroup improves.</p>
<p><b>Simply allowing ranges to narrow was not the main fix.</b> The signed width-scaled
variant barely changes the QB, RB and WR results. Different corrections for different groups,
and changes to how the endpoints move, matter more in this run than removing the expansion-only rule.</p>
<p><b>A broad group can hide a serious coverage gap.</b> QBs projected at 1–5 points have
{old.coverage:.1%} coverage under v3.1 and only {guarded.coverage:.1%} under the guarded selection
policy. The fixed variant with finer low-projection groups reaches {fine.coverage:.1%},
with average width {fine.width:.2f} versus {old.width:.2f} points and interval score
{fine.interval_score:.2f} versus {old.interval_score:.2f}. This is an example where paying a little
more interval cost may be worthwhile. The selector's guard currently checks the broader below-5
group, so it does not protect this smaller subgroup.</p>
<p><b>Lineups need a separate decision.</b> The adaptive selection policy worsens lineup
interval score. A fixed half-strength correction is a promising exploratory alternative:
{half.coverage:.1%} coverage with {half.width:.2f}-point width, compared with
{lineup.coverage:.1%} and {lineup.width:.2f} for v3.1. Choosing it after seeing these results
would require fresh validation; it is not an already-proven prospective winner.</p>
<p><b>Recommended follow-up:</b> test a player policy whose coverage checks also protect the
1–5-point QB subgroup, and validate lineup correction separately. Preserve the current prospective
trial and version any new contender explicitly. The full tables below include alternatives that
did not work as well as those that look promising.</p></section>"""


NAMES = {
    "v3": "Original v3",
    "v31": "Locked v3.1",
    "half_v31": "Half of v3.1 expansion",
    "scaled_signed": "Allow widening or narrowing",
    "additive_signed": "Additive widening or narrowing",
    "tail_scaled": "Separate tails, width-scaled",
    "tail_additive": "Separate tails, additive",
    "tail_scaled_recent": "Separate tails, recent 2 years",
    "tail_scaled_fine": "Separate tails, finer low groups",
    "selective": "Prior-score selection",
    "selective_guarded": "Prior selection with coverage guard",
}


def report(directory, scores, comparisons, diagnosis):
    primary = scores[scores.season.eq("all") & scores.level.eq(80) & scores.cohort.eq("all")]
    rows = []
    for position, group in primary.groupby("position"):
        base = group[group.method.eq("v31")].iloc[0]
        for method in ("v3", "v31", "tail_additive", "selective_guarded"):
            row = group[group.method.eq(method)].iloc[0]
            rows.append(
                f"<tr><td>{position}</td><td>{NAMES[method]}</td>"
                f"<td>{row.coverage:.1%}</td><td>{row.width:.2f}</td>"
                f"<td>{row.interval_score:.2f}</td>"
                f"<td>{(row.interval_score / base.interval_score - 1):+.1%}</td></tr>"
            )
    lead = []
    for position, group in primary.groupby("position"):
        base = group[group.method.eq("v31")].iloc[0]
        selected = group[group.method.eq("selective_guarded")].iloc[0]
        change = (selected.interval_score / base.interval_score - 1) * 100
        lead.append(
            f"<li><b>{position}:</b> the policy chosen using earlier seasons changes average "
            f"80% width from {base.width:.2f} to {selected.width:.2f} points, coverage from "
            f"{base.coverage:.1%} to {selected.coverage:.1%}, and interval score by {change:+.1f}%.</li>"
        )
    diagnosis_view = diagnosis[diagnosis.cohort.isin(["all", "5_to_10", "10_plus"])]
    diagnosis_cols = [
        "position",
        "cohort",
        "added_width",
        "saved_below_penalty",
        "saved_above_penalty",
        "net_score_cost",
    ]
    decomposition = diagnosis_view[diagnosis_cols].to_html(
        index=False, border=0, float_format=lambda v: f"{v:.3f}"
    )
    bootstrap = comparisons[
        comparisons.method.eq("selective_guarded")
        & comparisons.level.eq(80)
        & comparisons.cohort.eq("all")
    ]
    bootstrap = bootstrap[
        [
            "position",
            "season_equal_delta",
            "ci95_lower",
            "ci95_upper",
            "seasons_improved",
            "seasons",
        ]
    ]
    boot_table = bootstrap.to_html(index=False, border=0, float_format=lambda v: f"{v:.3f}")
    data = scores[scores.season.eq("all")].to_dict("records")
    payload = json.dumps(data, allow_nan=False).replace("</", "<\\/")
    page = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Opening Drive: interval width research</title>
<style>body{font:16px system-ui;margin:0;background:#edf2f4;color:#172d3d}
main{max-width:1200px;margin:auto;padding:30px}section{background:#fff;padding:25px;margin:22px 0;border-radius:10px}
h1{font-size:36px}h2{font-size:24px}p,li{max-width:1000px;line-height:1.65}.notice{border-left:4px solid #bf832e;padding-left:18px}
table{border-collapse:collapse;font-size:13px;width:100%}td,th{padding:9px;border-bottom:1px solid #dfe5e8;text-align:right}
td:first-child,th:first-child{text-align:left}.scroll{overflow:auto}select{padding:8px;margin:5px;font:inherit}
label{display:inline-block;margin:6px 10px 6px 0}svg{width:100%;max-height:440px}a{color:#087d88}
.legend{display:flex;flex-wrap:wrap;gap:8px;font-size:12px}.legend span{padding:4px 8px;background:#f1f4f5}
@media(max-width:600px){main{padding:12px}section{padding:14px}h1{font-size:28px}}</style></head><body><main>
<h1>Where does extra interval width help?</h1>
<p>This completed research experiment examines the costs of v3.1's wider ranges and tests more selective corrections.
All point forecasts and scoring distributions stay fixed. The existing v3/v3.1 prospective trial is preserved.</p>
<p class="notice">Historical development evidence: these seasons have been used before. Candidate comparisons are exploratory,
not untouched validation. Methods and selection rules were recorded before this run; no candidate is automatically deployed.</p>
<section><h2>What the chronological selection policy achieved</h2><p>The coverage-guarded policy chooses a method separately for
each position/scoring group using only the previous two validation seasons. It minimizes interval score among candidates whose
earlier coverage was within two percentage points below the target at each interval level. Future coverage is not guaranteed.</p><ul>"""
    page += "".join(lead) + "</ul></section>"
    page += interpretation(scores)
    page += """<section><h2>Why these alternatives might help</h2><ul>
<li><b>Expansion only:</b> v3.1 cannot narrow a range, even if an earlier calibration group over-covered.</li>
<li><b>Equal padding:</b> it widens both ends equally, although misses can be concentrated on one side.</li>
<li><b>Width scaling:</b> a wide forecast receives more additional points than a narrow forecast. Additive corrections test that assumption.</li>
<li><b>Pooling and drift:</b> projected points below 5 combine many near-zero players with more active players. Finer groups and two-year history test those choices.</li>
<li><b>Selective correction:</b> earlier validation can retain original v3 where a correction does not earn its extra width.</li>
</ul><p>These are tested explanations, not assumptions that every alternative improves every cohort.</p></section>
<section><h2>Coverage versus precision explorer</h2><p>Select a cohort to compare every method on the same player-weeks.
Lower width is useful only when coverage remains adequate. The dashed line marks nominal coverage.
Lower interval score is better; it penalizes both width and missed outcomes.</p>
<label>Position <select id="position"><option>QB</option><option>RB</option><option>WR</option><option>TE</option><option>LINEUP</option></select></label>
<label>Cohort <select id="cohort"></select></label>
<label>Interval <select id="level"><option value="80">80%</option><option value="90">90%</option></select></label>
<svg id="chart" viewBox="0 0 960 420" role="img" aria-label="Interval width and coverage comparison"></svg>
<div class="legend" id="legend"></div><div class="scroll" id="table"></div>
<p>Players: 2021–2025. Artificial offense lineups: 2023–2025, because 2021 supplies calibration and 2022 the first validation year.
Lineup results here cover both fixed roster designs; no K/DST or prospective results enter this research.</p></section>
<section><h2>Key comparisons</h2><p>The tail-additive method is a fixed hypothesis shown alongside the earlier-season selection policy;
it is not labeled an out-of-sample winner chosen after seeing this table. Scores below pool rows; season-level downloads and the
paired comparisons below show stability across years.</p><div class="scroll"><table><thead><tr><th>Position</th><th>Method</th>
<th>Coverage</th><th>Width</th><th>Interval score</th><th>Score vs v3.1</th></tr></thead><tbody>"""
    page += "".join(rows) + "</tbody></table></div></section>"
    page += """<section><h2>Where v3.1's extra width went</h2><p>
<b>Net score cost = added width − saved penalty below − saved penalty above.</b>
A positive net cost means widening was not fully repaid by reduced misses under this scoring rule.
The savings are in interval-score units, not percentages or extra expected fantasy points.</p><div class="scroll">"""
    page += decomposition + "</div></section>"
    page += """<section><h2>How stable is the guarded policy's gain?</h2><p>Negative deltas favor the research policy over v3.1.
The interval resamples whole seasons, preserving dependence within each season. There are only five player seasons and three lineup
seasons; these descriptive intervals do not correct for reused research years or multiple comparisons.</p><div class="scroll">"""
    page += boot_table + "</div></section>"
    page += """<section><h2>What happens next</h2><p>Use this evidence to choose a separately versioned prospective challenger.
Require the large QB/TE coverage improvements to persist while monitoring width and interval score for starters, low-projection players
and complete offense lineups. Existing pregame records are not replaced by this analysis. An 80% central interval does not have to contain
exactly 80% of discrete outcomes, especially when many players score zero; narrower is not automatically better.</p>
<p>The adjusted intervals remain separate from distribution quantiles and means. Negative endpoints are not clipped: the original scoring
contract permits negative scores. No availability mixture or new point estimator was fitted.</p></section>
<section><h2>Evidence and reproduction</h2><p><a href="scores.csv">All scores by season and cohort</a> ·
<a href="comparisons.csv">Paired differences and bootstrap</a> · <a href="diagnosis.csv">Cost decomposition</a> ·
<a href="players_examples.csv">Player examples</a> · <a href="lineups_examples.csv">Lineup examples</a> ·
<a href="protocol.json">Recorded experimental design</a> · <a href="manifest.json">Run receipt</a></p>
<p>Methods draw on <a href="https://arxiv.org/abs/1905.03222">Conformalized Quantile Regression</a> and the
<a href="https://arxiv.org/abs/2005.12881">decomposition of interval forecast scores</a>.
The chronological variants here do not inherit an unconditional coverage guarantee.</p></section>
<script>const data=PAYLOAD;const names=NAMES;const colors=['#526778','#087f8c','#a87516','#b33b50','#4169ae','#724aa3','#199668','#bf6330','#738b22','#b14998','#142b39'];
const p=document.getElementById('position'),c=document.getElementById('cohort'),l=document.getElementById('level');
function cohorts(){const previous=c.value;const values=[...new Set(data.filter(r=>r.position===p.value).map(r=>r.cohort))];
c.innerHTML=values.map(v=>`<option>${v}</option>`).join('');c.value=values.includes(previous)?previous:'all';if(p.value==='LINEUP')l.value='80';l.disabled=p.value==='LINEUP';render();}
function render(){const rows=data.filter(r=>r.position===p.value&&r.cohort===c.value&&r.level===Number(l.value));
const target=Number(l.value)/100;const minX=Math.max(0,Math.min(...rows.map(r=>r.width))*.9),maxX=Math.max(...rows.map(r=>r.width))*1.1;
const minY=Math.max(0,Math.min(target,...rows.map(r=>r.coverage))-.04),maxY=Math.min(1,Math.max(target,...rows.map(r=>r.coverage))+.04);
const x=v=>80+(v-minX)/(maxX-minX||1)*800,y=v=>350-(v-minY)/(maxY-minY||1)*300;
let s=`<line x1="80" x2="880" y1="350" y2="350" stroke="#87959c"/><line x1="80" x2="80" y1="50" y2="350" stroke="#87959c"/>`;
for(let i=0;i<=4;i++){let vx=minX+(maxX-minX)*i/4,vy=minY+(maxY-minY)*i/4;
s+=`<text x="${x(vx)}" y="375" text-anchor="middle" font-size="12">${vx.toFixed(1)}</text><text x="67" y="${y(vy)+4}" text-anchor="end" font-size="12">${(vy*100).toFixed(0)}%</text>`;}
s+=`<line x1="80" x2="880" y1="${y(target)}" y2="${y(target)}" stroke="#b08121" stroke-dasharray="6 4"/><text x="480" y="410" text-anchor="middle">Average interval width (points)</text>`;
rows.forEach(r=>{const i=Object.keys(names).indexOf(r.method);s+=`<circle cx="${x(r.width)}" cy="${y(r.coverage)}" r="7" fill="${colors[i]}" opacity=".85"><title>${names[r.method]}: width ${r.width.toFixed(2)}, coverage ${(r.coverage*100).toFixed(1)}%, score ${r.interval_score.toFixed(2)}</title></circle>`;});
document.getElementById('chart').innerHTML=s;document.getElementById('legend').innerHTML=rows.map(r=>`<span style="border-left:4px solid ${colors[Object.keys(names).indexOf(r.method)]}">${names[r.method]}</span>`).join('');
document.getElementById('table').innerHTML='<table><thead><tr><th>Method</th><th>Rows</th><th>Coverage</th><th>Width</th><th>Score</th><th>Miss below</th><th>Miss above</th></tr></thead><tbody>'+rows.sort((a,b)=>a.interval_score-b.interval_score).map(r=>`<tr><td>${names[r.method]}</td><td>${r.n}</td><td>${(r.coverage*100).toFixed(1)}%</td><td>${r.width.toFixed(2)}</td><td>${r.interval_score.toFixed(2)}</td><td>${(r.miss_below*100).toFixed(1)}%</td><td>${(r.miss_above*100).toFixed(1)}%</td></tr>`).join('')+'</tbody></table>';}
p.addEventListener('change',cohorts);c.addEventListener('change',render);l.addEventListener('change',render);cohorts();</script></main></body></html>"""
    page = page.replace("PAYLOAD", payload).replace("NAMES", json.dumps(NAMES))
    (directory / "index.html").write_text(page)


if __name__ == "__main__":
    import argparse
    import shutil
    from pathlib import Path

    import pandas as pd

    from ..weekly_v31.artifacts import digest, now, verify, write
    from .run import DEFAULT_REPORT, DEFAULT_RUN

    parser = argparse.ArgumentParser(description="Render a review from sealed experiment scores")
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    verify(args.run)
    args.report.mkdir(parents=True, exist_ok=True)
    tables = {
        name: pd.read_csv(args.run / f"{name}.csv")
        for name in ("scores", "comparisons", "diagnosis")
    }
    report(args.report, **tables)
    for name in (
        "scores.csv",
        "comparisons.csv",
        "diagnosis.csv",
        "protocol.json",
        "manifest.json",
        "players_examples.csv",
        "lineups_examples.csv",
    ):
        shutil.copy2(args.run / name, args.report / name)
    write(
        args.report / "report_receipt.json",
        dict(
            rendered_at=now().isoformat(),
            source_run_manifest_sha256=digest(args.run / "manifest.json"),
            renderer_sha256=digest(Path(__file__)),
            html_sha256=digest(args.report / "index.html"),
            note="Presentation and interpretation rendered from unchanged sealed research tables",
        ),
    )
