"""Render saved comparison reports as one offline browser document; no model execution."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
from urllib.parse import urlsplit

from markdown_it import MarkdownIt

STYLE = """
:root{color-scheme:light}body{margin:0;background:#f1f4f0;color:#172c2a;
font:16px/1.65 system-ui,sans-serif}header,main{max-width:1160px;margin:auto;padding:32px}
header{padding-bottom:18px}header p{max-width:900px}h1{line-height:1.2;font-size:34px}
h2{margin-top:32px;font-size:25px}h3{margin-top:26px}section{background:white;
padding:28px 34px;margin-bottom:28px;border-radius:12px;border-top:5px solid #167866}
.cards{display:grid;grid-template-columns:1fr 1fr;gap:16px}.card{background:#e3ece6;
padding:18px 22px;border-radius:9px}.card strong{font-size:20px;display:block}
.eyebrow{letter-spacing:.14em;font-size:12px;text-transform:uppercase;color:#527267}
nav{display:flex;gap:22px;flex-wrap:wrap;margin-top:24px}a{color:#096b61}
table{border-collapse:collapse;display:block;overflow:auto;width:100%;font-size:14px;
margin:20px 0}th,td{padding:10px 12px;border-bottom:1px solid #dbe5df;
text-align:left;white-space:nowrap}th{background:#edf3ef}code{background:#edf3ef;
padding:2px 5px;border-radius:4px}pre{overflow:auto;padding:16px;background:#edf3ef}
li{margin:5px 0}.note{padding:15px 20px;background:#fff2d9;border-radius:9px}
footer{font-size:13px;color:#52635d} @media(max-width:700px){header,main{padding:18px}
section{padding:20px}.cards{grid-template-columns:1fr}h1{font-size:27px}}
"""


def render_markdown(path: Path) -> str:
    parser = MarkdownIt("commonmark", {"html": False}).enable("table")
    tokens = parser.parse(path.read_text())
    for token in tokens:
        for child in token.children or []:
            if child.type != "link_open":
                continue
            href = child.attrGet("href") or ""
            if href and not href.startswith("#") and not urlsplit(href).scheme:
                parts = urlsplit(href)
                target = (path.parent / parts.path).resolve().as_uri()
                if parts.fragment:
                    target += "#" + parts.fragment
                child.attrSet("href", target)
    return parser.renderer.render(tokens, parser.options, {})


def build(rb_report: Path, qb_report: Path, output: Path) -> Path:
    reports = {"rb": rb_report.resolve(), "qb": qb_report.resolve()}
    sources = {
        key: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for key, path in reports.items()
    }
    sections = "".join(
        f'<section id="{key}">{render_markdown(path)}</section>' for key, path in reports.items()
    )
    source_names = html.escape(", ".join(p.name for p in reports.values()))
    document = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>Campaign comparisons — Armchair Labs</title>"
        f"<style>{STYLE}</style></head><body><header>"
        '<div class="eyebrow">Armchair Labs · Saved model evidence</div>'
        '<h1>Model campaign comparisons</h1><div class="cards">'
        '<div class="card"><strong>RB Exploratory Research</strong>'
        "<code>rb_exploratory_research</code><p>Original RB tree library: "
        "52 candidate recipes and the three-member ensemble. Original fitted weights retained.</p>"
        '</div><div class="card"><strong>Opening Drive v1</strong>'
        "<code>opening_drive_v1</code><p>New RB, QB, and WR fits: four candidate recipes, "
        "two-member ensembles, and dated draft-history inputs.</p></div></div>"
        '<p class="note">Opening Drive reused the RB research methods and base feature pipeline, '
        "then refit the RB weights. Historical accuracy below uses matching players, seasons, "
        "and cutoffs. Current forecast differences compare predictions, not accuracy.</p>"
        '<nav><a href="#rb">RB campaigns</a><a href="#qb">QB versus existing passing model</a>'
        '<a href="sources.json">Report provenance</a></nav></header>'
        f"<main>{sections}<footer>Rendered from saved comparison reports. "
        "No training, model promotion, or legacy deletion occurs when building this view. "
        f"Sources: {source_names}"
        "</footer></main></body></html>"
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "sources.json").write_text(json.dumps(sources, indent=2) + "\n")
    destination = output / "index.html"
    destination.write_text(document)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rb-report", required=True, type=Path)
    parser.add_argument("--qb-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(build(args.rb_report, args.qb_report, args.output).resolve())


if __name__ == "__main__":
    main()
