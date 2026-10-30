"""Render evaluated evidence and prospective status without refitting any model."""

import argparse
import html
import json
from pathlib import Path

import pandas as pd

from ..weekly_v31.artifacts import digest
from .artifacts import DEFAULT_INTEGRATION, DEFAULT_LOG, ROOT, check


def table(path, groups=None, cohort=None):
    frame = pd.read_csv(path)
    if cohort is not None:
        frame = frame[frame.cohort.eq(cohort)]
    if groups:
        values = [
            c
            for c in (
                "rmse",
                "mae",
                "bias",
                "crps",
                "coverage80",
                "width80",
                "interval_score80",
                "spearman",
            )
            if c in frame
        ]
        frame = frame.groupby(groups)[values].mean().reset_index()
    return frame.to_html(index=False, border=0, float_format=lambda x: f"{x:.4f}")


def report(
    integration=DEFAULT_INTEGRATION,
    log=DEFAULT_LOG,
    output=ROOT / "data/outputs/opening_drive_v4/index.html",
):
    integration, log, output = map(Path, (integration, log, output))
    receipt = check(integration)
    horizons = Path(receipt["dependencies"]["horizons"]["path"])
    uncertainty = Path(receipt["dependencies"]["uncertainty"]["path"])
    status_path = log / "review.json"
    if not status_path.exists():
        status_path = log / "scorecard.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {"status": "pending"}
    sections = [
        ("Matched v3.2 bridge: Week 2", table(horizons / "v32_bridge_scores.csv")),
        ("Horizons across origins 1–16", table(horizons / "scores.csv", ["position", "method"])),
        (
            "Weekly distributions",
            table(uncertainty / "scores.csv", ["horizon", "position", "method"], "all"),
        ),
        (
            "Isolated retraining",
            table(uncertainty / "retraining_scores.csv", ["horizon", "method"]),
        ),
        (
            "Whole-horizon distributions",
            table(integration / "horizon_uncertainty_scores.csv", ["position", "method"]),
        ),
        (
            "Bridge distribution comparison",
            table(integration / "bridge_uncertainty_scores.csv", ["position", "method"]),
        ),
        ("Bridge calibration support", table(integration / "bridge_uncertainty_support.csv")),
        (
            "Component forecasts",
            table(horizons / "component_scores.csv", ["position", "horizon", "target"]),
        ),
    ]
    content = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Opening Drive v4 review</title>
<style>body{font:16px system-ui;max-width:1200px;margin:32px auto;padding:0 20px;color:#142838}
table{border-collapse:collapse;font-size:14px}
td,th{padding:8px;border-bottom:1px solid #ddd;text-align:right}
section{overflow:auto;margin:32px 0}pre{white-space:pre-wrap;background:#f3f5f7;padding:16px}
.notice{border-left:4px solid #b76b00;padding:16px;background:#fff7e8}</style>
<h1>Opening Drive v4 candidate</h1><p class="notice"><b>V3.2 remains the default.</b>
V4 restores component forecasts and extends v3-based modeling across horizons.
Historical gains are mixed; future validation remains pending.</p>
<p>The weekly calibration selector retained v3.2's distributions. Residual retraining is a
separate challenger and is not used in published v4 means. Week 2 matched bridge results
regress for RB, WR and TE. Other origins compare against fixed-origin v3 persistence;
they do not establish an improvement over the shipped v3.2 bridge.</p>
<p>Aggregate tables average the displayed evaluation cells equally. Full CSVs preserve
position, origin, season and cohort results. Historical years were reused in development.
Whole-total distributions model dependent total errors; the original bridge had no such ranges.
Pregame transfer of early-week total calibration and bye-origin horizon predictions are
not separately historically validated. Point and component estimates are separate models.</p>"""
    content += (
        "<h2>Prospective evidence</h2><pre>" + html.escape(json.dumps(status, indent=2)) + "</pre>"
    )
    for title, body in sections:
        content += f"<section><h2>{html.escape(title)}</h2>{body}</section>"
    content += (
        "<h2>Release requirements</h2><pre>"
        + html.escape((integration / "release_policy.json").read_text())
        + "</pre>"
    )
    content += (
        f"<p>Integration SHA-256: <code>{digest(integration / 'manifest.json')}</code></p></html>"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--integration", type=Path, default=DEFAULT_INTEGRATION)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data/outputs/opening_drive_v4/index.html"
    )
    args = parser.parse_args()
    print(report(args.integration, args.log, args.output))
