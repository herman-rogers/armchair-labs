"""Summarize the sealed QB/rookie correction without fitting or changing a bundle."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from engine.chairstats import CampaignStore

from .revise_v2 import CAMPAIGN, PARENT_SHA, write_json


def build(run: Path, registry: Path):
    store = CampaignStore(registry)
    campaign = store.campaign(CAMPAIGN)
    verification = json.loads((run / "export/verification.json").read_text())
    if campaign["manifest_sha256"] != verification["candidate_manifest_sha256"]:
        raise ValueError("Readout must describe the verified candidate revision")
    rows, choices = [], {}
    for model_id in ("qb_season", "qb_remaining"):
        report = store.metrics(CAMPAIGN, model_id)
        choices[model_id] = report["qb_revision"]["final_choices"]
        for target in ("attempts", "passing_yards"):
            for cohort in ("all_candidates", "high_prior_usage"):
                selected = {
                    recipe: next(
                        r
                        for r in report["rows"]
                        if r["target"] == target
                        and r["cohort"] == cohort
                        and r["year"] is None
                        and r["recipe"] == recipe
                    )
                    for recipe in ("control", "ensemble")
                }
                rows.append(
                    {
                        "model_id": model_id,
                        "target": target,
                        "cohort": cohort,
                        "old": selected["control"],
                        "revised_policy": selected["ensemble"],
                    }
                )
    current_coherence = []
    for model_id, origin in (("qb_season", 0), ("qb_remaining", 2)):
        for row in store.forecast(CAMPAIGN, model_id, 2026, origin)["rows"]:
            values = row["predictions"]
            if values["completions"] > values["attempts"] + 1e-6:
                current_coherence.append(
                    {
                        "model_id": model_id,
                        "player_name": row["player_name"],
                        "attempts": values["attempts"],
                        "completions": values["completions"],
                    }
                )
    result = {
        "campaign_id": CAMPAIGN,
        "manifest_sha256": campaign["manifest_sha256"],
        "parent_manifest_sha256": PARENT_SHA,
        "evaluation_years": campaign["evaluation_years"],
        "evaluation": rows,
        "current_2026_choices": choices,
        "current_completion_attempt_inconsistencies": current_coherence,
        "coverage": verification["coverage"],
        "replay": verification["replay"],
        "interpretation": [
            "Evaluation compares policies selected before each forecast year; the 2026 choice "
            "uses 2023–2025 evidence and can differ from earlier choices.",
            "High prior usage here means at least 300 attempts in the preceding season.",
            "Gains are modest and mixed: the prior-window guard does not guarantee better "
            "results in the following season. Review the high-usage results separately.",
            "Seven QB targets, including direct fantasy points, retain their original weights. "
            "Existing RB/WR target forecasts are preserved; recovered rookies can change ranks.",
            f"Targets remain independent: {len(current_coherence)} current QB rows predict "
            "slightly more completions than attempts. Details are retained in the JSON evidence; "
            "no untested constraint adjustment was applied.",
            "2026 origin 2 is a reconstructed after-Week-2 snapshot, not a current-week forecast.",
            "Reused historical development seasons; no untouched test claim.",
        ],
    }
    destination = run / "export"
    write_json(destination / "readout.json", result)

    def table(headers, records):
        return (
            "<table><thead><tr>"
            + "".join(f"<th>{html.escape(str(h))}</th>" for h in headers)
            + "</tr></thead><tbody>"
            + "".join(
                "<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in row) + "</tr>"
                for row in records
            )
            + "</tbody></table>"
        )

    body = "<h1>Opening Drive v2 · QB and rookie correction</h1>"
    body += (
        "<p>The campaign ID stays <code>opening_drive_v2</code>. "
        "This page records the new revision.</p>"
    )
    body += (
        "<h2>Historical comparison, 2021–2025</h2><p>Lower error is better. "
        "These are chronological policy results on all eligible QB candidates.</p>"
    )
    body += table(
        ["Model", "Target", "Old MAE", "Revised MAE", "Old RMSE", "Revised RMSE"],
        [
            [
                r["model_id"],
                r["target"],
                *[
                    f"{r[version][metric]:.2f}"
                    for metric in ("mae", "rmse")
                    for version in ("old", "revised_policy")
                ],
            ]
            for r in rows
            if r["cohort"] == "all_candidates"
        ],
    )
    body += "<h2>Selected for the saved 2026 forecasts</h2>"
    body += (
        "<p>Control retains the original policy. Features adds explicit QB inputs to the trees. "
        "Component features multiplies the improved attempts forecast by a blended estimate "
        "of yards per attempt.</p>"
    )
    body += table(
        ["Model", "Target", "Selected policy", "Selection years"],
        [
            [mid, target, choice["winner"], ", ".join(map(str, choice["validation_years"]))]
            for mid, targets in choices.items()
            for target, choice in targets.items()
        ],
    )
    body += "<h2>QBs with at least 300 prior-season attempts</h2>"
    body += table(
        ["Model", "Target", "Old MAE", "Revised MAE", "Old RMSE", "Revised RMSE"],
        [
            [
                r["model_id"],
                r["target"],
                *[
                    f"{r[version][metric]:.2f}"
                    for metric in ("mae", "rmse")
                    for version in ("old", "revised_policy")
                ],
            ]
            for r in rows
            if r["cohort"] == "high_prior_usage"
        ],
    )
    body += "<h2>Candidate coverage</h2>"
    body += table(
        ["Model", "Before", "After", "Added"],
        [[r["model_id"], r["before"], r["after"], len(r["added"])] for r in result["coverage"]],
    )
    body += (
        "<h2>Interpretation</h2><ul>"
        + "".join(f"<li>{html.escape(s)}</li>" for s in result["interpretation"])
        + "</ul>"
    )
    body += (
        f"<p>Manifest: <code>{campaign['manifest_sha256']}</code></p>"
        '<p><a href="readout.json">Full evidence summary</a></p>'
    )
    document = (
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" '
        'content="width=device-width,initial-scale=1"><title>Opening Drive v2 revision</title>'
        "<style>body{max-width:1100px;margin:40px auto;padding:0 24px;font:16px/1.6 system-ui;"
        "color:#193b32;background:#f5f4ed}table{width:100%;border-collapse:collapse;background:white}"
        "th,td{padding:10px;text-align:left;border-bottom:1px solid #dce3da}"
        "code{overflow-wrap:anywhere}h1,h2{line-height:1.2}li{margin-bottom:8px}</style>"
        + body
        + "</html>"
    )
    (destination / "readout.html").write_text(document)
    return destination / "readout.html"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    args = parser.parse_args()
    print(build(args.run, args.registry))


if __name__ == "__main__":
    main()
