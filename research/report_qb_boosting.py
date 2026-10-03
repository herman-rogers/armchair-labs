"""Render the immutable QB sweep into a reviewable report and scientific figures.

Run with: uv run --no-sync --with matplotlib python research/report_qb_boosting.py RUN_DIR
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from engine.data.releases import digest

ROOT = Path(__file__).resolve().parents[1]
LABELS = {
    "season_points": "Fantasy points",
    "passing_yards": "Passing yards",
    "rushing_yards": "Rushing yards",
    "attempts": "Pass attempts",
    "carries": "Carries",
    "scoring_appearances": "Scoring appearances",
    "season_appearance": "Appearance (Brier)",
    "passing_efficiency": "Passing yards/attempt",
    "rushing_efficiency": "Rushing yards/carry",
}


def render(run, output):
    manifest = json.loads((run / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if digest(run / name) != expected:
            raise ValueError(f"Changed research artifact: {name}")
    report = json.loads((run / "report.json").read_text())
    selections = json.loads((run / "selections.json").read_text())
    configs = json.loads((run / "configurations.json").read_text())
    curves = pl.read_csv(run / "capacity_curves.csv").to_dicts()
    targets = report["targets"]
    comparisons = report["comparisons"]

    def result(target, model="selected_canonical", comparator="boost", window="modern"):
        return next(
            r
            for r in comparisons
            if r["target"] == target
            and r["model"] == model
            and r["comparator"] == comparator
            and r["window"] == window
        )

    def fmt(value):
        return f"{value:.4f}" if abs(value) < 1 else f"{value:,.2f}"

    def metric(r):
        return "mse" if r["target"] == "season_appearance" else "mae"

    output.parent.mkdir(parents=True, exist_ok=True)
    assets = output.parent / "assets"
    assets.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    shapes = [
        (1, 2, "Depth 1 / 2 leaves"),
        (5, 15, "Depth 5 / 15 leaves"),
        (None, 63, "Unlimited depth / 63 leaves"),
    ]
    for ax, target in zip(
        axes, ["passing_yards", "season_points", "passing_efficiency"], strict=True
    ):
        for depth, leaves, label in shapes:
            subset = sorted(
                [
                    r
                    for r in curves
                    if r["target"] == target
                    and r["window"] == "modern"
                    and r["max_depth"] == depth
                    and r["max_leaf_nodes"] == leaves
                    and r["learning_rate"] == 0.05
                    and r["min_samples_leaf"] == 30
                    and r["l2_regularization"] == 10
                    and r["max_features"] == 1
                ],
                key=lambda r: r["max_iter"],
            )
            (line,) = ax.plot(
                [r["max_iter"] for r in subset], [r["mae"] for r in subset], marker="o", label=label
            )
            ax.plot(
                [r["max_iter"] for r in subset],
                [r["train_mae"] for r in subset],
                linestyle="--",
                color=line.get_color(),
                alpha=0.7,
            )
        ax.axhline(
            result(target)["reference_mae"],
            color="#444444",
            linestyle=":",
            label="Original 120-tree boost",
        )
        ax.set(title=LABELS[target], xlabel="Trees", ylabel="MAE (equal weight per fold)")
        ax.set_xscale("log", base=2)
        ax.set_xticks([30, 60, 120, 240, 480], labels=["30", "60", "120", "240", "480"])
        ax.grid(alpha=0.15)
    axes[0].legend(fontsize=8)
    fig.suptitle(
        "QB capacity diagnostic · 2019–2025 future seasons\n"
        "Solid = future-season error; dashed = expanding-training error. "
        "Retrospective fixed configurations, not selected-policy scores.",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    plot = assets / "qb_boosting_capacity_2026-09-24.png"
    fig.savefig(plot, dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8), sharey=True)
    for ax, model, title in zip(
        axes,
        ["selected_canonical", "selected_mse"],
        ["Selection by canonical loss", "Selection by squared error"],
        strict=True,
    ):
        gains = []
        for target in targets:
            r = result(target, model)
            key = r["metric"]
            gains.append(100 * r["gain"] / r["reference_" + key])
        colors = ["#007d73" if x >= 0 else "#ba4944" for x in gains]
        ax.barh(np.arange(len(targets)), gains, color=colors)
        ax.axvline(0, color="#333333", lw=0.8)
        ax.set(title=title, xlabel="Error reduction vs original boost (%)")
        ax.grid(axis="x", alpha=0.15)
        ax.set_yticks(np.arange(len(targets)), labels=[LABELS[t] for t in targets])
    axes[0].invert_yaxis()
    fig.suptitle(
        "Chronologically selected policies · 2019–2025\n"
        "Settings selected using earlier seasons only; "
        "historical research, not prospective validation."
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(assets / "qb_boosting_policy_2026-09-24.png", dpi=170)
    plt.close(fig)

    passing = result("passing_yards")
    efficiency = result("passing_efficiency")
    efficiency_ridge = result("passing_efficiency", comparator="ridge")
    passing_mse = result("passing_yards", "selected_mse")
    adjusted = [r for r in comparisons if "holm_p_value" in r]
    supported = [r for r in adjusted if r["holm_p_value"] <= 0.05 and r["gain"] > 0]
    lines = [
        "# QB boosted-tree capacity research · September 24, 2026",
        "",
        f"Completed **{report['configurations']} configurations "
        "× nine outcomes × 19 annual folds** "
        f"({report['configurations'] * report['folds']:,} candidate/fold evaluations). "
        f"All {report['reproduced_model_folds']} saved baseline/ridge/boost model-fold outputs "
        "reproduced, with maximum prediction difference "
        f"{report['maximum_reproduction_difference']:g}. "
        "Canonical error aggregation and row counts also reproduced "
        "for every target, model and window.",
        "",
        f"For modern passing yards, chronological canonical-loss selection has MAE "
        f"**{passing['mae']:.2f}**, versus **{passing['reference_mae']:.2f}** for the existing "
        f"booster. Gain: {passing['gain']:.2f} yards, 95% paired season-bootstrap interval "
        f"[{passing['ci_low']:.2f}, {passing['ci_high']:.2f}]. "
        "Across all outcomes and both selection rules, "
        f"**{len(supported)}/18** modern improvements "
        "over the fixed booster survive the predeclared Holm adjustment.",
        "",
        "**Disposition: research only.** This study isolates capacity changes on the nine "
        "canonical preseason full-season QB outcomes. It does not test weekly forecasts or "
        "change production models. All historical years have been reused in prior research; "
        "even the recent frozen-settings check is not an independent holdout.",
        "",
        "## Interpretation",
        "",
        "The tested tuning policies do not establish a broad improvement over the existing "
        "booster. Keep the canonical serving policy unchanged. Passing-yard and fantasy-point "
        "MAE worsen slightly, with uncertainty spanning improvement and deterioration.",
        "",
        f"Passing efficiency merits focused prospective comparison: MAE improves "
        f"{100 * efficiency['gain'] / efficiency['reference_mae']:.2f}% versus fixed boost, "
        f"with improvement in {efficiency['positive_years']}/{efficiency['years']} modern "
        "seasons and a similar gain across full history. The most-used selected setup "
        "uses just 30 trees and a larger minimum leaf size of 60. Its incremental gain "
        f"over ridge is only {efficiency_ridge['gain']:.3f} yards/attempt "
        f"(95% descriptive interval [{efficiency_ridge['ci_low']:.3f}, "
        f"{efficiency_ridge['ci_high']:.3f}]), so superiority to ridge remains uncertain.",
        "",
        "Optimizing squared error produces a small passing-yard tradeoff: "
        f"RMSE falls from {passing_mse['reference_rmse']:.2f} to {passing_mse['rmse']:.2f}, "
        f"while MAE rises from {passing_mse['reference_mae']:.2f} to "
        f"{passing_mse['mae']:.2f}. The MSE-gain interval includes zero. Greater tree "
        "capacity substantially reduces training error, but does not consistently improve "
        "future-season error; the stress test below makes that failure visible.",
        "",
        "## Matched canonical errors",
        "",
        "Every season receives equal weight. Errors below are MAE in the named units, except "
        "appearance probability, which uses Brier score. Zero outcomes and backups remain; "
        "rate targets use only defined denominators. “Selected” is the full chronological "
        "canonical-loss policy, not the best configuration chosen after seeing results.",
        "",
    ]
    for window, label in [("modern", "2019–2025"), ("all_history", "2007–2025")]:
        lines += [
            f"### {label}",
            "",
            "| Outcome | Scored rows | Canonical reference | Ridge | Original boost | Selected |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for target in targets:
            r = result(target, window=window)
            b = result(target, comparator="baseline", window=window)
            ridge = result(target, comparator="ridge", window=window)
            key = metric(r)
            lines.append(
                f"| {LABELS[target]} | {r['n']:,} | {fmt(b['reference_' + key])} | "
                f"{fmt(ridge['reference_' + key])} | {fmt(r['reference_' + key])} | "
                f"{fmt(r[key])} |"
            )
        lines += [""]
    lines += [
        "## Squared error and uncertainty",
        "",
        "Expected-value forecasts also need squared-error evaluation. This independently "
        "declared selection policy optimizes earlier MSE with an MAE guardrail. RMSE is the "
        "square root of equal-season MSE. Holm values cover all nine outcomes × both policies; "
        "unadjusted bootstrap intervals below are descriptive.",
        "",
        "**Small-sample limit:** seven modern seasons permit a smallest two-sided exact "
        "sign-flip p-value of 2/128 = .015625. The first Holm threshold for 18 tests is "
        ".05/18 = .00278. Therefore this conservative modern family cannot establish "
        "adjusted significance even if every season improves. Interpret effect sizes, "
        "uncertainty and additional future evidence; a failed significance threshold "
        "alone cannot establish that tuning has no value.",
        "",
        "| Modern outcome | Original RMSE | MSE-policy RMSE | MSE reduction | Holm p |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for target in targets:
        r = result(target, "selected_mse")
        lines.append(
            f"| {LABELS[target]} | {fmt(r['reference_rmse'])} | {fmt(r['rmse'])} | "
            f"{100 * r['gain'] / r['reference_mse']:.2f}% | {r['holm_p_value']:.3f} |"
        )
    lines += [
        "",
        "![Chronological policy comparison](assets/qb_boosting_policy_2026-09-24.png)",
        "",
        "| Canonical-loss policy, modern | Gain vs boost | 95% season interval | "
        "95% two-year-block interval | Winning years | Holm p |",
        "| --- | ---: | --- | --- | ---: | ---: |",
    ]
    for target in targets:
        r = result(target)
        lines.append(
            f"| {LABELS[target]} | {fmt(r['gain'])} | "
            f"[{fmt(r['ci_low'])}, {fmt(r['ci_high'])}] | "
            f"[{fmt(r['block_ci_low'])}, {fmt(r['block_ci_high'])}] | "
            f"{r['positive_years']}/{r['years']} | {r['holm_p_value']:.3f} |"
        )
    lines += [
        "",
        "## Capacity, tree count and overfitting",
        "",
        "39 paths vary depth (1/2/3/5/unlimited), maximum leaves (2–63), learning rate "
        "(.02/.05/.10), minimum leaf samples (5–60), L2 (0/10/100) and feature fraction "
        "(.5/.75/1). Each is evaluated at 30/60/120/240/480 trees. The bounded design "
        "covers structural interactions and regularization controls, not every Cartesian "
        "combination. Maximum leaves is the width/capacity proxy.",
        "",
        "![Training and future-season errors](assets/qb_boosting_capacity_2026-09-24.png)",
        "",
        "Dashed errors use each expanding training set, while solid errors use its next "
        "season. The populations and sample sizes differ: the gap is an overfitting "
        "diagnostic, not an unbiased estimate of optimism. The plotted fixed-configuration "
        "curves inspect reused test years and must not be substituted for policy results.",
        "",
        "Unlimited depth, 63 leaves, minimum leaf 5, L2=0, learning rate .10 provides the "
        "deliberate stress test below (modern passing yards).",
        "",
        "| Trees | Training MAE | Next-season MAE | Next-season RMSE |",
        "| --- | ---: | ---: | ---: |",
    ]
    stress = sorted(
        [
            r
            for r in curves
            if r["target"] == "passing_yards"
            and r["window"] == "modern"
            and r["max_leaf_nodes"] == 63
            and r["min_samples_leaf"] == 5
            and r["learning_rate"] == 0.1
        ],
        key=lambda r: r["max_iter"],
    )
    for r in stress:
        lines.append(
            f"| {r['max_iter']} | {r['train_mae']:.2f} | {r['mae']:.2f} | {np.sqrt(r['mse']):.2f} |"
        )
    lines += [
        "",
        "## Selection stability and hindsight",
        "",
        "Training starts in 2004. Annual tests start in 2007. Until three earlier "
        "validation seasons exist, each policy uses the fixed booster. Subsequently "
        "the selector uses earlier equal-season losses and a paired one-standard-error "
        "preference for simpler models, subject to the declared guardrails. Test-year "
        "outcomes are unavailable to the selector; there is no random row split.",
        "",
        "| Outcome | Most-used modern configuration | Seasons using it | "
        "Distinct modern selections | Hindsight-best modern error* | Actual policy error |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for target in targets:
        counts = Counter(
            r["configuration"]
            for r in selections
            if r["target"] == target and r["policy"] == "canonical" and r["season"] >= 2019
        )
        common, n = counts.most_common(1)[0]
        hindsight = next(
            r for r in report["hindsight"] if r["target"] == target and r["window"] == "modern"
        )
        r = result(target)
        lines.append(
            f"| {LABELS[target]} | `{common}` | {n}/7 | {len(counts)} | "
            f"{fmt(hindsight['loss'])} | {fmt(r[metric(r)])} |"
        )
    lines += [
        "",
        "*Hindsight error uses the same 2019–2025 labels to choose and score "
        "the winner. It is explicitly optimistic search evidence, not achieved forecast "
        "performance. A changing chronological policy can occasionally beat every fixed "
        "configuration; that does not validate hindsight selection.",
        "",
        "Most-used canonical-policy settings:",
        "",
        "| Outcome | Depth | Leaves | Trees | Rate | Min leaf | L2 | Feature fraction |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for target in targets:
        counts = Counter(
            r["configuration"]
            for r in selections
            if r["target"] == target and r["policy"] == "canonical" and r["season"] >= 2019
        )
        c = next(c for c in configs if c["id"] == counts.most_common(1)[0][0])
        lines.append(
            f"| {LABELS[target]} | {c['max_depth'] or 'unlimited'} | "
            f"{c['max_leaf_nodes']} | {c['max_iter']} | {c['learning_rate']} | "
            f"{c['min_samples_leaf']} | {c['l2_regularization']} | {c['max_features']} |"
        )
    lines += [
        "",
        "## Recent frozen-settings check",
        "",
        "Select settings using only 2007–2022, then keep them fixed for 2023–2025. "
        "Training expands annually, so completed 2023 labels can train the 2024 model "
        "but cannot change the chosen configuration. Three seasons provide limited evidence.",
        "",
        "| 2023–2025 outcome | Original boost | Annually selected | Settings frozen before 2023 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for target in targets:
        r = result(target, window="recent_locked")
        locked = result(target, "locked_canonical", window="recent_locked")
        key = metric(r)
        lines.append(
            f"| {LABELS[target]} | {fmt(r['reference_' + key])} | {fmt(r[key])} | "
            f"{fmt(locked[key])} |"
        )
    lines += [
        "",
        "## Cutoff-defined subgroups",
        "",
        "Modern passing-yard canonical-policy MAE, using prior passing workload. These "
        "groups do not establish starting status, health, or the cause of an error.",
        "",
        "| Prior attempts | Rows | Canonical reference | Original boost | Selected |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    groups = [
        r
        for r in report["subgroups"]
        if r["target"] == "passing_yards"
        and r["window"] == "modern"
        and r["field"] == "prior_workload"
        and r["policy"] == "canonical"
        and r["comparator"] == "boost"
    ]
    for r in groups:
        b = next(
            s
            for s in report["subgroups"]
            if s["target"] == r["target"]
            and s["window"] == r["window"]
            and s["field"] == r["field"]
            and s["group"] == r["group"]
            and s["policy"] == r["policy"]
            and s["comparator"] == "baseline"
        )
        lines.append(
            f"| {r['group']} | {r['n']} | {b['reference_mae']:.2f} | "
            f"{r['reference_mae']:.2f} | {r['mae']:.2f} |"
        )
    harm = [
        r
        for r in report["subgroups"]
        if r["window"] == "modern"
        and r["comparator"] == "boost"
        and r["policy"] == "canonical"
        and r["n"] >= 100
        and r["years"] >= 3
        and r["gain"] < -0.1 * r["reference_" + r["metric"]]
    ]
    lines += [
        "",
        f"Across all outcomes, **{len(harm)} eligible modern subgroup comparisons** "
        "(at least 100 rows and three seasons) worsen canonical loss by more than 10% "
        "against fixed boost. Overlapping groups are descriptive, not independent tests.",
    ]
    for r in harm:
        lines.append(
            f"- {LABELS[r['target']]} · {r['field']}={r['group']}: "
            f"{-100 * r['gain'] / r['reference_' + r['metric']]:.1f}% worse, n={r['n']}."
        )
    lines += [
        "",
        "## Limitations and reproducibility",
        "",
        *["- " + s for s in report["limitations"]],
        "",
        "Feature sets, missingness, fallback rules, nonnegative clipping and dated "
        "availability caps exactly match the canonical source. The existing missing "
        "retirement/role-vintage limitations are retained to avoid confounding capacity "
        "with input repairs. This study cannot determine injury-caused or role-caused "
        "shares of error. Current 2026 outcomes were not used.",
        "",
        "Bootstrap intervals resample saved annual errors without refitting or repeating "
        "model selection. They omit the uncertainty of a new training/search realization. "
        "The paired one-standard-error simplicity rule is a heuristic whose own choice "
        "has not been independently validated.",
        "",
        f"- [Frozen protocol](../data/research/{run.name}/protocol.md)",
        f"- [Full results and subgroup errors](../data/research/{run.name}/report.json)",
        f"- [All settings and capacity curves](../data/research/{run.name}/capacity_curves.csv)",
        f"- [Annual selections](../data/research/{run.name}/selections.json)",
        f"- [Policy predictions](../data/research/{run.name}/policy_predictions.parquet)",
        f"- [Hashed manifest](../data/research/{run.name}/manifest.json)",
        "- Candidate row predictions and training/test errors are retained in each "
        "`folds/TARGET_YEAR.npz` and `.json`; array rows follow `configurations.json`.",
        "",
        "```sh",
        ".venv/bin/python research/qb_boosting_sweep.py --version NEW_VERSION --workers 4",
        "uv run --no-sync --with matplotlib python research/report_qb_boosting.py "
        "data/research/NEW_VERSION",
        "```",
        "",
        "Parameter semantics: [official scikit-learn documentation]"
        "(https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html).",
    ]
    output.write_text("\n".join(lines) + "\n")
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "docs/research/qb_boosting_capacity_2026-09-24.md"
    )
    args = parser.parse_args()
    render(args.run, args.output)
