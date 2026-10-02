"""Paired season-level comparisons and descriptive discovery stability."""

# ruff: noqa: E501
from __future__ import annotations

import html
import json
from collections import Counter

import numpy as np
import polars as pl


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def build_report(destination, predictions, folds, permutations, diagnostics, config, inventory):
    df = pl.DataFrame(predictions).with_columns(
        (pl.col("prediction") - pl.col("actual")).abs().alias("ae"),
        (pl.col("prediction") - pl.col("actual")).pow(2).alias("se"),
    )
    annual = (
        df.group_by("position", "model", "season")
        .agg(
            pl.col("ae").mean().alias("mae"),
            pl.col("se").mean().alias("mse"),
            pl.len().alias("n"),
        )
        .sort("position", "model", "season")
    )
    annual.write_csv(destination / "annual_scores.csv")
    subgroups = df.group_by("position", "model", "population", "season").agg(
        pl.col("ae").mean().alias("mae"),
        pl.col("se").mean().alias("mse"),
        pl.len().alias("n"),
    )
    subgroups.write_csv(destination / "population_scores.csv")
    era = df.with_columns(
        pl.when(pl.col("season") >= 2017)
        .then(pl.lit("2017-2025"))
        .otherwise(pl.lit("2008-2016"))
        .alias("era")
    )
    era.group_by("position", "model", "era", "season").agg(
        pl.col("ae").mean().alias("mae"),
        pl.col("se").mean().alias("mse"),
    ).group_by("position", "model", "era").agg(
        pl.col("mae").mean(),
        pl.col("mse").mean(),
        pl.len().alias("seasons"),
    ).write_csv(destination / "era_scores.csv")
    summaries = []
    for position in config["positions"]:
        controls = {
            name: annual.filter((pl.col("position") == position) & (pl.col("model") == name))
            for name in ("basic_tree", "legacy_screen_tree", "inventory_tree15", "raw_tree15")
        }
        for name in sorted(df["model"].unique()):
            values = annual.filter((pl.col("position") == position) & (pl.col("model") == name))
            for control, base in controls.items():
                paired = values.join(
                    base, on=["position", "season"], suffix="_control", validate="1:1"
                )
                gain = (paired["mse_control"] - paired["mse"]).to_numpy()
                mae_gain = (paired["mae_control"] - paired["mae"]).to_numpy()
                rng = np.random.default_rng(config["seed"])
                bootstrap = gain[
                    rng.integers(0, len(gain), (config["bootstrap_draws"], len(gain)))
                ].mean(axis=1)
                summaries.append(
                    {
                        "position": position,
                        "model": name,
                        "control": control,
                        "seasons": len(gain),
                        "rows": int(values["n"].sum()),
                        "mse": float(values["mse"].mean()),
                        "mae": float(values["mae"].mean()),
                        "mse_gain": float(gain.mean()),
                        "mae_gain": float(mae_gain.mean()),
                        "mse_gain_pct": float(100 * gain.mean() / paired["mse_control"].mean()),
                        "ci_low": float(np.quantile(bootstrap, 0.025)),
                        "ci_high": float(np.quantile(bootstrap, 0.975)),
                        "seasons_improved": int((gain > 0).sum()),
                    }
                )
    pl.DataFrame(summaries).write_csv(destination / "comparisons.csv")
    write_json(destination / "comparisons.json", summaries)
    selected = Counter((r["position"], r["search"], r["chosen_model"]) for r in folds)
    choices = [
        {"position": p, "search": s, "model": m, "folds": n}
        for (p, s, m), n in sorted(selected.items())
    ]
    write_json(destination / "selection_counts.json", choices)
    stability, pairs = Counter(), Counter()
    eligible = [r for r in diagnostics if r["season"] >= config["evaluation_start"]]
    for row in eligible:
        if row["model"].endswith("selected_svr"):
            for name in row.get("selected_features", []):
                stability[row["position"], row["model"], name] += 1
            for pair in row.get("split_cooccurrences", []):
                pairs[row["position"], row["model"], pair["first"], pair["second"]] += 1
    stability_rows = [
        {"position": p, "model": m, "feature": f, "folds": n}
        for (p, m, f), n in stability.most_common()
    ]
    pair_rows = [
        {"position": p, "model": m, "first": a, "second": b, "folds": n}
        for (p, m, a, b), n in pairs.most_common()
    ]
    write_json(destination / "feature_selection_stability.json", stability_rows)
    write_json(destination / "split_cooccurrences.json", pair_rows)
    permutation = (
        pl.DataFrame(permutations)
        .group_by("position", "model", "group")
        .agg(
            pl.col("mse_increase").mean(),
            (pl.col("mse_increase") > 0).sum().alias("positive_seasons"),
            pl.len().alias("seasons"),
        )
        .sort(["position", "model", "mse_increase"], descending=[False, False, True])
    )
    permutation.write_csv(destination / "permutation_stability.csv")
    lines = [
        "# Automated representation experiment",
        "",
        "Research only: preseason full-season league points, QB/RB/WR/TE. "
        "All methods forecast identical candidates. No current rankings or production models changed.",
        "",
        f"Inventory: {inventory['admitted_count']} admissible candidates, "
        f"{inventory['binary_count_descriptive_only']} binary; {inventory['market_count']} market inputs. "
        f"Own-data inventory has {len(inventory['columns']['inventory'])} columns; "
        f"raw history has {len(inventory['columns']['raw'])}. No individual-statistic result filters inputs.",
        "",
        "Raw history means 13 weekly observations across the last three calendar seasons, "
        "annual observations for earlier lags 4–25, and measured player/status/college fields. "
        "This is a generic tabular representation, not an end-to-end sequence neural network. "
        "The broad inventory also includes existing summaries and handcoded formulas.",
        "",
        f"Evaluation: {config['evaluation_start']}–{config['evaluation_end']}. "
        "Every model trains on all earlier completed seasons. Search chooses a recipe from "
        f"the last {config['inner_seasons']} earlier chronological validation seasons by equal-season MSE. "
        "Cached validation predictions were each fitted only on years before their own validation year.",
        "",
        "## Paired results against the compact Basic tree",
        "",
        "MSE is primary; MAE is secondary. Positive gain means lower error. "
        "Intervals resample whole seasons and are descriptive, unadjusted for multiple comparisons. "
        "These historical seasons have been used in prior research; this is not fresh prospective confirmation.",
        "",
        "| Position | Method | MSE | MAE | MSE gain % | MSE gain 95% interval |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    display = [
        "basic_tree",
        "legacy_screen_tree",
        "raw_tree15",
        "raw_search",
        "inventory_tree15",
        "inventory_search",
        "joint_search",
        "inventory_market_tree",
    ]
    for r in summaries:
        if r["control"] == "basic_tree" and r["model"] in display:
            lines.append(
                f"| {r['position']} | {r['model']} | {r['mse']:.1f} | {r['mae']:.2f} | "
                f"{r['mse_gain_pct']:+.1f}% | [{r['ci_low']:.1f}, {r['ci_high']:.1f}] |"
            )
    lines += [
        "",
        "## Repeatedly useful groups in held-out permutations",
        "",
        "Shown for the unfiltered raw tree. Positive MSE increase means shuffling the group "
        "hurt predictions. Correlated substitutes can obscure importance; shuffling can create "
        "unusual combinations. These diagnostic scores never select a model.",
        "",
        "| Position | Group | Mean MSE increase | Positive seasons |",
        "| --- | --- | ---: | ---: |",
    ]
    for p in config["positions"]:
        rows = permutation.filter(
            (pl.col("position") == p) & (pl.col("model") == "raw_tree15")
        ).head(5)
        for r in rows.to_dicts():
            lines.append(
                f"| {p} | {r['group']} | {r['mse_increase']:.1f} | {r['positive_seasons']}/{r['seasons']} |"
            )
    lines += [
        "",
        "## Interpretation and artifacts",
        "",
        "Search has to outperform its unfiltered comparators; a selected subset is not "
        "automatically better. See comparisons.csv for matched comparisons against the "
        "old selector and both unfiltered trees, plus population_scores.csv and era_scores.csv.",
        "",
        "Feature-selection recurrence and parent/child split co-occurrence files are hypothesis "
        "generators. Co-occurrence does not demonstrate an interaction benefit. PCA loadings, "
        "model coefficients and forecast errors are not causal effects. A future paired ablation "
        "or preregistered prospective test is needed before promoting an interaction.",
        "",
        "The old-selector control reproduces its top-80 rank-correlation gate and binary exclusion "
        "using the same new tree/preprocessing as the unfiltered control; it is not a replay of "
        "the entire historical discovery stack. All missingness indicators remain paired with values.",
        "",
        "Full validation choices: folds.json. Source/config/dependency hashes: manifest.json. "
        "Forecasts: predictions.parquet. Every trial has its own create-only directory. "
        "Only status=complete with unchanged protected inputs is usable.",
        "",
        "Data limits: historical source vintages are reconstructed; missing records do not prove "
        "inactivity; weekly history covers three prior seasons with older annual totals; "
        "raw history omits some tracking/context sources present in the broader inventory. "
        "Raw-versus-inventory differences therefore reflect information as well as representation. "
        "No weekly horizon, neural architecture search, or automatically generated symbolic interaction "
        "search was evaluated. The market arm is a separate fixed benchmark, not part of own-data search.",
        "",
        "Leakage reference: [scikit-learn common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage).",
    ]
    report = "\n".join(lines) + "\n"
    (destination / "report.md").write_text(report)
    table_rows = "".join(
        "<tr>"
        + "".join(
            f"<td>{html.escape(str(r[k]))}</td>"
            for k in (
                "position",
                "model",
                "control",
                "mae",
                "mse",
                "mse_gain_pct",
                "ci_low",
                "ci_high",
            )
        )
        + "</tr>"
        for r in summaries
    )
    page = """<!doctype html><meta charset="utf-8"><title>Representation experiment</title>
<style>body{font:15px system-ui;margin:2rem;max-width:1400px}table{border-collapse:collapse;width:100%}
td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}input{font:inherit;padding:8px;width:70%}
pre{white-space:pre-wrap;line-height:1.5}thead{position:sticky;top:0;background:white}</style>
<h1>Representation experiment</h1><p>Research only. Positive gain means lower error; intervals are descriptive.</p>
<p><a href="report.md">Readout</a> · <a href="comparisons.csv">Download comparisons</a> · <a href="manifest.json">Provenance</a></p>
<label>Filter comparisons <input id="filter" placeholder="QB raw_search"></label>
<table><thead><tr><th>Position</th><th>Model</th><th>Control</th><th>MAE</th><th>MSE</th><th>MSE gain %</th><th>CI low</th><th>CI high</th></tr></thead><tbody>"""
    page += table_rows + "</tbody></table><h2>Readout</h2><pre>" + html.escape(report) + "</pre>"
    page += r"""<script>document.querySelector('#filter').addEventListener('input', e=>{const terms=e.target.value.toLowerCase().split(/\s+/);document.querySelectorAll('tbody tr').forEach(r=>r.hidden=!terms.every(t=>r.textContent.toLowerCase().includes(t)))})</script>"""
    (destination / "index.html").write_text(page)
