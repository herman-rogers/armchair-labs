"""Describe marginal value, capacity boundaries, calibration and discovered inputs."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl

from ..data import digest
from ..evaluate import paired_interval, score
from ..run import write_json
from .ensemble import capture_score
from .run import LAB


def paired_methods(frame, model, control, seed=20260925):
    keys = ["player_id", "position", "year", "origin", "horizon"]
    a = frame.filter(pl.col("model") == model)
    b = frame.filter(pl.col("model") == control).select(
        *keys, pl.col("prediction").alias("control")
    )
    matched = a.join(b, on=keys, validate="1:1")
    if not a.height or matched.height != a.height:
        raise ValueError("Unmatched ensemble comparison")
    rows = []
    for (year,), part in matched.partition_by("year", as_dict=True).items():
        part = part.sort("player_id")
        actual, pred, ref = [part[c].to_numpy() for c in ["actual", "prediction", "control"]]
        origin = part["origin"].to_numpy()
        k = {"QB": 10, "RB": 20, "WR": 30, "TE": 10}[part["position"][0]]
        rows.append(
            dict(
                year=year,
                mse=float(np.mean((pred - actual) ** 2)),
                control_mse=float(np.mean((ref - actual) ** 2)),
                mae_gain=float(np.mean(abs(ref - actual) - abs(pred - actual))),
                ndcg_gain=score(actual, pred, origin)["ndcg24"]
                - score(actual, ref, origin)["ndcg24"],
                capture_gain=capture_score(actual, pred, origin, k)
                - capture_score(actual, ref, origin, k),
            )
        )
    delta = np.array([r["control_mse"] - r["mse"] for r in rows])
    return dict(
        model=model,
        control=control,
        n=matched.height,
        years=len(rows),
        mse_reduction_pct=float(100 * delta.mean() / np.mean([r["control_mse"] for r in rows])),
        mae_gain=float(np.mean([r["mae_gain"] for r in rows])),
        ndcg_gain=float(np.mean([r["ndcg_gain"] for r in rows])),
        capture_gain=float(np.mean([r["capture_gain"] for r in rows])),
        mse_gain_ci=paired_interval(delta, 5000, seed),
        capture_gain_ci=paired_interval([r["capture_gain"] for r in rows], 5000, seed),
        years_won=int((delta > 0).sum()),
        capture_years_won=sum(r["capture_gain"] > 0 for r in rows),
    )


def paired_scores(metrics, model, control, seed=20260925):
    """Compare annual losses on identical evaluated populations, not pooled rows."""
    a = metrics.filter(pl.col("model") == model).sort("year")
    b = metrics.filter(pl.col("model") == control).sort("year")
    if not a.height or not a.select("year", "n").equals(b.select("year", "n")):
        raise ValueError("Annual comparisons require identical seasons and row counts")
    delta = (b["mse"] - a["mse"]).to_numpy()
    return dict(
        model=model,
        control=control,
        years=a.height,
        n=int(a["n"].sum()),
        mse_reduction_pct=float(100 * delta.mean() / b["mse"].mean()),
        mae_gain=float((b["mae"] - a["mae"]).mean()),
        ndcg_gain=float((a["ndcg24"] - b["ndcg24"]).mean()),
        mse_gain_ci=paired_interval(delta, 5000, seed),
        years_won=int((delta > 0).sum()),
    )


def input_coverage(data):
    """Describe measurement availability; never use this to select predictors."""
    panel = pl.read_parquet(data / "panel.parquet")
    metadata = json.loads((data / "features.json").read_text())
    x = np.load(data / "x.npy", mmap_mode="r")
    groups = np.array(metadata["groups"])
    records = []
    # One preseason row per candidate season avoids counting the four target
    # horizons as four independent sources of preseason information.
    for position in ["QB", "RB", "WR", "TE"]:
        for population in ["all", "rookie", "returner", "market_only"]:
            mask = (panel["position"].to_numpy() == position) & (
                panel["horizon"].to_numpy() == "season"
            ) & (panel["year"].to_numpy() >= 2019) & (panel["year"].to_numpy() <= 2025)
            if population != "all":
                mask &= panel["population"].to_numpy() == population
            if not mask.any():
                continue
            values = x[mask]
            observed = np.isfinite(values)
            for group in ["ALL", *sorted(set(groups))]:
                cols = np.ones(len(groups), dtype=bool) if group == "ALL" else groups == group
                counts = observed[:, cols].sum(axis=0)
                records.append(
                    dict(
                        position=position,
                        population=population,
                        years="2019–2025",
                        group=group,
                        rows=int(mask.sum()),
                        columns=int(cols.sum()),
                        entirely_missing_columns=int((counts == 0).sum()),
                        observed_cell_fraction=float(counts.sum() / (mask.sum() * cols.sum())),
                        fully_observed_columns=int((counts == mask.sum()).sum()),
                    )
                )
    return records


def collect(report, source):
    result, latest, budgets, extremes, cohorts = [], [], [], [], []
    feature_views, heldout_capacity, serving_library = [], [], []
    root_summary = json.loads((report / "summary.json").read_text())
    protocol = json.loads((source / "protocol.json").read_text())
    for folder in report.iterdir():
        if not (folder / "forecasts.parquet").exists():
            continue
        position, horizon = folder.name.split("_", 1)
        frame = pl.read_parquet(folder / "forecasts.parquet")
        metrics = pl.read_parquet(folder / "metrics.parquet")
        for path in protocol["paths"]:
            if path["family"] not in {"lgb", "xgb", "cat", "hist", "extra", "forest"}:
                continue
            for larger, smaller in [(120, 40), (360, 120)]:
                heldout_capacity.append(
                    dict(
                        position=position,
                        horizon=horizon,
                        family=path["family"],
                        path=path["name"],
                        **paired_scores(
                            metrics,
                            f"base:{path['name']}_t{larger}",
                            f"base:{path['name']}_t{smaller}",
                        ),
                    )
                )
        # These first six paths share structural settings. Other path comparisons
        # can confound representation, regularization and recency choices.
        views = ["summary", "raw", "inventory", "all", "market", "basic"]
        for larger, smaller in [(3, 0), (3, 1), (4, 3), (3, 5)]:
            for count in [40, 120, 360]:
                feature_views.append(
                    dict(
                        position=position,
                        horizon=horizon,
                        view=views[larger],
                        control_view=views[smaller],
                        trees=count,
                        **paired_scores(
                            metrics,
                            f"base:lgb_{larger:02d}_t{count}",
                            f"base:lgb_{smaller:02d}_t{count}",
                        ),
                    )
                )
        second = report / "published_library" / folder.name / "forecasts.parquet"
        if second.exists():
            main_matched = pl.read_parquet(folder / "matched.parquet").select(frame.columns)
            extra = pl.read_parquet(second)
            for policy in ["policy_mse", "policy_mae", "policy_rank", "policy_capture"]:
                pair = pl.concat(
                    [
                        extra.filter(pl.col("model") == policy),
                        main_matched.filter(pl.col("model") == policy).with_columns(
                            pl.lit("new_library").alias("model")
                        ),
                    ]
                )
                serving_library.append(
                    dict(
                        position=position,
                        horizon=horizon,
                        **paired_methods(pair, policy, "new_library"),
                    )
                )
        for policy in ["policy_mse", "policy_mae", "policy_rank", "policy_capture"]:
            result.append(
                dict(position=position, horizon=horizon, **paired_methods(frame, policy, "top_3"))
            )
        choices = json.loads((folder / "choices.json").read_text())
        for row in choices:
            if row["year"] == 2025:
                latest.append(dict(position=position, horizon=horizon, **row["selected"]))
        for row in json.loads((folder / "budget.json").read_text()):
            if row["year"] >= 2019:
                budgets.append(dict(position=position, horizon=horizon, **row))
        selected = frame.filter(pl.col("model").str.starts_with("policy_"))
        selected = selected.with_columns((pl.col("prediction") - pl.col("actual")).alias("error"))
        for row in selected.sort(pl.col("error").abs(), descending=True).head(10).to_dicts():
            extremes.append(row)
        for (model, population, year), part in selected.partition_by(
            ["model", "population", "year"], as_dict=True
        ).items():
            diff = (part["prediction"] - part["actual"]).to_numpy()
            cohorts.append(
                dict(
                    position=position,
                    horizon=horizon,
                    model=model,
                    population=population,
                    year=year,
                    n=len(diff),
                    mse=float(np.mean(diff**2)),
                    mae=float(np.mean(abs(diff))),
                    bias=float(np.mean(diff)),
                )
            )
    feature_counts, pair_counts, warnings, capacity = Counter(), Counter(), Counter(), []
    data_name = json.loads((source / "protocol.json").read_text())["data_run"]
    metadata = json.loads((LAB / "runs" / data_name / "features.json").read_text())
    for file in (source / "fits").rglob("*.json"):
        diag = json.loads(file.read_text())
        if diag["year"] < 2019:
            continue
        view = diag["feature_view_indices"]
        for j in diag["discovery"].get("selected_indices", []):
            feature_counts[metadata["names"][view[j]]] += 1
        for a, b in diag["discovery"].get("pairs", []):
            pair_counts[(metadata["names"][view[a]], metadata["names"][view[b]])] += 1
        for warning in diag.get("warnings", []):
            warnings[warning] += 1
        for row in diag.get("capacity", []):
            capacity.append(
                dict(
                    position=diag["position"],
                    horizon=diag["horizon"],
                    year=diag["year"],
                    path=diag["path"]["name"],
                    family=diag["path"]["family"],
                    **row,
                )
            )
    report_manifest = json.loads((report / "manifest.json").read_text())
    capacity_name = report_manifest.get("capacity_source", "deep_capacity_001")
    adaptive = [
        json.loads(p.read_text()) for p in (LAB / "runs" / capacity_name / "fits").rglob("*.json")
    ]
    return dict(
        ensemble_vs_top3=result,
        controlled_feature_views=feature_views,
        heldout_tree_capacity=heldout_capacity,
        existing_library_increment=serving_library,
        input_coverage=input_coverage(LAB / "runs" / data_name),
        last_year_compositions=latest,
        budget_curves=budgets,
        large_errors=extremes,
        cohorts=cohorts,
        discovered_features=feature_counts.most_common(),
        discovered_pairs=[
            dict(a=a, b=b, count=count) for (a, b), count in pair_counts.most_common()
        ],
        optimizer_warnings=dict(warnings),
        training_capacity=capacity,
        adaptive_capacity=adaptive,
        summary=root_summary,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="deep_search_002")
    parser.add_argument("--report", default="deep_report_001")
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if any(Path(s).name != s or s in {".", ".."} for s in [args.source, args.report, args.run_id]):
        raise ValueError("Invalid run id")
    source, report, out = [LAB / "runs" / s for s in [args.source, args.report, args.run_id]]
    manifest = json.loads((report / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("Incomplete ensemble report")
    out.mkdir(exist_ok=False)
    results = collect(report, source)
    for name, value in results.items():
        write_json(out / f"{name}.json", value)
        if name in {
            "ensemble_vs_top3",
            "budget_curves",
            "cohorts",
            "training_capacity",
            "large_errors",
        }:
            pl.DataFrame(value, infer_schema_length=None).write_parquet(out / f"{name}.parquet")
    lines = [
        "# What the larger ensemble search added",
        "",
        "Comparisons below use the same rows and equally weighted test seasons.",
        "",
        "| Position | Horizon | Policy | MSE reduction vs top-three blend | MAE gain | Years won |",
        "|---|---|---|---:|---:|---:|",
    ]
    for r in results["ensemble_vs_top3"]:
        if r["model"] == "policy_mse":
            lines.append(
                f"| {r['position']} | {r['horizon']} | {r['model']} | "
                f"{r['mse_reduction_pct']:+.2f}% | {r['mae_gain']:+.3f} | "
                f"{r['years_won']}/{r['years']} |"
            )
    lines += [
        "",
        "See last_year_compositions.json for the actual 2025 method choices and weights.",
        "Those are retrospective 2025 choices, not newly issued 2026 forecasts.",
        "Discovered feature/pair recurrence describes what training fits used; it is not causal",
        "importance or proof that any one interaction improves predictions.",
        "Training-capacity and library-budget curves are diagnostics, not independent trials.",
        "Larger models can fit the training data better while worsening later forecasts.",
        "input_coverage.json describes nonmissing input cells, including derived values and flags;",
        "it does not mean every cell is a separate directly observed measurement.",
        "controlled_feature_views.json compares matching tree settings across distinct input sets.",
        "heldout_tree_capacity.json compares tree-count prefixes on the same evaluation seasons.",
        "existing_library_increment.json measures adding saved models on matched rows.",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    write_json(
        out / "manifest.json",
        dict(
            status="complete",
            research_only=True,
            source_manifest=digest(source / "manifest.json"),
            report_manifest=digest(report / "manifest.json"),
            implementation_sha256=digest(Path(__file__)),
            artifacts={p.name: digest(p) for p in out.iterdir() if p.is_file()},
        ),
    )


if __name__ == "__main__":
    main()
