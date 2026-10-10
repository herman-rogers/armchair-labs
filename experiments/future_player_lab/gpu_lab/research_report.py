"""Integrate completed RB research evidence without selecting or retraining models.

Run only after campaign postprocessing and profile integration have completed.
All report inputs and generated outputs receive a separate provenance manifest.
"""

# Embedded standalone HTML keeps its own readable line layout.
# ruff: noqa: E501

from __future__ import annotations

import argparse
import html
import json
import math
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import polars as pl

from .run import ROOT, RUNS, STATS, digest, json_write

POLICIES = ["selected", "ensemble_top3", "ensemble_diverse", "ensemble_convex"]
POLICY_LABELS = {
    "selected": "Selected",
    "ensemble_top3": "Top three",
    "ensemble_diverse": "Diverse",
    "ensemble_convex": "Convex",
}
KEY = ["horizon", "year", "origin", "player_id", "target"]


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def fmt(value, places=2):
    if value is None or isinstance(value, float) and not math.isfinite(value):
        return "—"
    if isinstance(value, (float, np.floating)):
        return f"{value:,.{places}f}"
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    return str(value)


def inline(text):
    escaped = html.escape(str(text))
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    return re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', escaped)


def test_summary(path):
    """Read JUnit totals once per leaf suite, without double-counting wrappers."""
    root = ET.parse(path).getroot()
    suites = [node for node in root.iter("testsuite") if not node.findall("testsuite")]
    if not suites:
        raise ValueError(f"No test suites in validation evidence: {path}")
    counts = {
        name: sum(int(suite.get(name, "0")) for suite in suites)
        for name in ("tests", "failures", "errors", "skipped")
    }
    counts["passed"] = counts["tests"] - sum(counts[k] for k in ("failures", "errors", "skipped"))
    if counts["failures"] or counts["errors"]:
        raise ValueError(f"Validation evidence contains failed tests: {path}")
    return counts


class Evidence:
    def __init__(self, root):
        self.root = Path(root)
        self.inputs = {}
        self.expected = {}

    def register(self, folder, manifest):
        for name, expected in manifest.get("files", {}).items():
            self.expected[str((Path(folder) / name).resolve())] = expected

    def bind(self, path):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Required completed evidence is missing: {path}")
        actual = digest(path)
        expected = self.expected.get(str(path.resolve()))
        if expected is not None and actual != expected:
            raise ValueError(f"Completed evidence hash changed: {path}")
        self.inputs[self.relative(path)] = actual
        return path

    def relative(self, path):
        return Path(os.path.relpath(Path(path).resolve(), self.root.resolve())).as_posix()

    def json(self, path):
        return json.loads(self.bind(path).read_text())

    def frame(self, path):
        return pl.read_parquet(self.bind(path))

    def link(self, path, label):
        self.bind(path)
        return f"[{label}]({self.relative(path)})"


class Document:
    def __init__(self):
        self.markdown, self.html = [], []

    def heading(self, text, level=2):
        anchor = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
        self.markdown += ["#" * level + " " + text, ""]
        self.html.append(f'<h{level} id="{anchor}">{html.escape(text)}</h{level}>')

    def paragraph(self, text):
        self.markdown += [text, ""]
        self.html.append(f"<p>{inline(text)}</p>")

    def table(self, rows, columns, *, limit=None):
        rows = rows.to_dicts() if isinstance(rows, pl.DataFrame) else list(rows)
        if limit is not None:
            rows = rows[:limit]
        if not rows:
            self.paragraph("No eligible observations for this comparison.")
            return
        self.markdown += [
            "| " + " | ".join(label for _, label in columns) + " |",
            "|" + "---|" * len(columns),
        ]
        body = []
        for row in rows:
            values = [fmt(row.get(key)) for key, _ in columns]
            self.markdown.append("| " + " | ".join(v.replace("|", "\\|") for v in values) + " |")
            body.append("<tr>" + "".join(f"<td>{inline(v)}</td>" for v in values) + "</tr>")
        self.markdown.append("")
        self.html.append(
            '<div class="table-wrap"><table><thead><tr>'
            + "".join(f"<th>{html.escape(label)}</th>" for _, label in columns)
            + "</tr></thead><tbody>"
            + "".join(body)
            + "</tbody></table></div>"
        )

    def figure(self, relative, caption):
        self.markdown += [f"![{caption}]({relative})", ""]
        self.html.append(
            f'<figure><a href="{html.escape(relative)}"><img loading="lazy" '
            f'src="{html.escape(relative)}" alt="{html.escape(caption)}"></a>'
            f"<figcaption>{html.escape(caption)}</figcaption></figure>"
        )


def annual_policy_scores(predictions, evaluation_years):
    rows = predictions.filter(
        pl.col("year").is_in(evaluation_years) & pl.col("recipe").is_in([*POLICIES, "persistence"])
    )
    baseline = rows.filter(pl.col("recipe") == "persistence").select(
        *KEY, pl.col("prediction").alias("baseline")
    )
    matched = (
        rows.filter(pl.col("recipe").is_in(POLICIES))
        .join(baseline, on=KEY, how="inner", validate="m:1")
        .filter(
            pl.col("actual").is_finite()
            & pl.col("prediction").is_finite()
            & pl.col("baseline").is_finite()
        )
    )
    return (
        matched.group_by("horizon", "target", "recipe", "year")
        .agg(
            pl.len().alias("n"),
            ((pl.col("prediction") - pl.col("actual")) ** 2).mean().alias("mse"),
            ((pl.col("baseline") - pl.col("actual")) ** 2).mean().alias("baseline_mse"),
        )
        .with_columns(
            pl.col("mse").sqrt().alias("rmse"),
            pl.col("baseline_mse").sqrt().alias("baseline_rmse"),
            (100 * (1 - pl.col("mse") / pl.col("baseline_mse"))).alias("mse_improvement_pct"),
        )
        .sort("horizon", "target", "recipe", "year")
    )


def paired_distribution_scores(distributions):
    """Compare distribution scores only on identical known player/origin outcomes."""
    baseline = distributions.filter(pl.col("recipe") == "persistence_calibrated").select(
        *KEY,
        pl.col("crps").alias("baseline_crps"),
        pl.col("coverage80").alias("baseline_coverage80"),
        pl.col("width80").alias("baseline_width80"),
    )
    matched = (
        distributions.filter(pl.col("recipe").is_in(POLICIES))
        .join(baseline, on=KEY, how="inner", validate="m:1")
        .filter(pl.col("crps").is_finite() & pl.col("baseline_crps").is_finite())
    )
    return (
        matched.group_by("horizon", "target", "recipe")
        .agg(
            pl.len().alias("n"),
            *[
                pl.col(name).mean()
                for name in (
                    "crps",
                    "baseline_crps",
                    "coverage80",
                    "baseline_coverage80",
                    "width80",
                    "baseline_width80",
                )
            ],
        )
        .with_columns(
            (pl.col("crps") - pl.col("baseline_crps")).alias("crps_difference"),
            pl.when(pl.col("baseline_crps") > 0)
            .then(100 * (1 - pl.col("crps") / pl.col("baseline_crps")))
            .otherwise(None)
            .alias("crps_improvement_pct"),
        )
        .sort("horizon", "target", "recipe")
    )


def matrix_rows(frame, value, targets=STATS, policy_column="recipe"):
    lookup = {(r["horizon"], r["target"], r[policy_column]): r for r in frame.iter_rows(named=True)}
    output = []
    for horizon in ("season", "remaining"):
        for target in targets:
            anchor = lookup.get((horizon, target, "selected"), {})
            row = {
                "horizon": horizon,
                "target": target,
                "n": anchor.get("n"),
                "baseline_rmse": anchor.get("baseline_rmse"),
                "rmse": anchor.get("rmse"),
            }
            row.update(
                {
                    policy: lookup.get((horizon, target, policy), {}).get(value)
                    for policy in POLICIES
                }
            )
            output.append(row)
    return output


def direct_specification_overlap(selections, recipes):
    """Count repeated direct settings, without changing saved policy membership."""
    recipe_map = {r["id"]: r for r in recipes}
    records = []
    for row in selections.filter(
        pl.col("policy").is_in(["ensemble_top3", "ensemble_diverse"])
        & pl.col("target").is_in(["carries", "targets", "points"])
    ).iter_rows(named=True):
        groups = {}
        for member in row["members"]:
            specification = {
                k: v
                for k, v in recipe_map.get(member, {"baseline": member}).items()
                if k not in {"id", "target_mode"}
            }
            signature = json.dumps(specification, sort_keys=True)
            groups.setdefault(signature, []).append(member)
        duplicates = [members for members in groups.values() if len(members) > 1]
        records.append(
            {
                **{k: row[k] for k in ("horizon", "year", "target", "policy")},
                "members": ", ".join(row["members"]),
                "total_members": len(row["members"]),
                "unique_direct_specifications": len(groups),
                "duplicate_members": len(row["members"]) - len(groups),
                "duplicate_groups": json.dumps(duplicates),
            }
        )
    return pl.DataFrame(records).sort("horizon", "target", "policy", "year")


def strong_baseline_rmse(predictions, affine_predictions, evaluation_years):
    """Use one exact three-way population for the PPR RMSE display."""
    rows = predictions.filter(
        (pl.col("target") == "points") & pl.col("year").is_in(evaluation_years)
    )
    trees = rows.filter(pl.col("recipe") == "ensemble_top3").select(
        *KEY, "actual", pl.col("prediction").alias("ensemble_top3")
    )
    plain = rows.filter(pl.col("recipe") == "persistence").select(
        *KEY, pl.col("prediction").alias("persistence")
    )
    affine = affine_predictions.filter(
        (pl.col("recipe") == "affine_selected")
        & (pl.col("target") == "points")
        & pl.col("year").is_in(evaluation_years)
    ).select(
        *KEY, pl.col("actual").alias("affine_actual"), pl.col("prediction").alias("affine_selected")
    )
    matched = trees.join(plain, on=KEY, validate="1:1").join(affine, on=KEY, validate="1:1")
    np.testing.assert_array_equal(matched["actual"].to_numpy(), matched["affine_actual"].to_numpy())
    matched = matched.filter(
        pl.all_horizontal(
            [
                pl.col(n).is_finite()
                for n in ["actual", "persistence", "affine_selected", "ensemble_top3"]
            ]
        )
    )
    output = []
    for (horizon,), group in matched.group_by("horizon"):
        for recipe in ["persistence", "affine_selected", "ensemble_top3"]:
            residual = group[recipe].to_numpy() - group["actual"].to_numpy()
            output.append(
                {
                    "horizon": horizon,
                    "recipe": recipe,
                    "n": group.height,
                    "rmse": float(np.sqrt(np.mean(residual**2))),
                    "mse": float(np.mean(residual**2)),
                    "mae": float(np.mean(np.abs(residual))),
                }
            )
    return pl.DataFrame(output).sort("horizon", "recipe")


def profile_affine_comparison(profile_predictions, affine_predictions, evaluation_years):
    """Descriptive fixed-recipe comparisons on identical observable PPR outcomes."""
    left = profile_predictions.filter(
        (pl.col("target") == "points")
        & pl.col("year").is_in(evaluation_years)
        & ~pl.col("recipe").is_in(["training_mean", "persistence"])
    ).select(*KEY, "recipe", "actual", "prediction")
    right = affine_predictions.filter(
        (pl.col("target") == "points")
        & (pl.col("recipe") == "affine_selected")
        & pl.col("year").is_in(evaluation_years)
    ).select(
        *KEY, pl.col("actual").alias("baseline_actual"), pl.col("prediction").alias("baseline")
    )
    matched = left.join(right, on=KEY, validate="m:1")
    np.testing.assert_array_equal(
        matched["actual"].to_numpy(), matched["baseline_actual"].to_numpy()
    )
    matched = matched.filter(
        pl.all_horizontal([pl.col(c).is_finite() for c in ("actual", "prediction", "baseline")])
    )
    return (
        matched.group_by("horizon", "recipe")
        .agg(
            pl.len().alias("n"),
            ((pl.col("prediction") - pl.col("actual")) ** 2).mean().alias("mse"),
            ((pl.col("baseline") - pl.col("actual")) ** 2).mean().alias("baseline_mse"),
        )
        .with_columns(
            pl.col("mse").sqrt().alias("rmse"),
            pl.col("baseline_mse").sqrt().alias("baseline_rmse"),
            (100 * (1 - pl.col("mse") / pl.col("baseline_mse"))).alias("mse_improvement_pct"),
        )
        .sort("horizon", "mse", "recipe")
    )


def descriptive_winners(landscape, evaluation_years):
    candidates = landscape.filter(
        pl.col("year").is_in(evaluation_years) & pl.col("family").is_not_null()
    )
    groups = ["horizon", "target", "recipe", "family", "representation", "view", "history"]
    scored = (
        candidates.group_by(groups)
        .agg(
            pl.col("year").n_unique().alias("seasons"),
            pl.col("mse").mean().sqrt().alias("equal_season_rmse"),
            pl.col("normalized_mse").mean().alias("mean_normalized_mse"),
        )
        .filter(pl.col("seasons") == len(evaluation_years))
        .sort("mean_normalized_mse", "recipe")
    )
    family = scored.unique(["horizon", "target", "family"], keep="first", maintain_order=True)
    representation = scored.unique(
        ["horizon", "target", "representation"], keep="first", maintain_order=True
    )
    return family, representation


def graphics(out, annual, cohorts, ablations, strong_rmse):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dest = out / "report/figures"
    dest.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    colors = ["#1768ac", "#13855b", "#bc5c1d", "#8754ac"]
    paths = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    labels = ["Plain\npersistence", "Affine\npersistence", "Top-three\nensemble"]
    recipes = ["persistence", "affine_selected", "ensemble_top3"]
    for ax, horizon in zip(axes, ("season", "remaining"), strict=True):
        subset = strong_rmse.filter(pl.col("horizon") == horizon)
        lookup = {r["recipe"]: r for r in subset.iter_rows(named=True)}
        bars = ax.bar(
            labels, [lookup[r]["rmse"] for r in recipes], color=["#718096", "#8754ac", "#1768ac"]
        )
        ax.bar_label(bars, fmt="%.2f", padding=4)
        ax.set(
            title=f"{horizon}: {lookup['ensemble_top3']['n']:,} matched rows",
            ylabel="Direct PPR RMSE (period points)",
        )
        ax.margins(y=0.16)
    fig.suptitle("Tree improvement beyond an earlier-year affine baseline")
    fig.tight_layout()
    for extension in ("png", "svg", "pdf"):
        path = dest / f"strong_baseline_rmse.{extension}"
        fig.savefig(path, dpi=180)
        paths.append(path)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, horizon in zip(axes, ("season", "remaining"), strict=True):
        for policy, color in zip(POLICIES, colors, strict=True):
            rows = annual.filter(
                (pl.col("horizon") == horizon)
                & (pl.col("target") == "points")
                & (pl.col("recipe") == policy)
            ).sort("year")
            ax.plot(
                rows["year"],
                rows["mse_improvement_pct"],
                marker="o",
                color=color,
                label=POLICY_LABELS[policy],
            )
        ax.axhline(0, color="#777", linewidth=0.7)
        ax.set(
            title=horizon,
            xlabel="Evaluation season",
            ylabel="PPR MSE improvement vs plain persistence (%)",
            xticks=sorted(annual["year"].unique().to_list()),
        )
        ax.legend(fontsize=8)
    fig.suptitle("Does PPR improvement persist across seasons?")
    fig.tight_layout()
    for extension in ("png", "svg", "pdf"):
        path = dest / f"annual_points.{extension}"
        fig.savefig(path, dpi=180)
        paths.append(path)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    for ax, horizon in zip(axes, ("season", "remaining"), strict=True):
        for j, (policy, color) in enumerate(zip(POLICIES, colors, strict=True)):
            subset = cohorts.filter(
                (pl.col("horizon") == horizon)
                & (pl.col("recipe") == policy)
                & (pl.col("cohort") == "high_prior_usage")
            )
            values = dict(zip(subset["target"], subset["mse_improvement_pct"], strict=True))
            ax.barh(
                np.arange(len(STATS)) + (j - 1.5) * 0.18,
                [values.get(t, np.nan) for t in STATS],
                height=0.18,
                color=color,
                label=POLICY_LABELS[policy],
            )
        ax.set_yticks(range(len(STATS)), [s.replace("_", " ") for s in STATS])
        ax.axvline(0, color="#777", linewidth=0.7)
        ax.set(title=horizon, xlabel="MSE improvement over matched persistence (%)")
        ax.legend(fontsize=8)
    axes[0].invert_yaxis()
    fig.suptitle("High prior usage RBs: threshold learned only from earlier seasons")
    fig.tight_layout()
    for extension in ("png", "svg", "pdf"):
        path = dest / f"high_usage.{extension}"
        fig.savefig(path, dpi=180)
        paths.append(path)
    plt.close(fig)
    if ablations.height:
        subset = ablations.filter(pl.col("target") == "points")
        views = sorted(subset["view"].unique().to_list())
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
        for ax, horizon in zip(axes, ("season", "remaining"), strict=True):
            lookup = {
                r["view"]: r["mse_improvement_pct"]
                for r in subset.filter(pl.col("horizon") == horizon).iter_rows(named=True)
            }
            values = [lookup.get(view, np.nan) for view in views]
            ax.barh(views, values, color=["#13855b" if v >= 0 else "#b84b3a" for v in values])
            ax.axvline(0, color="#777", linewidth=0.7)
            ax.set(title=horizon, xlabel="PPR MSE improvement over full-input XGBoost (%)")
        axes[0].invert_yaxis()
        fig.suptitle("Source ablations: fixed engine and hyperparameters")
        fig.tight_layout()
        for extension in ("png", "svg", "pdf"):
            path = dest / f"source_ablations.{extension}"
            fig.savefig(path, dpi=180)
            paths.append(path)
        plt.close(fig)
    return paths


def build_report(
    out,
    *,
    profiles=None,
    profile_predict=None,
    play_context=None,
    benchmark=None,
    dimensions=None,
    draft=None,
    affine=None,
):
    out = Path(out).resolve()
    evidence = Evidence(out)
    state = evidence.json(out / "manifest.json")
    if state.get("status") != "complete":
        raise ValueError(
            "Integrated report requires completed training; partial scores are not read"
        )
    postprocess = evidence.json(out / "postprocess.json")
    if postprocess.get("status") != "complete":
        raise ValueError(
            "Integrated report requires completed analysis, diagnostics, refits and figures"
        )
    profiles = Path(profiles or out / "profiles")
    profile_predict = Path(profile_predict or RUNS / "rb_profile_predict_20261003_001")
    play_context = Path(play_context or RUNS / "rb_gpu_play_context_20261003_001")
    benchmark = Path(benchmark or RUNS / "gpu_benchmark_001")
    dimensions = Path(dimensions or RUNS / "rb_gpu_dimensions_20261003_001")
    draft = Path(draft or RUNS / "rb_gpu_draft_capital_20261003_001")
    affine = Path(affine or RUNS / "rb_gpu_calibrated_baselines_20261003_001")
    config = evidence.json(out / "config.json")
    signature = evidence.json(out / "signature.json")
    analysis = out / "analysis"
    for folder in (analysis, out / "final_models", profiles, profile_predict, play_context):
        manifest = evidence.json(folder / "manifest.json")
        if manifest.get("status", "complete") != "complete":
            raise ValueError(f"Incomplete evidence: {folder}")
        evidence.register(folder, manifest)
    conditional_manifest = evidence.json(profiles / "conditional/manifest.json")
    if conditional_manifest.get("status") != "complete":
        raise ValueError("Profile-conditioned analysis has not completed")
    evidence.register(profiles / "conditional", conditional_manifest)
    coherence_manifest = evidence.json(out / "stat_consistency/manifest.json")
    if coherence_manifest.get("status") != "complete":
        raise ValueError("Stat-consistency diagnostics have not completed")
    evidence.register(out / "stat_consistency", coherence_manifest)
    audit = evidence.json(out / "audit.json")
    if audit.get("status") != "passed":
        raise ValueError("Independent campaign audit has not passed")
    independent_review = evidence.json(out / "independent_review/results.json")
    review_source = evidence.bind(out / "independent_review/review.py")
    evidence.bind(out / "independent_review/README.md")
    if independent_review.get("status") != "complete" or independent_review.get(
        "review_source_sha256"
    ) != digest(review_source):
        raise ValueError("Independent saved-policy review has not completed for the pinned source")
    dimension_manifest = evidence.json(dimensions / "manifest.json")
    if dimension_manifest.get("status") != "complete":
        raise ValueError("Representation-dimension sensitivity study has not completed")
    evidence.register(dimensions, dimension_manifest)
    dimension_training = evidence.json(dimensions / "training/manifest.json")
    dimension_comparisons = evidence.frame(
        dimensions / "analysis/controlled_comparisons.parquet"
    ).filter(pl.col("scope") == "evaluation")
    draft_manifest = evidence.json(draft / "manifest.json")
    if draft_manifest.get("status") != "complete":
        raise ValueError("Draft-capital coverage-gap supplement has not completed")
    evidence.register(draft, draft_manifest)
    draft_audit = evidence.json(draft / "audit.json")
    if draft_audit.get("status") != "passed":
        raise ValueError("Independent draft-capital artifact audit has not passed")
    if draft_audit.get("study_manifest_sha256") != digest(draft / "manifest.json"):
        raise ValueError("Draft-capital audit does not bind the completed study manifest")
    draft_comparisons = evidence.frame(draft / "analysis/controlled_comparisons.parquet").filter(
        pl.col("scope") == "evaluation"
    )
    draft_annual = evidence.frame(draft / "analysis/annual_comparisons.parquet").filter(
        pl.col("scope") == "evaluation"
    )
    affine_manifest = evidence.json(affine / "manifest.json")
    affine_integration = affine / "comparisons" / out.name
    affine_comparison_manifest = evidence.json(affine_integration / "manifest.json")
    if (
        affine_manifest.get("status") != "complete"
        or affine_comparison_manifest.get("status") != "complete"
    ):
        raise ValueError("Stronger simple-baseline integration has not completed")
    evidence.register(affine, affine_manifest)
    evidence.register(affine_integration, affine_comparison_manifest)

    def affine_table(name):
        return evidence.frame(affine_integration / name).filter(
            pl.col("recipe").is_in(POLICIES) & (pl.col("baseline") == "affine_selected")
        )

    affine_matched = affine_table("matched_comparisons.parquet")
    affine_annual = affine_table("annual_comparisons.parquet")
    affine_cohorts = affine_table("cohort_comparisons.parquet")
    affine_annual_cohorts = affine_table("annual_cohort_comparisons.parquet")
    matched = evidence.frame(analysis / "matched_baseline.parquet")
    headline = matched.filter(pl.col("recipe").is_in(POLICIES))
    predictions = evidence.frame(analysis / "predictions.parquet")
    selection_overlap = direct_specification_overlap(
        evidence.frame(analysis / "policy_selection.parquet"), config["recipes"]
    )
    affine_predictions = evidence.frame(affine / "predictions.parquet")
    strong_rmse = strong_baseline_rmse(predictions, affine_predictions, config["evaluation_years"])
    annual = annual_policy_scores(predictions, config["evaluation_years"])
    cohorts = evidence.frame(analysis / "usage_cohorts.parquet")
    distributions = evidence.frame(analysis / "distribution_summary.parquet")
    paired_distributions = paired_distribution_scores(
        evidence.frame(analysis / "distributions.parquet")
    )
    comparisons = evidence.frame(analysis / "controlled_comparisons.parquet")
    landscape = evidence.frame(analysis / "landscape.parquet")
    permutation = evidence.frame(analysis / "source_permutation.parquet")
    native = evidence.frame(analysis / "native_importance.parquet")
    if "target" not in permutation.columns:
        permutation = pl.DataFrame(
            schema={
                "horizon": pl.String,
                "target": pl.String,
                "family": pl.String,
                "delta_mse": pl.Float64,
                "features": pl.Int64,
                "year": pl.Int64,
            }
        )
    if "target" not in native.columns:
        native = pl.DataFrame(
            schema={
                "horizon": pl.String,
                "target": pl.String,
                "recipe": pl.String,
                "feature": pl.String,
                "importance": pl.Float64,
            }
        )
    family, representation = descriptive_winners(landscape, config["evaluation_years"])
    recipe_map = {r["id"]: r for r in config["recipes"]}
    source_ids = []
    for row in comparisons.iter_rows(named=True):
        recipe, control = recipe_map[row["recipe"]], recipe_map[row["control"]]
        if (
            recipe["view"] != "all"
            and recipe["params"] == control["params"]
            and not recipe["history"]
        ):
            source_ids.append(recipe["id"])
    ablations = comparisons.filter(pl.col("recipe").is_in(source_ids))
    controlled = comparisons.filter(pl.col("controlled_representation"))
    permutation_summary = (
        permutation.group_by("horizon", "target", "family")
        .agg(
            pl.col("delta_mse").mean().alias("mean_delta_mse"),
            pl.col("delta_mse").std().alias("repeat_sd"),
            pl.col("features").first(),
            pl.col("year").first(),
        )
        .sort(["horizon", "target", "mean_delta_mse"], descending=[False, False, True])
    )
    profile_methods = evidence.frame(profiles / "method_comparison.parquet")
    profile_skill = evidence.frame(profile_predict / "matched_comparisons.parquet")
    profile_counts = evidence.frame(profile_predict / "cohort_counts.parquet")
    profile_affine = profile_affine_comparison(
        evidence.frame(profile_predict / "predictions.parquet"),
        affine_predictions,
        config["evaluation_years"],
    )
    conditional_path = profiles / "conditional/conditional_errors.parquet"
    conditional = evidence.frame(conditional_path)
    bdb = evidence.frame(play_context / "summary.parquet")
    bdb_uncertainty = evidence.json(
        play_context.with_name(play_context.name + "_uncertainty") / "intervals.json"
    )
    timing = evidence.json(benchmark / "report.json")
    final_models = evidence.json(out / "final_models/manifest.json")
    exports = out / "report/exports"
    exports.mkdir(parents=True, exist_ok=True)
    tables = {
        "headline": headline,
        "annual_policy_scores": annual,
        "usage_cohorts": cohorts,
        "distributions": distributions,
        "paired_distributions": paired_distributions,
        "dimension_sensitivity": dimension_comparisons,
        "draft_capital": draft_comparisons,
        "draft_capital_annual": draft_annual,
        "affine_baseline_comparisons": affine_matched,
        "affine_baseline_annual": affine_annual,
        "affine_baseline_cohorts": affine_cohorts,
        "affine_baseline_annual_cohorts": affine_annual_cohorts,
        "strong_baseline_rmse": strong_rmse,
        "direct_specification_overlap": selection_overlap,
        "profile_affine_comparison": profile_affine,
        "controlled_representations": controlled,
        "source_ablations": ablations,
        "best_family_descriptive": family,
        "best_representation_descriptive": representation,
        "source_permutation": permutation_summary,
        "native_importance": native,
        "profile_predictions": profile_skill,
        "profile_conditioned_errors": conditional,
    }
    for name, table in tables.items():
        table.write_csv(exports / f"{name}.csv")
    images = graphics(out, annual, cohorts, ablations, strong_rmse)
    doc = Document()
    doc.heading("Running back GPU research: integrated findings", 1)
    doc.paragraph(
        f"Completed {state['completed_fits']:,} target/fold fits across "
        f"{len(config['recipes'])} frozen recipes, two forecast horizons, eight targets "
        f"and {len(config['years'])} chronological forecast years. "
        f"The policy comparisons below evaluate {config['evaluation_years'][0]}–"
        f"{config['evaluation_years'][-1]}; each year's selection uses only the previous "
        f"{config['selection_years']} years. These are historical development results: "
        "the data, including 2025, had already been studied."
    )
    for horizon in config["horizons"]:
        row = headline.filter(
            (pl.col("horizon") == horizon)
            & (pl.col("target") == "points")
            & (pl.col("recipe") == "ensemble_top3")
        ).row(0, named=True)
        stronger = affine_matched.filter(
            (pl.col("horizon") == horizon)
            & (pl.col("target") == "points")
            & (pl.col("recipe") == "ensemble_top3")
        )
        if stronger.height:
            change = stronger["mse_improvement_pct"][0]
            direction = "reduces" if change >= 0 else "increases"
            doc.paragraph(
                f"{horizon.capitalize()} direct PPR: the prespecified top-three tree ensemble "
                f"{direction} matched MSE by {fmt(abs(change))}% versus the stronger "
                f"affine-persistence baseline (RMSE {fmt(stronger['rmse'][0])} versus "
                f"{fmt(stronger['baseline_rmse'][0])}). Its MSE reduction versus plain "
                f"persistence is {fmt(row['mse_improvement_pct'])}% "
                f"(plain-persistence RMSE {fmt(row['baseline_rmse'])})."
            )
        high_filter = (
            (pl.col("horizon") == horizon)
            & (pl.col("target") == "points")
            & (pl.col("recipe") == "ensemble_top3")
            & (pl.col("usage_cohort") == "high_prior_usage")
        )
        high = affine_cohorts.filter(high_filter)
        high_years = affine_annual_cohorts.filter(high_filter)
        if high.height:
            high_change = high["mse_improvement_pct"][0]
            high_direction = "lower" if high_change >= 0 else "higher"
            doc.paragraph(
                f"For {horizon} RBs with high prior usage, top-three PPR MSE is "
                f"{fmt(abs(high_change))}% {high_direction} than affine persistence; "
                f"{high_years.filter(pl.col('mse_improvement_pct') > 0).height} of "
                f"{high_years.height} evaluation seasons improve."
            )
    doc.paragraph(
        "Top-three is a prespecified headline policy, not a winner selected from these "
        "results; all four policies are reported below. The stronger affine-baseline check "
        "was added after the completed preseason preview and remains development analysis. "
        "GPU support and hardware throughput are computing findings; predictive accuracy "
        "is assessed from the matched forecast losses, regardless of tree library."
    )
    doc.paragraph(
        "Start with the year-by-year and high-usage tables before interpreting pooled gains. "
        "A large inactive/low-volume population can make overall accuracy look better "
        "than accuracy for fantasy-relevant workloads."
    )
    doc.paragraph(
        evidence.link(analysis / "readout.md", "Core forecast readout")
        + " · "
        + evidence.link(profiles / "index.html", "Interactive player profiles")
        + " · "
        + evidence.link(profiles / "conditional/index.html", "Profile-conditioned errors")
        + " · "
        + evidence.link(out / "final_models/policies.json", "Reusable final policies")
    )
    player_browser = profiles / "players/index.html"
    if player_browser.exists():
        doc.paragraph(
            evidence.link(player_browser, "Search named players and their historical profiles")
        )
    doc.heading("Forecast accuracy across all targets")
    doc.paragraph(
        "All entries are percentage reductions in pooled-row MSE on identical player/origin "
        "rows versus prior calendar-week production persistence. Selected chooses one recipe; "
        "top three averages the three lowest earlier losses; diverse blends different model "
        "families; convex fits nonnegative weights using earlier forecasts. These four "
        "policies were specified before their evaluated outcomes. Top-three counts "
        "recipes: factorized challengers retain their raw-anchor direct specifications "
        "for carries, targets and PPR, so those selections can contain duplicate direct "
        "settings. The diverse policy requires different engine families."
    )
    overlap_summary = (
        selection_overlap.group_by("horizon", "target", "policy")
        .agg(
            pl.len().alias("evaluation_years"),
            (pl.col("duplicate_members") > 0).sum().alias("years_with_duplicates"),
            pl.col("unique_direct_specifications").mean().alias("mean_distinct_specifications"),
        )
        .sort("horizon", "target", "policy")
    )
    doc.table(
        overlap_summary,
        [
            ("horizon", "Horizon"),
            ("target", "Target"),
            ("policy", "Policy"),
            ("years_with_duplicates", "Years with duplicate direct settings"),
            ("evaluation_years", "Evaluation years"),
            ("mean_distinct_specifications", "Mean distinct specifications"),
        ],
    )
    doc.paragraph(
        "This counts actual saved policy membership and identical direct estimator/input "
        "settings, not a numerical identity test of CUDA forecasts. Membership remains "
        "unchanged; annual recipe lists and duplicate groups are exported for review. "
        "The four-policy accuracy tables retain the diverse-policy comparison."
    )
    doc.paragraph(
        "Season forecasts cover the full regular season from a preseason origin. "
        "Remaining-season forecasts start after completed Week 2. Target units are "
        "period totals, with per-calendar-week rates used inside the fitting procedure."
    )
    doc.table(
        matrix_rows(headline, "mse_improvement_pct"),
        [
            ("horizon", "Horizon"),
            ("target", "Target"),
            ("n", "Rows"),
            ("baseline_rmse", "Baseline RMSE"),
            ("rmse", "Selected RMSE"),
            *[(p, POLICY_LABELS[p] + " MSE %") for p in POLICIES],
        ],
    )
    point_components = headline.filter(pl.col("target") == "points_from_stats")
    doc.table(
        point_components.sort("horizon", "recipe"),
        [
            ("horizon", "Horizon"),
            ("recipe", "Policy"),
            ("rmse", "PPR from stat means RMSE"),
            ("mse_improvement_pct", "MSE improvement %"),
        ],
    )
    doc.paragraph(
        "PPR from stat means adds the five scoring components after independently forecasting "
        "the seven stats. It is distinct from the direct PPR model and does not specify a "
        "joint fantasy-points distribution. Scoring is rushing/receiving yards ÷ 10, plus "
        "receptions, plus six times rushing/receiving touchdowns; fumbles, passing, bonuses, "
        "returns and conversions are outside this contract."
    )
    consistency = out / "stat_consistency/readout.md"
    if consistency.exists():
        consistency_forecasts = evidence.frame(
            out / "stat_consistency/forecast_consistency.parquet"
        )
        consistency_observed = evidence.frame(out / "stat_consistency/observed_consistency.parquet")
        doc.table(
            consistency_forecasts.filter(
                (pl.col("recipe") == "ensemble_top3")
                & pl.col("check").is_in(
                    ["receptions_exceed_targets", "direct_points_minus_component_means"]
                )
            ),
            [
                ("horizon", "Horizon"),
                ("check", "Top-three consistency check"),
                ("n", "Known rows"),
                ("violation_pct", "Violations %"),
                ("mean_absolute_difference", "Mean absolute direct/component PPR gap"),
            ],
        )
        doc.paragraph(
            "Independent marginal stat means can violate cross-stat relationships. "
            f"The observed labels have {consistency_observed['violations'].sum():,} "
            "violations of these expected-count relationships. Jointly coherent means "
            "are a concrete next development objective; any repair would need a new "
            "chronological evaluation. "
            + evidence.link(
                consistency, "Inspect reception/opportunity and PPR coherence diagnostics"
            )
            + "; this check describes saved forecasts and does not tune or repair evaluated predictions."
        )
    doc.figure(
        "analysis/figures/predictive_skill.png", "Forecast skill relative to matched persistence"
    )
    doc.heading("A stronger simple baseline")
    doc.figure(
        "report/figures/strong_baseline_rmse.png",
        "Direct PPR RMSE on the same rows for plain persistence, affine_selected and the top-three ensemble",
    )
    doc.paragraph(
        "An affine-persistence forecast learns an intercept and nonnegative slope from "
        "strictly earlier labels, allowing persistence to shrink toward a population mean. "
        "Two fixed variants use expanding history or three earlier seasons; affine_selected "
        "chooses between them from the previous three years' normalized MSE. This check was "
        "specified after the completed preseason preview and does not change the main tree "
        "policies. It adds no covariates or prediction-interval claims. The table compares "
        "each main policy to affine_selected on identical rows."
    )
    doc.table(
        matrix_rows(affine_matched, "mse_improvement_pct"),
        [
            ("horizon", "Horizon"),
            ("target", "Target"),
            ("n", "Matched rows"),
            ("baseline_rmse", "Affine RMSE"),
            ("rmse", "Selected tree RMSE"),
            *[(p, POLICY_LABELS[p] + " MSE reduction %") for p in POLICIES],
        ],
    )
    affine_stability = (
        affine_annual.filter((pl.col("target") == "points") & (pl.col("recipe") == "ensemble_top3"))
        .join(
            affine_annual_cohorts.filter(
                (pl.col("target") == "points")
                & (pl.col("recipe") == "ensemble_top3")
                & (pl.col("usage_cohort") == "high_prior_usage")
            ).select(
                "horizon",
                "year",
                pl.col("n").alias("high_n"),
                pl.col("mse_improvement_pct").alias("high_skill"),
            ),
            on=["horizon", "year"],
            how="left",
            validate="1:1",
        )
        .sort("horizon", "year")
    )
    doc.table(
        affine_stability,
        [
            ("horizon", "Horizon"),
            ("year", "Year"),
            ("n", "Rows"),
            ("rmse", "Top-three PPR RMSE"),
            ("baseline_rmse", "Affine RMSE"),
            ("mse_improvement_pct", "All-row MSE reduction %"),
            ("high_n", "High-usage rows"),
            ("high_skill", "High-usage MSE reduction %"),
        ],
    )
    doc.paragraph(
        "Top-three is a prespecified display policy, not a winner selected from this table. "
        "Annual and high-usage results matter even when the pooled average improves. "
        "The linked comparison intervals bootstrap equal-season error differences; table "
        "RMSE and percentage reductions pool player rows. Affine point recalibration is "
        "distinct from persistence_calibrated, the residual-distribution baseline below. "
        + evidence.link(affine / "readout.md", "Simple-baseline fitting and chronology")
        + " · "
        + evidence.link(affine_integration / "readout.md", "Matched tree-versus-affine evidence")
    )
    doc.heading("Stability across years and prior workload")
    stability = []
    for horizon in config["horizons"]:
        for target in STATS:
            row = {"horizon": horizon, "target": target}
            for policy in POLICIES:
                block = annual.filter(
                    (pl.col("horizon") == horizon)
                    & (pl.col("target") == target)
                    & (pl.col("recipe") == policy)
                )
                row[policy] = (
                    f"{block.filter(pl.col('mse_improvement_pct') > 0).height}/{block.height}"
                )
            stability.append(row)
    doc.table(
        stability,
        [
            ("horizon", "Horizon"),
            ("target", "Target"),
            *[(p, POLICY_LABELS[p] + " positive years") for p in POLICIES],
        ],
    )
    doc.figure(
        "report/figures/annual_points.png",
        "Annual PPR MSE improvement versus plain persistence for the four prespecified policies",
    )
    doc.paragraph(
        "The following table restricts each evaluation year to RBs whose known prior "
        "carries-plus-targets per calendar week meet or exceed the 75th percentile of positive "
        "usage observed in earlier years. Membership is determined before the outcome. "
        "The same four policies are evaluated without selecting a new high-usage winner."
    )
    high = cohorts.filter(pl.col("cohort") == "high_prior_usage")
    doc.table(
        matrix_rows(high, "mse_improvement_pct"),
        [
            ("horizon", "Horizon"),
            ("target", "Target"),
            ("n", "High-usage rows"),
            ("baseline_rmse", "Baseline RMSE"),
            ("rmse", "Selected RMSE"),
            *[(p, POLICY_LABELS[p] + " MSE %") for p in POLICIES],
        ],
    )
    doc.figure("report/figures/high_usage.png", "High-usage RB accuracy across all eight targets")
    doc.table(
        cohorts.filter((pl.col("target") == "points") & (pl.col("recipe") == "ensemble_top3")).sort(
            "horizon", "cohort"
        ),
        [
            ("horizon", "Horizon"),
            ("cohort", "Prior-usage cohort"),
            ("n", "Rows"),
            ("rmse", "Top-three RMSE"),
            ("baseline_rmse", "Baseline RMSE"),
            ("mse_improvement_pct", "MSE improvement %"),
        ],
    )
    doc.paragraph(
        "Uncertainty intervals in the core evidence resample five seasons and describe the "
        "equal-season mean MSE difference. Headline percentages and RMSE pool player rows; "
        "these weightings differ. Five resampling blocks give coarse uncertainty, and repeated "
        "players/teams plus repeated historical research are not fully accounted for."
    )
    doc.heading("Which tree families and representations worked?")
    doc.paragraph(
        "The tables in this section describe the completed search. Choosing their best row "
        "after inspecting 2021–2025 is not an independent selected-policy test. Family winners "
        "may differ in hyperparameters, history windows and source views; controlled "
        "representation comparisons below hold engine settings and input rows fixed. "
        "Each descriptive winner minimizes mean annual MSE divided by that year's observed "
        "target variance; the displayed RMSE is the square root of mean annual MSE."
    )
    search_rows = []
    for model_family in sorted({r["family"] for r in config["recipes"]}):
        library = [r for r in config["recipes"] if r["family"] == model_family]
        search_rows.append(
            {
                "family": model_family,
                "recipes": len(library),
                "representations": len({r["representation"]["kind"] for r in library}),
                "source_views": len({r["view"] for r in library}),
                "history_windows": ", ".join(
                    str(y) if y else "all earlier" for y in sorted({r["history"] for r in library})
                ),
                "factorized_recipes": sum(r.get("target_mode") == "factorized" for r in library),
            }
        )
    doc.table(
        search_rows,
        [
            ("family", "Tree family"),
            ("recipes", "Recipes"),
            ("representations", "Representations"),
            ("source_views", "Source views"),
            ("history_windows", "Training-history years"),
            ("factorized_recipes", "Opportunity × efficiency recipes"),
        ],
    )
    doc.paragraph(
        "The library combines fixed engine/representation comparisons, seeded draws of "
        "tree counts, depths, learning rates and regularization, explicit history windows, "
        "source ablations, and opportunity-times-efficiency challengers. The complete "
        "parameters were frozen before fitting. This finite tested search is not "
        "exhaustive hyperparameter optimization. "
        + evidence.link(out / "config.json", "Inspect every recipe")
    )
    doc.table(
        family.filter(pl.col("target") == "points").sort("horizon", "equal_season_rmse"),
        [
            ("horizon", "Horizon"),
            ("family", "Family"),
            ("recipe", "Descriptive best recipe"),
            ("representation", "Representation"),
            ("view", "Source view"),
            ("equal_season_rmse", "Equal-season PPR RMSE"),
        ],
    )
    doc.table(
        representation.filter(pl.col("target") == "points").sort("horizon", "equal_season_rmse"),
        [
            ("horizon", "Horizon"),
            ("representation", "Representation"),
            ("family", "Best family"),
            ("equal_season_rmse", "Equal-season PPR RMSE"),
        ],
    )
    doc.figure(
        "analysis/figures/representations.png", "Controlled representation effects for direct PPR"
    )
    fitted_representations = sorted({r["representation"]["kind"] for r in config["recipes"]})
    doc.paragraph(
        "Fitted representations: "
        + ", ".join(f"`{name}`" for name in fitted_representations)
        + ". Correlation pruning removes redundant inputs; PCA/SVD compress inputs into "
        "components, while random projection uses a seeded random compression. Feature "
        "clustering selects representative input columns; the player profiles below cluster "
        "player rows. Raw-plus-PCA retains usable inputs alongside compressed components. "
        "Source-balanced PCA gives each source block a comparable contribution. Missingness, "
        "imputation, scaling and all learned reductions use only that fold's earlier years. "
        "An unsupervised method can still leak future information if fitted outside its "
        "training fold; descriptive all-year profiles are kept separate here."
    )
    doc.heading("Representation dimension sensitivity")
    doc.paragraph(
        f"A separate prespecified supplement completed {dimension_training['completed_fits']:,} "
        "additional target/fold fits: PCA, source-balanced PCA, SVD and random projection at "
        "16 and 128 components, using the fixed XGBoost anchor. Verified raw-input and "
        "48-component controls come from the original campaign. This grid was specified "
        "before examining candidate scores and does not change the main 52 recipes or "
        "the four chronological policies. The comparisons below describe 2021–2025; "
        "selecting the best dimension after seeing these results is development analysis. "
        "Dimensions are requested component counts; the supplement records each fitted "
        "projection's effective output width."
    )
    doc.table(
        dimension_comparisons.filter(pl.col("target") == "points").sort(
            "horizon", "representation", "n_components"
        ),
        [
            ("horizon", "Horizon"),
            ("representation", "Representation"),
            ("n_components", "Requested components"),
            ("n", "Matched rows"),
            ("rmse", "PPR RMSE"),
            ("raw_rmse", "Raw RMSE"),
            ("reference48_rmse", "48-component RMSE"),
            ("mse_improvement_pct", "MSE reduction vs raw %"),
            ("vs_48_mse_improvement_pct", "MSE reduction vs 48 %"),
        ],
    )
    doc.paragraph(
        evidence.link(dimensions / "readout.md", "Dimension-sensitivity methods and all targets")
        + " · "
        + evidence.link(dimensions / "manifest.json", "Supplement provenance and verified controls")
    )
    doc.heading("Source ablations and feature evidence")
    doc.table(
        ablations.filter(pl.col("target") == "points").sort("horizon", "view"),
        [
            ("horizon", "Horizon"),
            ("view", "Input view"),
            ("rmse", "PPR RMSE"),
            ("control_rmse", "Full-input control RMSE"),
            ("mse_improvement_pct", "MSE improvement %"),
        ],
    )
    if ablations.height:
        doc.figure("report/figures/source_ablations.png", "Fixed-XGBoost source ablations")
    important = (
        permutation_summary.filter(pl.col("target") == "points")
        .group_by("horizon", maintain_order=True)
        .head(5)
    )
    doc.table(
        important,
        [
            ("horizon", "Horizon"),
            ("family", "Permuted source"),
            ("features", "Columns"),
            ("mean_delta_mse", "Mean increase in PPR MSE"),
            ("repeat_sd", "Repeat SD"),
        ],
    )
    doc.table(
        native.filter(pl.col("target") == "points")
        .sort("importance", descending=True)
        .group_by("horizon", maintain_order=True)
        .head(5),
        [
            ("horizon", "Horizon"),
            ("recipe", "Selected model"),
            ("feature", "Native feature/component"),
            ("importance", "Importance"),
        ],
    )
    doc.paragraph(
        "Source permutation diagnostics use the final held-out year and never feed back "
        "into the fitted feature set. Correlated groups can substitute for one another; "
        "joint shuffling can also create implausible combinations. Negative or noisy effects "
        "do not prove a source is useless. Native importance is learned in training and "
        "is biased toward features favored by the estimator; neither diagnostic is causal. "
        "In the remaining-season PPR model, missing:current__pbp_rush__field_goal_attempt "
        "and missing:current__pbp_rush__complete_pass are missingness indicators. They can "
        "proxy whether rushing records exist or source coverage, and should not be read "
        "as evidence of kicking or passing skill. Importance values are not additive "
        "shares of explained outcome variance."
    )
    doc.heading("Draft capital: an audited coverage gap")
    doc.paragraph(
        "The main 4,810-input campaign contains no draft-capital feature. It deliberately "
        "excluded combine draft-event fields, while mixed integer/string roster draft_number "
        "schemas promoted that field to text, which numeric extraction omitted. These are "
        "valid known-as-of event fields, so the omission is a coverage gap rather than a "
        "finding of no predictive value. An isolated supplement retains every main input "
        "and appends overall pick, draft round and years since the recorded draft year. "
        "Exact-ID joins and event/observation cutoffs apply; unknown values stay missing."
    )
    doc.table(
        draft_comparisons.filter(pl.col("target") == "points").sort("horizon", "family", "cohort"),
        [
            ("horizon", "Horizon"),
            ("family", "Fixed anchor"),
            ("cohort", "Cohort"),
            ("n", "Matched rows"),
            ("rmse", "PPR RMSE with draft fields"),
            ("control_rmse", "Original raw RMSE"),
            ("mse_improvement_pct", "MSE reduction %"),
        ],
    )
    draft_stability = (
        draft_annual.filter(
            (pl.col("target") == "points")
            & pl.col("cohort").is_in(["all", "known_current_draft_year"])
        )
        .group_by("horizon", "family", "cohort")
        .agg(
            pl.len().alias("years"),
            (pl.col("mse_improvement_pct") > 0).sum().alias("improved_years"),
            pl.col("mse_improvement_pct").min().alias("worst_skill"),
            pl.col("year").sort_by("mse_improvement_pct").first().alias("worst_year"),
        )
        .sort("horizon", "family", "cohort")
    )
    doc.table(
        draft_stability,
        [
            ("horizon", "Horizon"),
            ("family", "Fixed anchor"),
            ("cohort", "Cohort"),
            ("improved_years", "Years with lower PPR MSE"),
            ("years", "Evaluation years"),
            ("worst_year", "Worst year"),
            ("worst_skill", "Worst-year MSE reduction %"),
        ],
    )
    doc.paragraph(
        f"The study adds {draft_manifest['completed_fits']:,} evaluation target fits and "
        f"{draft_manifest['final_refit_fits']:,} separately counted all-through-2025 refits. "
        "LightGBM and XGBoost retain their main raw-anchor parameters, seeds and threads; "
        "the population and per-target unknown masks are unchanged. dated_draft_entrants "
        "means the original dated-draft candidate basis; known_current_draft_year also "
        "includes rookies already observed by Week 2 with a verified draft year. Neither "
        "claims complete undrafted-rookie coverage. The grid is fixed before its outcomes, "
        "but was added after the preseason preview and is not promoted into the 52-recipe "
        "main policies. "
        + evidence.link(draft / "readout.md", "Draft-capital study and reusable bundles")
        + " · "
        + evidence.link(draft / "source_audit.json", "Draft field/schema/time audit")
        + " · "
        + evidence.link(draft / "coverage.json", "Known draft-feature coverage")
        + " · "
        + evidence.link(draft / "audit.json", "Independent draft model/provenance replay audit")
    )
    doc.heading("Predictive distributions")
    doc.paragraph(
        "Point metrics above use plain `persistence`; uncertainty metrics here use "
        "`persistence_calibrated`. The following CRPS comparison joins each policy to "
        "calibrated persistence on "
        "identical player/origin/target rows with finite scores. Lower CRPS is better; "
        "positive percentage reductions favor the policy. Coverage and interval width "
        "below use those same matched rows. Coverage is a fraction; interval width uses "
        "the target's original units. These summary means pool player rows."
    )
    doc.table(
        paired_distributions.filter(pl.col("recipe").is_in(["selected", "ensemble_top3"])),
        [
            ("horizon", "Horizon"),
            ("target", "Target"),
            ("recipe", "Policy"),
            ("n", "Matched rows"),
            ("crps", "CRPS"),
            ("baseline_crps", "Baseline CRPS"),
            ("crps_improvement_pct", "CRPS reduction %"),
            ("coverage80", "80% coverage"),
            ("baseline_coverage80", "Baseline coverage"),
            ("width80", "80% width"),
            ("baseline_width80", "Baseline width"),
        ],
    )
    doc.paragraph(
        "The next table reports each policy's available population, which can differ when "
        "persistence is unknown. Coverage and interval width describe the same saved "
        "predictive samples as CRPS."
    )
    doc.table(
        distributions.filter(
            pl.col("recipe").is_in(["selected", "ensemble_top3", "persistence_calibrated"])
        ).sort("horizon", "target", "recipe"),
        [
            ("horizon", "Horizon"),
            ("target", "Target"),
            ("recipe", "Policy"),
            ("n", "Rows"),
            ("crps", "CRPS"),
            ("coverage80", "80% coverage (fraction)"),
            ("width80", "80% width"),
        ],
    )
    doc.figure(
        "analysis/figures/calibration.png",
        "Measured coverage of earlier-residual intervals for the top-three recipe ensemble",
    )
    doc.paragraph(
        "Lower CRPS is better within a target and horizon. Coverage alone is insufficient: "
        "wide intervals can cover more outcomes while making weaker forecasts. Residual "
        "scores use each policy's finite forecasts and labels; listed row counts can differ "
        "when persistence is unknown, so unequal-population CRPS comparisons are descriptive. "
        "Residual "
        "calibration uses earlier out-of-season forecasts, but selection and calibration "
        "reuse those earlier years and can be optimistic. Count distributions are censored "
        "at zero; yardage remains signed. No joint seven-stat distribution is claimed."
    )
    doc.heading("Unsupervised RB profiles and conditional predictability")
    doc.table(
        profile_methods.filter(pl.col("k") == 5).sort("horizon", "variant"),
        [
            ("horizon", "Horizon"),
            ("variant", "Geometry / cluster method"),
            ("silhouette", "Silhouette"),
            ("mean_seed_ari", "Seed stability ARI"),
            ("explained_variance", "Explained variance fraction"),
        ],
    )
    doc.paragraph(
        "Profile plots describe all available years and examine k=3, 5 and 8. Separate "
        "earlier-year k=5 fits define prediction cohorts. A profile number is local to "
        "its year and geometry; it is not a stable named football role across fits. "
        "Silhouette, seed stability and mixture BIC describe geometry, not forecasting skill."
    )
    profile_rows = []
    fixed = profile_skill.filter(
        (pl.col("target") == "points") & ~pl.col("recipe").is_in(["training_mean", "persistence"])
    )
    for recipe in sorted(fixed["recipe"].unique().to_list()):
        row = {"recipe": recipe}
        for r in fixed.filter(pl.col("recipe") == recipe).iter_rows(named=True):
            row[f"{r['horizon']}_{r['baseline']}"] = r["mse_improvement_pct"]
        profile_rows.append(row)
    doc.table(
        profile_rows,
        [
            ("recipe", "Fixed profile-mean recipe"),
            ("season_training_mean", "Season vs mean %"),
            ("season_persistence", "Season vs persistence %"),
            ("remaining_training_mean", "Remaining vs mean %"),
            ("remaining_persistence", "Remaining vs persistence %"),
        ],
    )
    doc.paragraph(
        f"The profile-only benchmark evaluates twelve fixed hard/soft membership and "
        f"shrinkage recipes; it does not select one using the evaluated outcomes. "
        f"{profile_counts.filter(pl.col('small_cohort')).height:,} training profile/target "
        "cells have weighted or effective counts below 20. Profile-conditioned means are "
        "a separate test of useful profile information, rather than evidence that visually "
        "distinct clusters must improve tree forecasts. "
        + evidence.link(profile_predict / "readout.md", "Profile forecast evidence")
    )
    doc.table(
        profile_affine.group_by("horizon", maintain_order=True).head(1),
        [
            ("horizon", "Horizon"),
            ("recipe", "Retrospective lowest-MSE fixed profile recipe"),
            ("n", "Matched rows"),
            ("rmse", "Profile PPR RMSE"),
            ("baseline_rmse", "Affine PPR RMSE"),
            ("mse_improvement_pct", "Profile MSE reduction vs affine %"),
        ],
    )
    doc.paragraph(
        "Even this retrospective best fixed profile recipe is weaker than affine_selected "
        "for PPR at both horizons. A gain against unshrunk persistence therefore does not "
        "establish competitive standalone profile forecasts. Profile geometry is useful here "
        "for describing players and diagnosing where the tree forecasts fail. The exported "
        "table retains all twelve fixed recipes; this retrospective comparison does not "
        "define or evaluate a selected profile policy."
    )
    reliable = conditional.filter(
        (pl.col("target") == "points") & (pl.col("recipe") == "ensemble_top3") & (pl.col("n") >= 20)
    )
    local_summary = (
        reliable.group_by("variant", "horizon")
        .agg(
            pl.len().alias("annual_profile_cells"),
            (pl.col("mse_improvement_pct") > 0).sum().alias("improved_cells"),
            pl.col("mse_improvement_pct").median().alias("median_skill"),
            pl.col("mse_improvement_pct").min().alias("worst_skill"),
            pl.col("mse_improvement_pct").max().alias("best_skill"),
            pl.col("coverage80").min().alias("minimum_coverage80"),
            pl.col("coverage80").max().alias("maximum_coverage80"),
        )
        .sort("horizon", "variant")
    )
    doc.table(
        local_summary,
        [
            ("horizon", "Horizon"),
            ("variant", "Profile method"),
            ("annual_profile_cells", "Annual cohorts, n≥20"),
            ("improved_cells", "Improved cohorts"),
            ("median_skill", "Median PPR MSE gain %"),
            ("worst_skill", "Worst %"),
            ("best_skill", "Best %"),
            ("minimum_coverage80", "Lowest 80% interval coverage (fraction)"),
            ("maximum_coverage80", "Highest coverage (fraction)"),
        ],
    )
    doc.paragraph(
        "These conditional results use the prespecified top-three policy and keep year-local "
        "profile IDs separate. The best/worst cells describe heterogeneity after evaluation; "
        "they are not a new selection rule. Small groups, low-variance outcomes and dependence "
        "can make percentage errors unstable. Near-nominal overall interval coverage can "
        "hide substantial undercoverage and overcoverage in these annual subgroups. "
        + evidence.link(
            profiles / "conditional/index.html", "Inspect annual cohort descriptions and bias"
        )
    )
    doc.heading("Separate handoff-level context experiment")
    doc.table(
        bdb,
        [
            ("model", "Handoff model"),
            ("plays", "2019 plays"),
            ("rmse", "Yardage RMSE"),
            ("mae", "MAE"),
            ("crps", "CRPS"),
            ("covered80", "80% coverage (fraction)"),
            ("width80", "80% width"),
        ],
    )
    doc.table(
        [r for r in bdb_uncertainty["results"] if r["cluster"] == "GameId"],
        [
            ("comparator", "All-player context versus"),
            ("metric", "Metric difference"),
            ("mean", "Difference"),
            ("low95", "95% low"),
            ("high95", "95% high"),
        ],
    )
    doc.paragraph(
        "This asks a different question: given a rushing play already at handoff, does "
        "observed motion and surrounding-player context help forecast play yards? Six "
        "fixed XGBoost parameter configurations across three feature views produce eighteen "
        "fits using 2017 training and 2018 selection/calibration; "
        "three selected refits evaluate 7,273 RB carries from the partial 2019 archive. "
        "All 21 fits used verified CUDA. All ten teammates and eleven defenders contribute "
        "spatial/motion summaries. Legacy IDs join snapshots within a play only. "
        "The game-block intervals do not remove crossed player dependence, and this gain "
        "does not establish any season-forecast improvement. "
        + evidence.link(play_context / "readout.md", "Handoff readout and artifacts")
        + "."
    )
    doc.heading("GPU execution and practical throughput")
    timing_rows = []
    for name in sorted({r["id"] for r in timing["fits"]}):
        rows = [r for r in timing["fits"] if r["id"] == name]
        warm = [r for r in rows if r["timing_state"] == "warm_fit"]
        timing_rows.append(
            {
                "configuration": name,
                "first_seconds": rows[0]["total_seconds"],
                "warm_seconds": float(np.median([r["total_seconds"] for r in warm])),
                "warm_fit_seconds": float(np.median([r["fit_seconds"] for r in warm])),
                "rmse": float(np.mean([r["rmse"] for r in rows])),
                "device": rows[0]["execution"]["effective_device"],
            }
        )
    doc.table(
        timing_rows,
        [
            ("configuration", "Backend"),
            ("device", "Verified device"),
            ("first_seconds", "First total s"),
            ("warm_seconds", "Warm total s"),
            ("warm_fit_seconds", "Warm fit s"),
            ("rmse", "Yardage RMSE"),
        ],
    )
    doc.paragraph(
        f"Benchmark inputs: {timing['train_rows']:,} training and {timing['test_rows']:,} "
        f"test rows, {timing['raw_features']:,} raw-derived columns, "
        f"{timing['model_features']:,} post-imputation features; one preseason rushing-yard "
        "target. Only XGBoost CPU/CUDA settings are matched algorithmically. Other engines "
        "differ statistically. First-fit timing excludes interpreter startup; driver/filesystem "
        "caches and unrelated GPU activity are not fully controlled. These numbers do not "
        "establish a whole-campaign speedup. NVIDIA cuML support establishes successful "
        "GPU execution, not superior forecast accuracy. "
        + evidence.link(benchmark / "report.json", "Timing details and prediction differences")
    )
    doc.heading("Data coverage, repairs and reproducibility")
    dataset_rows = []
    raw_assets = None
    for horizon in config["horizons"]:
        inputs = evidence.json(out / horizon / "inputs.json")
        meta = inputs["provenance"]
        dataset_rows.append(
            {
                "horizon": horizon,
                "rows": meta["rows"],
                "columns": meta["columns"],
                "players": meta["unique_players"],
                "years": f"{min(meta['years'])}–{max(meta['years'])}",
                "raw_sha256": meta["contract"]["raw_manifest_sha256"],
            }
        )
        evidence.bind(Path(meta["cache_directory"]) / "dataset.json")
        if raw_assets is None:
            raw_assets = evidence.frame(
                Path(meta["cache_directory"]) / "raw_catalog.parquet"
            ).height
    doc.table(
        dataset_rows,
        [
            ("horizon", "Horizon"),
            ("rows", "Player/origin rows"),
            ("players", "Unique RBs"),
            ("columns", "Raw-derived inputs"),
            ("years", "Forecast years"),
        ],
    )
    ref = evidence.json(ROOT / "data/releases/current.json")
    doc.paragraph(
        f"Shared-data release `{ref['release']}` has manifest SHA-256 "
        f"`{ref['manifest_sha256']}`. The fitted canonical raw snapshot has SHA-256 "
        f"`{dataset_rows[0]['raw_sha256']}`. Inputs begin with raw observations from 2001; "
        f"{raw_assets:,} raw assets are inventoried, and no previous screened/prepared feature "
        "matrix or saved project forecast is used. "
        "Prior/current/recent histories, two-season/career summaries and last observations "
        "are broad deterministic transformations. Training-fold usability filtering is recorded."
    )
    doc.paragraph(
        "Before fitting, a profile diagnostic exposed full franchise names on 365 preseason "
        "and 131 in-season drafted entrants, producing false zero schedule counts and missing "
        "team context. Exact franchise aliases repaired those joins; corrected inputs and "
        "profiles were rebuilt before the campaign. Unknown team schedules now remain missing, "
        "while genuine byes are zero. Partial target labels remain target-specific; recorded "
        "unknown historical stats remain unknown. Current-season 2023+ FTN participation "
        "is masked because its documented release occurs after the postseason."
    )
    doc.paragraph(
        "Canonical source eligibility is about identity, timing and meaning, not earlier "
        "predictive screening. Undated injury reports, retrospective contracts/ranking scrapes, "
        "opaque college stat_N columns, unresolved identities and future market lines are "
        "excluded. Newer FTN/BDB auxiliary fields are inventoried separately; unimplemented "
        "safe joins are coverage gaps, not evidence that those sources lack value. "
        + evidence.link(
            Path(config["input_cache"]) / "auxiliary_inventory.json", "Verified auxiliary inventory"
        )
    )
    checkpoints = sorted(out.glob("*/fits/*/*/complete.json"))
    doc.paragraph(
        f"The completed campaign contains {len(checkpoints):,} verified fold checkpoints, "
        f"{state['completed_fits']:,} target/fold fits and {state.get('failed_folds', 0):,} failed "
        f"folds; {len(final_models['models']):,} final fitted artifacts were refitted using "
        "completed history for next use. Factorized targets fit additional component trees, "
        "so target/fold fits are not the count of underlying estimators. Final refits are "
        "reusable artifacts, not prospective validation. CUDA requests fail rather than "
        "silently training on CPU; one GPU worker and one CPU worker ran concurrently."
    )
    if audit is not None:
        forecast_rows = audit["candidate_forecast_rows"]
        if isinstance(forecast_rows, dict):
            forecast_rows = sum(forecast_rows.values())
        doc.paragraph(
            f"The independent audit counted {audit['target_pipeline_fits']:,} target pipelines, "
            f"{audit['estimator_fits_including_factorized_components']:,} fitted native estimators "
            f"including factorized components, and {forecast_rows:,} candidate "
            "forecast rows. It verifies checkpoint completeness, saved artifact hashes, source "
            "snapshots, forecast chronology and device execution. "
            + evidence.link(out / "audit.json", "Exact native-estimator counts and audit findings")
        )
    doc.paragraph(
        "A separate saved-artifact review verifies policy/member chronology, weights, "
        "serialization and device metadata; independently recomputes dimension comparisons; "
        "and replays selected policy/projection pipelines through restored preprocessing, "
        "column order and exposure handling. These verify saved behavior and do not add "
        "prospective predictive evidence. "
        + evidence.link(out / "independent_review/results.json", "Completed independent review")
        + " · "
        + evidence.link(out / "independent_review/README.md", "Reproduce the review")
        + " · "
        + evidence.link(review_source, "Exact review source")
    )
    methodology_audit = out / "methodology_final_audit.json"
    if methodology_audit.exists():
        if evidence.json(methodology_audit).get("status") != "passed":
            raise ValueError("Methodology integration audit did not pass")
        doc.paragraph(
            evidence.link(methodology_audit, "Independent baseline/profile methodology audit")
            + " checks matched populations, chronological profile fits, exact season-bootstrap "
            "enumeration and invariance of overall errors across profile partitions."
        )
    validation = []
    for name, label in (
        ("application_tests", "Application suite"),
        ("research_tests", "Research suite"),
        ("profile_integration_tests", "Profile integration suite"),
        ("dimension_tests", "Dimension sensitivity suite"),
        ("draft_capital_tests", "Draft-capital suite"),
        ("calibrated_baseline_tests", "Affine-baseline suite"),
    ):
        path = out / f"{name}.xml"
        if path.exists():
            validation.append({"suite": label, **test_summary(evidence.bind(path))})
            doc.paragraph(
                evidence.link(path, f"{label} JUnit results")
                + " · "
                + evidence.link(out / f"{name}.log", "Full validation log")
            )
    if validation:
        doc.table(
            validation,
            [
                ("suite", "Validation"),
                ("passed", "Passed"),
                ("skipped", "Skipped"),
                ("failures", "Failures"),
                ("errors", "Errors"),
            ],
        )
    doc.paragraph(
        "Training source/package hashes, dirty-workspace snapshots, data provenance, feature "
        "order, preprocessing, final models and policy weights are saved. The report's own "
        "`research_manifest.json` hashes its dependencies and exports. Software: "
        + ", ".join(f"{k} {v}" for k, v in signature["packages"].items() if v is not None)
        + "."
    )
    doc.paragraph(
        "Remaining limits: historical provider revisions are not original publication "
        "vintages; undrafted/unlinked newcomers and last-observed team assignments remain "
        "imperfect; market/ECR comparisons are unavailable under this timing contract. "
        "This finite tested search is not exhaustive hyperparameter optimization and can "
        "miss better models. Unsupervised structure and feature "
        "importance do not establish player ability or causation. No production rankings "
        "were replaced, and no 2026 outcomes were graded."
    )
    doc.heading("Evidence and downloadable artifacts")
    links = [
        (analysis / "matched_baseline.parquet", "Matched comparisons and intervals"),
        (analysis / "annual_metrics.parquet", "Annual recipe metrics"),
        (analysis / "policy_selection.parquet", "Chronological selections and weights"),
        (analysis / "predictions.parquet", "Player-level forecasts"),
        (analysis / "distributions.parquet", "Player-level distribution scores"),
        (out / "final_models/manifest.json", "Final model manifest"),
        (out / "signature.json", "Training code and environment signature"),
        (profiles / "optimizer_audit.json", "Profile optimizer audit"),
        (ROOT / "docs/operations/shared-data.md", "GCS restoration and verification"),
        (ROOT / "docs/research/tracking_source_inventory_2026-10-02.md", "Tracking source scope"),
    ]
    for path, label in links:
        doc.paragraph(evidence.link(path, label))
    for path in sorted(exports.glob("*.csv")):
        doc.paragraph(f"[{path.stem.replace('_', ' ')} CSV]({evidence.relative(path)})")
    for path in images:
        if path.suffix in {".svg", ".pdf"}:
            doc.paragraph(
                f"[{path.stem.replace('_', ' ')} {path.suffix[1:].upper()}]({evidence.relative(path)})"
            )
    for path in sorted((analysis / "figures").glob("*.svg")):
        doc.paragraph(evidence.link(path, path.stem.replace("_", " ") + " SVG"))
    (out / "readout.md").write_text("\n".join(doc.markdown) + "\n")
    explorer = clean(headline.filter(pl.col("target").is_in(STATS)).to_dicts())
    encoded = json.dumps(explorer, allow_nan=False).replace("<", "\\u003c")
    page = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>RB GPU research</title>
<style>body{font:16px system-ui,sans-serif;max-width:1280px;margin:32px auto;padding:0 24px;color:#172c40;background:#f7f9fc}
h1{font-size:2.4rem}h2{border-top:1px solid #c8d2dd;padding-top:28px;margin-top:42px}
p{line-height:1.6;max-width:1120px}a{color:#145c9e}code{font-size:.87em;overflow-wrap:anywhere}
.table-wrap{overflow:auto;margin:20px 0}table{border-collapse:collapse;font-size:13px;width:100%;background:white}
td,th{border-bottom:1px solid #d5dfe8;padding:9px;text-align:right;vertical-align:top}
td:first-child,th:first-child{text-align:left}th{background:#eaf0f6}figure{margin:28px 0}
img{max-width:100%;height:auto}figcaption{color:#526476;font-size:14px}select{padding:8px;margin:10px}
.explorer{background:white;border:1px solid #d5dfe8;padding:18px;border-radius:8px}.status{color:#38684a}
@media print{body{max-width:none;font-size:10pt}.table-wrap{overflow:visible}h2{break-before:auto}figure,table{break-inside:avoid}}</style>
<body><nav><a href="readout.md">Markdown readout</a> · <a href="research_manifest.json">Report provenance</a> ·
<a href="profiles/index.html">Profiles</a> · <a href="profiles/conditional/index.html">Conditional errors</a></nav>
<p class="status">Completed historical research · corrected raw inputs · no production promotion</p>
<div class="explorer"><strong>Compare a target and horizon</strong><div id="controls"></div><div id="comparison"></div>
<p>Positive MSE improvement means lower error. Negative interval endpoints favor the policy.
Intervals describe equal-season squared-error differences; percentage gains pool player rows.</p></div>
__BODY__<script>const rows=__DATA__;const fields=['horizon','target'];
for(const field of fields){const select=document.createElement('select');select.id=field;
[...new Set(rows.map(r=>r[field]))].forEach(v=>select.add(new Option(v,v)));select.onchange=render;
document.querySelector('#controls').appendChild(select)}document.querySelector('#target').value='points';
function render(){const subset=rows.filter(r=>fields.every(f=>r[f]===document.getElementById(f).value));
const cols=['recipe','n','rmse','baseline_rmse','mse_improvement_pct','delta_low95','delta_high95'];
const labels={recipe:'Policy',n:'Rows',rmse:'RMSE',baseline_rmse:'Plain-persistence RMSE',
mse_improvement_pct:'MSE improvement vs plain persistence (%)',delta_low95:'Season ΔMSE 95% low',delta_high95:'Season ΔMSE 95% high'};
let t='<table><tr>'+cols.map(c=>'<th>'+labels[c]+'</th>').join('')+'</tr>';
for(const r of subset)t+='<tr>'+cols.map(c=>'<td>'+(r[c]===null?'—':typeof r[c]==='number'?r[c].toFixed(c==='n'?0:2):r[c])+'</td>').join('')+'</tr>';
document.querySelector('#comparison').innerHTML=t+'</table>'}render();</script></body></html>"""
    (out / "index.html").write_text(
        page.replace("__BODY__", "\n".join(doc.html)).replace("__DATA__", encoded)
    )
    source = out / "report/research_report.py"
    source.write_bytes(Path(__file__).read_bytes())
    outputs = [out / "readout.md", out / "index.html", source, *exports.glob("*.csv"), *images]
    # Validate every local evidence/export/image link in the generated HTML.
    links = re.findall(r'(?:href|src)="([^"]+)"', (out / "index.html").read_text())
    for link in links:
        if link == "research_manifest.json" or "://" in link or link.startswith("#"):
            continue
        if not (out / html.unescape(link)).exists():
            raise ValueError(f"Broken report link: {link}")
    json_write(
        out / "research_manifest.json",
        {
            "status": "complete",
            "scope": "Integrated report; no fitting or model selection performed",
            "generator_sha256": digest(Path(__file__)),
            "inputs": evidence.inputs,
            "files": {evidence.relative(path): digest(path) for path in outputs},
        },
    )
    return out / "index.html"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="rb_gpu_20261003_001")
    args = parser.parse_args()
    print(build_report(RUNS / args.run_id))
