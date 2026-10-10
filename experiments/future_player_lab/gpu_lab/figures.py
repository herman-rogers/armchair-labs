"""Standalone scientific figures for the completed RB campaign."""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from .run import RUNS, STATS


def figures(out):
    config = json.loads((out / "config.json").read_text())
    horizons = config["horizons"]
    years = config["evaluation_years"]
    destination = out / "analysis/figures"
    destination.mkdir(exist_ok=True)
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130}
    )
    skill = pl.read_parquet(out / "analysis/matched_baseline.parquet")

    def panels(height):
        fig, axes = plt.subplots(
            1, len(horizons), figsize=(7 * len(horizons), height), sharey=True, squeeze=False
        )
        return fig, axes[0]

    fig, axes = panels(7)
    targets = [target for target in STATS + ["points_from_stats"] if target in skill["target"]]
    for ax, horizon in zip(axes, horizons, strict=True):
        for i, (model, color) in enumerate(
            [
                ("selected", "#145c9e"),
                ("ensemble_top3", "#009e73"),
                ("ensemble_diverse", "#d55e00"),
                ("ensemble_convex", "#9467bd"),
            ]
        ):
            subset = skill.filter((pl.col("horizon") == horizon) & (pl.col("recipe") == model))
            values = dict(zip(subset["target"], subset["mse_improvement_pct"], strict=True))
            ax.barh(
                np.arange(len(targets)) + (i - 1.5) * 0.18,
                np.asarray([values.get(t, np.nan) for t in targets], dtype=float),
                height=0.18,
                label=model,
                color=color,
            )
        ax.set_yticks(range(len(targets)), [t.replace("_", " ") for t in targets])
        ax.axvline(0, color="#666", linewidth=0.8)
        ax.set(title=horizon, xlabel="MSE improvement over matched persistence (%)")
        ax.legend(loc="lower right")
    axes[0].invert_yaxis()
    fig.suptitle(
        f"RB forecasts: selection uses only earlier seasons (evaluation {min(years)}–{max(years)})"
    )
    fig.tight_layout()
    fig.savefig(destination / "predictive_skill.png", dpi=180)
    fig.savefig(destination / "predictive_skill.svg")
    plt.close(fig)

    comparisons = pl.read_parquet(out / "analysis/controlled_comparisons.parquet").filter(
        pl.col("controlled_representation")
    )
    families = sorted(comparisons["family"].unique().to_list())
    reps = sorted(comparisons["representation"].unique().to_list())
    fig, axes = panels(5)
    point_values = comparisons.filter(pl.col("target") == "points")[
        "mse_improvement_pct"
    ].to_numpy()
    finite = point_values[np.isfinite(point_values)]
    limit = max(10, float(np.max(np.abs(finite)))) if len(finite) else 10
    for ax, horizon in zip(axes, horizons, strict=True):
        if not families or not reps:
            ax.text(
                0.5,
                0.5,
                "No controlled representation comparisons in this run",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(horizon)
            ax.set_axis_off()
            continue
        subset = comparisons.filter((pl.col("horizon") == horizon) & (pl.col("target") == "points"))
        lookup = {
            (r["family"], r["representation"]): r["mse_improvement_pct"]
            for r in subset.iter_rows(named=True)
        }
        array = np.array(
            [[lookup.get((f, r), np.nan) for r in reps] for f in families], dtype=float
        )
        ax.imshow(array, cmap="RdYlGn", vmin=-limit, vmax=limit, aspect="auto")
        for i in range(len(families)):
            for j in range(len(reps)):
                label = f"{array[i, j]:.1f}" if np.isfinite(array[i, j]) else "—"
                ax.text(j, i, label, ha="center", va="center")
        ax.set_xticks(range(len(reps)), reps, rotation=35, ha="right")
        ax.set_yticks(range(len(families)), families)
        ax.set_title(horizon)
    fig.suptitle(
        "Direct PPR: MSE improvement (%) from representation vs same engine's raw inputs\n"
        "Matched rows and fixed model settings; descriptive development comparison"
    )
    fig.tight_layout()
    fig.savefig(destination / "representations.png", dpi=180)
    fig.savefig(destination / "representations.svg")
    plt.close(fig)

    calibration = pl.read_parquet(out / "analysis/distribution_summary.parquet")
    fig, axes = panels(6)
    stats = [target for target in STATS if target in config["targets"]]
    for ax, horizon in zip(axes, horizons, strict=True):
        subset = calibration.filter(
            (pl.col("horizon") == horizon) & (pl.col("recipe") == "ensemble_top3")
        )
        lookup = {r["target"]: r["coverage80"] for r in subset.iter_rows(named=True)}
        ax.barh(
            range(len(stats)),
            np.asarray([lookup.get(s, np.nan) for s in stats], dtype=float),
            color="#145c9e",
        )
        ax.axvline(0.8, color="#d55e00", linestyle="--", label="Nominal 80%")
        ax.set_yticks(range(len(stats)), [s.replace("_", " ") for s in stats])
        ax.set(xlim=(0, 1), xlabel="Empirical 80% interval coverage", title=horizon)
        ax.legend()
    axes[0].invert_yaxis()
    fig.suptitle("Earlier-fold residual calibration, measured on later seasons")
    fig.tight_layout()
    fig.savefig(destination / "calibration.png", dpi=180)
    fig.savefig(destination / "calibration.svg")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    figures(RUNS / args.run_id)
