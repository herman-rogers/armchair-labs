"""Compare v2 to frozen v1 on the same eligible rows, separating eligibility effects."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from ...gpu_lab.run import digest, json_write
from ..audit import audit_cell
from ..run import metrics
from .data import ROOT, SOURCE_DIR, V1_RUN, load_sources


def audit_states(audit, events, schedule):
    """Independently reconstruct latest source states with relational operations."""
    for (year, origin), block in audit.partition_by(["year", "origin"], as_dict=True).items():
        cutoffs = block["availability_as_of"].unique().to_list()
        if len(cutoffs) != 1:
            raise ValueError("One season/origin has multiple availability cutoffs")
        cutoff = datetime.fromisoformat(cutoffs[0])
        season = schedule.filter(pl.col("season") == year)
        days = (season.filter(pl.col("week") <= origin) if origin else season)["gameday"].cast(
            pl.Date
        )
        day = days.max() + timedelta(days=1) if origin else days.min()
        expected_cutoff = datetime(day.year, day.month, day.day, 12, tzinfo=UTC)
        if cutoff != expected_cutoff:
            raise ValueError("Availability cutoff differs from forecast calendar")
        known = events.filter(
            (pl.col("available_at") <= cutoff)
            & (pl.col("available_at") >= datetime(year - 1, 1, 1, tzinfo=UTC))
            & pl.col("player_id").is_in(block["player_id"].implode())
        )
        clocks = known.group_by("player_id").agg(pl.col("available_at").max())
        latest = known.join(clocks, on=["player_id", "available_at"], how="inner")
        states = (
            latest.group_by("player_id")
            .agg(
                pl.col("status").n_unique().alias("count"), pl.col("status").first().alias("state")
            )
            .with_columns(
                pl.when(pl.col("count") == 1)
                .then(pl.col("state"))
                .otherwise(pl.lit("UNKNOWN"))
                .alias("expected")
            )
        )
        checked = block.join(states, on="player_id", how="left").with_columns(
            pl.col("expected").fill_null("UNKNOWN")
        )
        if checked.filter(pl.col("availability_status") != pl.col("expected")).height:
            raise ValueError(f"Latest source state differs at {year}/{origin}")
    return dict(rows=audit.height, latest_states_verified=True)


def selected_points(frame, k):
    return float(
        frame.sort(["prediction", "player_id"], descending=[True, False]).head(k)["actual"].sum()
    )


def compare(run, parent=V1_RUN, source_dir=SOURCE_DIR):
    run, parent = Path(run), Path(parent)
    sources, _ = load_sources(source_dir)
    source_sha = digest(Path(source_dir) / "manifest.json")
    cells = []
    for path in sorted(run.glob("*/manifest.json")):
        model_id = path.parent.name
        if model_id == "inputs":
            continue
        saved = json.loads(path.read_text())
        if "model_id" not in saved:
            continue
        audit = audit_cell(path.parent)
        parent_audit = audit_cell(parent / model_id)
        new = pl.read_parquet(path.parent / "evaluation_predictions.parquet")
        old_path = parent / model_id / "evaluation_predictions.parquet"
        old = pl.read_parquet(old_path)
        keys = ["year", "player_id", "target", "recipe"]
        pairs = new.join(
            old.select(
                *keys,
                pl.col("actual").alias("v1_actual"),
                pl.col("prediction").alias("v1_prediction"),
            ),
            on=keys,
            how="left",
            validate="1:1",
        )
        if pairs["v1_prediction"].null_count():
            raise ValueError("V2 comparison contains a player absent from v1")
        np.testing.assert_allclose(
            pairs["actual"].to_numpy(), pairs["v1_actual"].to_numpy(), equal_nan=True
        )
        comparisons = []
        for (target, recipe), group in pairs.partition_by(
            ["target", "recipe"], as_dict=True
        ).items():
            for year, part in [
                (None, group),
                *[(k[0], v) for k, v in group.partition_by("year", as_dict=True).items()],
            ]:
                a = part["actual"].to_numpy()
                v1, v2 = (
                    metrics(a, part["v1_prediction"].to_numpy()),
                    metrics(a, part["prediction"].to_numpy()),
                )
                comparisons.append(
                    dict(
                        target=target,
                        recipe=recipe,
                        year=year,
                        v1=v1,
                        v2=v2,
                        mse_reduction_pct=100 * (1 - v2["mse"] / v1["mse"]) if v1["mse"] else None,
                    )
                )
        candidate_audit = pl.read_parquet(run / "inputs" / model_id / "candidate_audit.parquet")
        dataset = json.loads((run / "inputs" / model_id / "dataset.json").read_text())
        if dataset["contract"]["availability_manifest_sha256"] != source_sha:
            raise ValueError("Availability audit source differs from training")
        state_audit = audit_states(candidate_audit, sources["events"], sources["schedule"])
        base_path = parent / "inputs" / model_id / "panel.parquet"
        base = pl.read_parquet(base_path)
        panel = pl.read_parquet(run / "inputs" / model_id / "panel.parquet")
        keep = candidate_audit.filter(pl.col("availability_eligible")).select("year", "player_id")
        expected = base.join(keep, on=["year", "player_id"], how="semi").sort("year", "player_id")
        if not panel.select(base.columns).sort("year", "player_id").equals(expected):
            raise ValueError("Availability changed original v1 features or labels")
        if candidate_audit.filter(
            (~pl.col("availability_eligible")) != (pl.col("availability_status") == "RET")
        ).height:
            raise ValueError("Eligibility rule differs from stated retirement policy")
        if candidate_audit.filter(
            pl.col("availability_observed_at") > pl.col("availability_as_of")
        ).height:
            raise ValueError("Availability source comes after forecast cutoff")
        ranking = []
        k = 12 if saved["position"] == "QB" else 24
        for year in range(2021, 2026):
            for recipe in ("ensemble", "affine", *saved["recipes"], "selected"):
                o = old.filter(
                    (pl.col("year") == year)
                    & (pl.col("target") == "points")
                    & (pl.col("recipe") == recipe)
                )
                n = new.filter(
                    (pl.col("year") == year)
                    & (pl.col("target") == "points")
                    & (pl.col("recipe") == recipe)
                )
                gated = o.join(n.select("player_id"), on="player_id", how="semi")
                ranking.append(
                    dict(
                        year=year,
                        recipe=recipe,
                        k=k,
                        v1_original=selected_points(o, k),
                        v1_same_eligible=selected_points(gated, k),
                        v2=selected_points(n, k),
                    )
                )
        # One historical usage threshold shared across versions; no label-defined cohorts.
        usage = "history_attempts" if saved["position"] == "QB" else "history_targets"
        usage_panel = base.select(
            "year",
            "player_id",
            ((pl.col(usage) + pl.col("history_carries")) / pl.col("history_weeks")).alias("usage"),
        )
        high_parts = []
        for year in range(2021, 2026):
            past = usage_panel.filter((pl.col("year") < year) & (pl.col("usage") > 0))
            threshold = float(np.quantile(past["usage"].drop_nulls().to_numpy(), 0.75))
            high_parts.append(
                pairs.filter(
                    (pl.col("year") == year)
                    & (pl.col("target") == "points")
                    & (pl.col("recipe") == "ensemble")
                ).join(
                    usage_panel.filter(
                        (pl.col("year") == year) & (pl.col("usage") >= threshold)
                    ).select("year", "player_id"),
                    on=["year", "player_id"],
                    how="inner",
                )
            )
        high = pl.concat(high_parts)
        cells.append(
            dict(
                model_id=model_id,
                audit=audit,
                parent_audit=parent_audit,
                availability_audit=state_audit,
                comparisons=comparisons,
                ranking=ranking,
                high_usage=dict(
                    v1=metrics(high["actual"].to_numpy(), high["v1_prediction"].to_numpy()),
                    v2=metrics(high["actual"].to_numpy(), high["prediction"].to_numpy()),
                ),
                excluded=candidate_audit.filter(~pl.col("availability_eligible")).to_dicts(),
                coverage=candidate_audit.group_by("year", "availability_status")
                .len()
                .sort("year", "availability_status")
                .to_dicts(),
                manifest_sha256=digest(path),
                v1_predictions_sha256=digest(old_path),
                base_matrix_preserved=True,
            )
        )
    if len(cells) != 6:
        raise ValueError("All six completed v2 cells are required")
    return dict(
        schema_version=1,
        campaign_id="opening_drive_v2",
        cells=cells,
        comparison=(
            "Paired finite outcomes on v2 eligible rows. V1 gate-only ranking isolates "
            "population effect; retraining plus availability features are a combined treatment."
        ),
        limitation=(
            "Historical development comparison prompted by v1 failures; "
            "no pristine holdout or causal claim for individual features."
        ),
    )


def markdown(report):
    lines = [
        "# Opening Drive v2: dated availability",
        "",
        report["comparison"],
        "",
        report["limitation"],
        "",
        "| Model | Paired rows | V1 RMSE | V2 RMSE | MSE reduction | High-usage MSE reduction |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for cell in report["cells"]:
        row = next(
            r
            for r in cell["comparisons"]
            if r["target"] == "points" and r["recipe"] == "ensemble" and r["year"] is None
        )
        high = cell["high_usage"]
        lines.append(
            f"| {cell['model_id']} | {row['v2']['n']} | {row['v1']['rmse']:.2f} | "
            f"{row['v2']['rmse']:.2f} | {row['mse_reduction_pct']:.2f}% | "
            f"{100 * (1 - high['v2']['mse'] / high['v1']['mse']):.2f}% |"
        )
    lines += [
        "",
        "## Selection diagnostics",
        "",
        "Same slots per position; entries are realized points of the forecast-selected players.",
        "",
        "| Model | Year | V1 original | V1 with eligibility gate | V2 | V2 affine |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for cell in report["cells"]:
        for row in cell["ranking"]:
            if row["recipe"] != "ensemble":
                continue
            affine = next(
                x for x in cell["ranking"] if x["year"] == row["year"] and x["recipe"] == "affine"
            )
            lines.append(
                f"| {cell['model_id']} | {row['year']} | {row['v1_original']:.2f} | "
                f"{row['v1_same_eligible']:.2f} | {row['v2']:.2f} | {affine['v2']:.2f} |"
            )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", type=Path, default=ROOT / "experiments/future_player_lab/runs/opening_drive_v2_001"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sources", type=Path, default=SOURCE_DIR)
    args = parser.parse_args()
    report = compare(args.run, source_dir=args.sources)
    args.output.mkdir(parents=True, exist_ok=False)
    json_write(args.output / "report.json", report)
    (args.output / "report.md").write_text(markdown(report))
    print(markdown(report))
