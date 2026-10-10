"""Score a registered saved model from raw cutoff-known inputs; never fit estimators."""

from __future__ import annotations

import argparse
import re
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
from threadpoolctl import threadpool_limits

from ..gpu_lab.data import source_family
from ..gpu_lab.refit import predict_policy
from ..gpu_lab.run import ROOT, RUNS, digest, json_write, packages
from .data import add_supplements, feature_matrix, feature_rows, history_inputs, snapshot_inputs
from .registry import REGISTRY, load_registered_policy, load_spec, model_contract
from .report import write_report

POLICIES = ("ensemble_top3", "selected", "ensemble_diverse")


def feature_replay(history, manifest, position):
    """Compare the outcome-free builder to the original sealed historical matrix."""
    inputs = manifest["input_provenance"]["remaining"]
    cache = Path(inputs["cache_directory"])
    if not cache.exists():
        cache = RUNS / "gpu_raw_inputs_001" / cache.name
    path = cache / "panel.parquet"
    if digest(path) != inputs["files"]["panel.parquet"]:
        raise ValueError("Original historical input checkpoint changed")
    names = inputs["names"]
    years = (2021, 2025)
    rows = 0
    original = pl.read_parquet(path)
    for year in years:
        expected = original.filter(pl.col("year") == year).sort(
            "year", "origin", "team", "player_id"
        )
        actual = feature_rows(history, position, year, 2)
        if not actual["player_id"].equals(expected["player_id"]):
            raise ValueError("Historical candidate population does not replay")
        np.testing.assert_allclose(
            feature_matrix(actual, names),
            feature_matrix(expected, names),
            rtol=1e-5,
            atol=1e-5,
            equal_nan=True,
        )
        rows += actual.height
    return {
        "status": "passed",
        "years": list(years),
        "rows": rows,
        "features": len(names),
        "description": f"{len(names):,} features match 2021 and 2025 saved inputs; {rows} rows.",
    }


def rank_predictions(panel, predictions, primary):
    result = panel.select(
        "player_id",
        "player_display_name",
        "position",
        "team",
        "year",
        "origin",
        "weeks",
        "scheduled_games",
        "history_games",
        "candidate_basis",
    )
    for policy, values in predictions.items():
        if np.asarray(values).shape != (panel.height,) or not np.isfinite(values).all():
            raise ValueError("Invalid prediction shape or nonfinite forecast")
        result = result.with_columns(pl.Series(f"ppr_{policy}", values))
    result = result.with_columns(
        pl.col(f"ppr_{primary}").alias("predicted_ppr"),
        (pl.col("history_games") > 0).alias("observed_this_season"),
        pl.lit("week_cutoff_reconstruction").alias("forecast_basis"),
    ).with_columns(
        pl.when(pl.col("scheduled_games") > 0)
        .then(pl.col("predicted_ppr") / pl.col("scheduled_games"))
        .otherwise(None)
        .alias("ppr_per_scheduled_game")
    )
    return result.sort(["predicted_ppr", "player_id"], descending=[True, False]).with_row_index(
        "rank", offset=1
    )


def forecast(model_id, snapshot, supplements, season, origin, run_id, history_cache, threads=4):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", run_id):
        raise ValueError("Unsafe run identifier")
    spec = load_spec(model_id)
    manifest, names = model_contract(spec, season, origin)
    if spec["horizon"] != "remaining" or spec["target"] != "points":
        raise ValueError("This command currently implements remaining-season PPR only")
    output = RUNS / run_id
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    issued_at = datetime.now(UTC).isoformat()
    json_write(output / "status.json", {"status": "running", "issued_at": issued_at})
    history = history_inputs(Path(history_cache))
    replay = feature_replay(history, manifest, spec["position"])
    print(replay["description"], flush=True)
    tables, source = snapshot_inputs(history, Path(snapshot), season, origin)
    tables, supplement_audit = add_supplements(tables, supplements, season, origin)
    panel = feature_rows(tables, spec["position"], season, origin)
    x = feature_matrix(panel, names)
    panel.write_parquet(output / "features.parquet")
    json_write(output / "feature_names.json", names)
    coverage = []
    for i, name in enumerate(names):
        coverage.append(
            {
                "feature": name,
                "source": source_family(name),
                "known_rows": int(np.isfinite(x[:, i]).sum()),
                "total_rows": panel.height,
            }
        )
    pl.DataFrame(coverage).write_csv(output / "coverage.csv")
    predictions, policy_records = {}, {}
    scoring_start = time.monotonic()
    with threadpool_limits(limits=threads):
        for policy in POLICIES:
            bundle = load_registered_policy(spec, policy)
            t = time.monotonic()
            predictions[policy] = predict_policy(bundle, x, names, panel["weeks"].to_numpy())
            # Reordering inputs must not change forecasts: the serving contract is by name.
            reordered = predict_policy(bundle, x[:, ::-1], names[::-1], panel["weeks"].to_numpy())
            np.testing.assert_allclose(predictions[policy], reordered, rtol=1e-6, atol=1e-5)
            policy_records[policy] = {
                "prediction_and_alignment_check_seconds": time.monotonic() - t,
                "members": len(bundle["components"]),
            }
            print(f"Scored {panel.height} {spec['position']}s with {policy}", flush=True)
    rankings = rank_predictions(panel, predictions, spec["default_policy"])
    rankings.write_csv(output / "rankings.csv")
    rankings.write_parquet(output / "rankings.parquet")
    json_write(output / "rankings.json", rankings.to_dicts())
    summary = {
        "model_id": model_id,
        "position": spec["position"],
        "season": season,
        "origin": origin,
        "issued_at": issued_at,
        "forecast_basis": "week_cutoff_reconstruction",
        "training_max_year": manifest["training_cutoff_year"],
        "scoring": spec["scoring"],
        "model_manifest_sha256": spec["model_manifest_sha256"],
        "raw_snapshot": source,
        "supplements": supplement_audit,
        "feature_replay": replay,
        "rows": panel.height,
        "features": len(names),
        "primary_policy": spec["default_policy"],
        "policies": policy_records,
        "scoring_including_load_seconds": time.monotonic() - scoring_start,
        "total_seconds": time.monotonic() - started,
        "limits": [
            "Later captures reconstruct the cutoff; not an originally issued prospective forecast.",
            "No 2026 outcomes used for fitting or model selection; no models refitted.",
            "Unknown source fields remain missing, not zero. See source audit and coverage.csv.",
            "Candidates and teams follow past observations; no current active-roster filter.",
            "No post-cutoff injury/transaction adjustments; policy differences are not intervals.",
        ],
    }
    json_write(output / "summary.json", summary)
    write_report(output, rankings, summary)
    source_dir = output / "implementation"
    source_dir.mkdir()
    sources = {}
    implementation = [*Path(__file__).parent.glob("*.py")]
    implementation.extend(
        [
            ROOT / "experiments/future_player_lab/notebooks/raw_distribution_data.py",
            ROOT / "experiments/future_player_lab/gpu_backends.py",
            *[
                ROOT / f"experiments/future_player_lab/gpu_lab/{name}.py"
                for name in ("data", "refit", "representations", "run", "factorized")
            ],
        ]
    )
    for path in sorted(implementation):
        dest = source_dir / path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        sources[str(path.relative_to(ROOT))] = digest(path)
    shutil.copy2(REGISTRY / f"{model_id}.json", output / "model.json")
    json_write(output / "status.json", {"status": "complete", "issued_at": issued_at})
    json_write(
        output / "manifest.json",
        {
            "schema_version": 1,
            "status": "complete",
            "model_id": model_id,
            "model_manifest_sha256": spec["model_manifest_sha256"],
            "packages": packages(),
            "sources": sources,
            "historical_cache_sha256": digest(Path(history_cache) / "manifest.json"),
            "files": {
                str(p.relative_to(output)): digest(p)
                for p in sorted(output.rglob("*"))
                if p.is_file()
            },
        },
    )
    print(f"Report: {output / 'index.html'}", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="rb_ppr_v1")
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--supplements", type=Path, required=True)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--origin", type=int, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--history-cache", type=Path, default=RUNS / "forecast_history_20261009_001"
    )
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    forecast(
        args.model,
        args.snapshot,
        args.supplements,
        args.season,
        args.origin,
        args.run_id,
        args.history_cache,
        args.threads,
    )


if __name__ == "__main__":
    main()
