"""Package original RB research for Scout Team; inference only, never refitting."""

from __future__ import annotations

# ruff: noqa: E501
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

from ..gpu_lab.run import digest, json_write

TARGETS = [
    "carries",
    "rushing_yards",
    "rushing_tds",
    "targets",
    "receptions",
    "receiving_yards",
    "receiving_tds",
    "points",
]
POLICY = "ensemble_top3"
STUDY = "rb_exploratory_research"

# Run archived source in an isolated interpreter, without the current repository
# or the original research directories on its import path.
INFERENCE = r"""
import importlib.metadata,json,sys
from pathlib import Path
base=Path(sys.argv[1]).resolve()
sys.path.insert(0,str(base/'implementation'))
import numpy as np
import polars as pl
from threadpoolctl import threadpool_limits
from experiments.future_player_lab.gpu_lab import refit
assert Path(refit.__file__).resolve().is_relative_to(base/'implementation')
m=json.loads((base/'final_models/manifest.json').read_text())
for key in ['numpy','polars','scipy','scikit-learn','xgboost','catboost','lightgbm']:
    if importlib.metadata.version(key)!=m['packages'][key]:
        raise ValueError('Original RB package version differs: '+key)
f=pl.read_parquet(base/'forecast/features.parquet')
names=json.loads((base/'forecast/feature_names.json').read_text())
x=f.select(names).to_numpy().astype(np.float32)
x[~np.isfinite(x)]=np.nan
pred={}
with threadpool_limits(limits=4):
    for target in ['carries','rushing_yards','rushing_tds','targets','receptions','receiving_yards','receiving_tds','points']:
        policy=refit.load_policy(base/'final_models','remaining',target,'ensemble_top3')
        pred[target]=refit.predict_policy(policy,x,names,f['weeks'].to_numpy()).tolist()
Path(sys.argv[2]).write_text(json.dumps({'player_ids':f['player_id'].to_list(),'predictions':pred},allow_nan=False))
"""


def verify_files(directory, manifest):
    for name, sha in manifest["files"].items():
        if digest(Path(directory) / name) != sha:
            raise ValueError(f"Original RB artifact changed: {directory}/{name}")


def normalized_rows(rankings, predicted):
    """Retain old points/ranks exactly; align newly materialized stats by exact ID."""
    original = {row["player_id"]: row for row in rankings.to_dicts()}
    ids = predicted["player_ids"]
    if len(original) != rankings.height or len(set(ids)) != len(ids) or set(original) != set(ids):
        raise ValueError("Original RB forecast and feature populations differ")
    if set(predicted["predictions"]) != set(TARGETS):
        raise ValueError("Original RB target contract differs")
    for values in predicted["predictions"].values():
        if len(values) != len(ids) or not np.isfinite(values).all():
            raise ValueError("Invalid original RB predictions")
    expected = np.array([original[player_id]["ppr_ensemble_top3"] for player_id in ids])
    actual = np.array(predicted["predictions"]["points"])
    np.testing.assert_allclose(actual, expected, atol=1e-8, rtol=1e-10)
    rows = []
    for index, player_id in enumerate(ids):
        prior = original[player_id]
        if (prior["position"], prior["year"], prior["origin"]) != ("RB", 2026, 2):
            raise ValueError("Original RB forecast cutoff differs")
        values = {target: float(predicted["predictions"][target][index]) for target in TARGETS}
        values["points"] = prior["ppr_ensemble_top3"]
        component = (
            0.1 * (values["rushing_yards"] + values["receiving_yards"])
            + values["receptions"]
            + 6 * (values["rushing_tds"] + values["receiving_tds"])
        )
        rows.append(
            {
                "player_id": player_id,
                "player_name": prior["player_display_name"],
                "team": prior["team"],
                "position": "RB",
                "season": 2026,
                "origin_week": 2,
                "horizon": "remaining",
                "rank": prior["rank"],
                "predictions": values,
                "scheduled_games": prior["scheduled_games"],
                "calendar_weeks": prior["weeks"],
                "points_per_scheduled_game": prior["ppr_per_scheduled_game"],
                "component_points": component,
                "candidate_basis": prior["candidate_basis"],
            }
        )
    return sorted(rows, key=lambda row: row["rank"]), float(np.max(abs(actual - expected)))


def make_archive(path, members):
    """Preserve source/member bytes and paths; never deserialize or rewrite weights."""
    records = {}
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, source in sorted(members.items()):
            records[name] = digest(source)
            archive.writestr(name, Path(source).read_bytes())
        archive.writestr("archive_receipt.json", json.dumps({"files": records}, indent=2))
    return records


def extract_archive(path, destination, expected_sha256):
    if digest(path) != expected_sha256:
        raise ValueError("Packaged RB archive checksum mismatch")
    destination = Path(destination).resolve()
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate archived RB member")
        receipt = json.loads(archive.read("archive_receipt.json"))
        if set(names) != {*receipt["files"], "archive_receipt.json"}:
            raise ValueError("RB archive members differ from receipt")
        for name, sha in receipt["files"].items():
            target = (destination / name).resolve()
            if (
                Path(name).is_absolute()
                or ".." in Path(name).parts
                or not target.is_relative_to(destination)
            ):
                raise ValueError("Unsafe RB archive member")
            content = archive.read(name)
            import hashlib

            if hashlib.sha256(content).hexdigest() != sha:
                raise ValueError("Archived RB member checksum mismatch")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)


def predict_archive(archive, expected_sha256):
    with tempfile.TemporaryDirectory(prefix="scout-rb-replay-") as temporary:
        base = Path(temporary)
        extract_archive(archive, base, expected_sha256)
        result = base / "predictions.json"
        subprocess.run(
            [sys.executable, "-I", "-c", INFERENCE, str(base), str(result)], check=True, cwd=base
        )
        return json.loads(result.read_text())


def _prepare(data_root, staging):
    root = Path(data_root).resolve().parent
    runs = root / "experiments/future_player_lab/runs"
    study = runs / "rb_gpu_20261003_001"
    final = study / "final_models"
    preview = runs / "rb_ppr_2026_w2_preview_002"
    research = json.loads((study / "research_manifest.json").read_text())
    for name in (
        "manifest.json",
        "config.json",
        "signature.json",
        "analysis/manifest.json",
        "analysis/predictions.parquet",
        "final_models/manifest.json",
    ):
        if digest(study / name) != research["inputs"][name]:
            raise ValueError(f"Original RB research input differs: {name}")
    final_manifest = json.loads((final / "manifest.json").read_text())
    analysis_manifest = json.loads((study / "analysis/manifest.json").read_text())
    preview_manifest = json.loads((preview / "manifest.json").read_text())
    for directory, receipt in (
        (final, final_manifest),
        (study / "analysis", analysis_manifest),
        (preview, preview_manifest),
    ):
        if receipt.get("status") != "complete":
            raise ValueError("Original RB source is incomplete")
        verify_files(directory, receipt)
    if preview_manifest["model_manifest_sha256"] != digest(final / "manifest.json"):
        raise ValueError("Original RB forecast model binding mismatch")
    summary = json.loads((preview / "summary.json").read_text())
    if (summary["position"], summary["season"], summary["origin"], summary["primary_policy"]) != (
        "RB",
        2026,
        2,
        POLICY,
    ):
        raise ValueError("Original RB primary forecast differs")
    signature = json.loads((study / "signature.json").read_text())
    members = {
        f"final_models/{name}": final / name for name in ["manifest.json", *final_manifest["files"]]
    }
    for name, sha in signature["sources"].items():
        source = study / "implementation" / name
        if digest(source) != sha:
            raise ValueError(f"Archived RB implementation changed: {name}")
        members[f"implementation/{name}"] = source
    for name, sha in final_manifest["sources"].items():
        source = final / Path(name).name
        if digest(source) != sha:
            raise ValueError(f"Archived RB refit source changed: {name}")
        members[f"implementation/{name}"] = source
    members.update(
        {
            f"forecast/{name}": preview / name
            for name in ["manifest.json", *preview_manifest["files"]]
        }
    )
    archive = staging / "original_rb_library.zip"
    archive_members = make_archive(archive, members)
    predictions = predict_archive(archive, digest(archive))
    rows, point_difference = normalized_rows(
        pl.read_parquet(preview / "rankings.parquet"), predictions
    )
    policies = json.loads((final / "policies.json").read_text())
    primary = [
        policy for policy in policies if policy["policy"] == POLICY and policy["target"] in TARGETS
    ]
    if len(primary) != 16 or any(len(p["members"]) != 3 for p in primary):
        raise ValueError("Original RB primary ensemble contract differs")
    json_write(staging / "primary_policies.json", primary)
    provenance = {
        "schema_version": 1,
        "campaign_id": "scout_team_v1",
        "campaign_name": "Scout Team v1",
        "source_study_id": STUDY,
        "source_study_name": "RB Exploratory Research",
        "source_run": "rb_gpu_20261003_001",
        "source_forecast_run": "rb_ppr_2026_w2_preview_002",
        "source_model_manifest_sha256": digest(final / "manifest.json"),
        "source_forecast_manifest_sha256": digest(preview / "manifest.json"),
        "source_research_manifest_sha256": digest(study / "research_manifest.json"),
        "primary_policy": POLICY,
        "training_through_season": 2025,
        "new_fits": 0,
        "archive_members": archive_members,
        "point_replay_max_absolute_difference": point_difference,
        "materialization": {
            "points": "Original saved numerical values preserved exactly",
            "other_targets": "New inference using original saved weights and original saved Week2 matrix",
        },
        "replay_contract": "rb_gpu_saved_policy_v1",
        "required_packages": final_manifest["packages"],
    }
    json_write(staging / "source_provenance.json", provenance)
    forecasts = {
        "schema_version": 1,
        "campaign_id": "scout_team_v1",
        "model_id": "rb_remaining",
        "position": "RB",
        "season": 2026,
        "origin_week": 2,
        "horizon": "remaining",
        "training_through_season": 2025,
        "scoring": summary["scoring"],
        "generated_at": datetime.now(UTC).isoformat(),
        "forecast_kind": "saved_points_plus_original_saved_model_inference",
        "source_study_id": STUDY,
        "source_forecast_issued_at": summary["issued_at"],
        "source_model_manifest_sha256": digest(final / "manifest.json"),
        "source_forecast_manifest_sha256": digest(preview / "manifest.json"),
        "source_feature_sha256": preview_manifest["files"]["features.parquet"],
        "materialization": provenance["materialization"],
        "point_replay_max_absolute_difference": point_difference,
        "snapshot": summary["raw_snapshot"],
        "supplements": summary["supplements"],
        "notes": [
            *summary["limits"],
            "Scout Team labels a preserved research family; no models were retrained.",
            "Direct PPR and the arithmetic sum of independent stat forecasts can differ.",
        ],
        "rows": rows,
    }
    json_write(staging / "rb_remaining_2026_w2.json", forecasts)
    (staging / "adapter_source.py").write_bytes(Path(__file__).read_bytes())
    all_predictions = pl.read_parquet(study / "analysis/predictions.parquet").filter(
        (pl.col("recipe") == POLICY)
        & pl.col("year").is_between(2021, 2025)
        & pl.col("target").is_in(TARGETS)
    )
    models = []
    for horizon in ("season", "remaining"):
        model_id = f"rb_{horizon}"
        selected = all_predictions.filter(pl.col("horizon") == horizon)
        selected.write_parquet(staging / f"{model_id}_evaluation.parquet")
        metrics = []
        for (target,), block in selected.partition_by("target", as_dict=True).items():
            for year, frame in [
                (None, block),
                *[
                    (int(key[0]), part)
                    for key, part in block.partition_by("year", as_dict=True).items()
                ],
            ]:
                y, p = frame["actual"].to_numpy(), frame["prediction"].to_numpy()
                mask = np.isfinite(y) & np.isfinite(p)
                err = y[mask] - p[mask]
                metrics.append(
                    {
                        "target": target,
                        "recipe": POLICY,
                        "year": year,
                        "n": int(mask.sum()),
                        "mse": float(np.mean(err**2)),
                        "rmse": float(np.sqrt(np.mean(err**2))),
                        "mae": float(np.mean(abs(err))),
                    }
                )
        json_write(
            staging / f"{model_id}_metrics.json",
            {
                "schema_version": 1,
                "campaign_id": "scout_team_v1",
                "source_study_id": STUDY,
                "source_run": "rb_gpu_20261003_001",
                "position": "RB",
                "horizon": horizon,
                "policy": POLICY,
                "evaluation_years": list(range(2021, 2026)),
                "rows": metrics,
                "evaluation_kind": "chronological_historical_development",
                "notes": [
                    "Metrics use original held-out predictions; no new fitting or policy selection.",
                    "52 exploratory recipes; prior-three-year selection and equal-weight top3 ensembles.",
                ],
            },
        )
        models.append(
            {
                "model_id": model_id,
                "position": "RB",
                "horizon": horizon,
                "origin_week": 0 if horizon == "season" else 2,
                "targets": TARGETS,
                "training_through_season": 2025,
                "scoring": summary["scoring"],
                "primary_policy": POLICY,
                "source_study_id": STUDY,
                "source_run": "rb_gpu_20261003_001",
                "feature_count": 4810,
                "model_generation": "original_exploratory_fitted_weights",
                "saved_model_member_count": 24,
                "capabilities": {
                    "saved_forecasts": horizon == "remaining",
                    "historical_predictions": True,
                    "weight_replay": horizon == "remaining",
                    "new_cutoff_inference": False,
                    "training_recipe_archived": True,
                    "saved_weights": True,
                },
                "replay_contract": "rb_gpu_saved_policy_v1",
                "replay_entrypoint": "python -m experiments.future_player_lab.campaigns.scout_rb replay",
                "forecast_availability": "Saved Week 2 points plus original-weight stat inference"
                if horizon == "remaining"
                else "No saved 2026 preseason forecast; original fitted models and evaluation retained",
                "metrics": staging / f"{model_id}_metrics.json",
                "forecasts": [
                    {
                        "season": 2026,
                        "origin_week": 2,
                        "path": staging / "rb_remaining_2026_w2.json",
                    }
                ]
                if horizon == "remaining"
                else [],
                "artifacts": {
                    "original_library": archive,
                    "source_provenance": staging / "source_provenance.json",
                    "primary_policies": staging / "primary_policies.json",
                    "original_model_manifest": final / "manifest.json",
                    "original_forecast_manifest": preview / "manifest.json",
                    "original_forecast_features": preview / "features.parquet",
                    "original_feature_names": preview / "feature_names.json",
                    "original_forecast_points": preview / "rankings.parquet",
                    "original_research_manifest": study / "research_manifest.json",
                    "evaluation_predictions": staging / f"{model_id}_evaluation.parquet",
                    "adapter_source": staging / "adapter_source.py",
                },
            }
        )
    json_write(
        staging / "manifest.json",
        {
            "status": "complete",
            "kind": "scout_team_rb_adapter",
            "source_study_id": STUDY,
            "new_fits": 0,
            "models": [m["model_id"] for m in models],
            "files": {p.name: digest(p) for p in staging.iterdir() if p.is_file()},
        },
    )
    return models


def prepare_rb(data_root: Path, output: Path) -> list[dict]:
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Scout Team RB staging already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".scout-rb-", dir=output.parent) as temporary:
        staging = Path(temporary)
        models = _prepare(data_root, staging)

        def relocate(value):
            if isinstance(value, Path) and value.is_relative_to(staging):
                return output / value.relative_to(staging)
            if isinstance(value, dict):
                return {k: relocate(v) for k, v in value.items()}
            if isinstance(value, list):
                return [relocate(v) for v in value]
            return value

        result = relocate(models)
        shutil.move(staging, output)
    return result


def replay_archive(archive, expected_sha256, forecast):
    predicted = predict_archive(archive, expected_sha256)
    saved = json.loads(Path(forecast).read_text())
    rows = {row["player_id"]: row for row in saved["rows"]}
    if set(rows) != set(predicted["player_ids"]):
        raise ValueError("Packaged RB replay population differs")
    maximum = 0.0
    for index, player_id in enumerate(predicted["player_ids"]):
        for target in TARGETS:
            expected = rows[player_id]["predictions"][target]
            actual = predicted["predictions"][target][index]
            np.testing.assert_allclose(actual, expected, atol=1e-8, rtol=1e-10)
            maximum = max(maximum, abs(actual - expected))
    return {
        "status": "passed",
        "models": "original_rb_remaining",
        "rows": len(rows),
        "targets": len(TARGETS),
        "max_absolute_difference": maximum,
        "new_fits": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    replay = sub.add_parser("replay")
    replay.add_argument("--archive", type=Path, required=True)
    replay.add_argument("--sha256", required=True)
    replay.add_argument("--forecast", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(replay_archive(args.archive, args.sha256, args.forecast), indent=2))


if __name__ == "__main__":
    main()
