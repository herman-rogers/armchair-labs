"""Offline, separately versioned historical rebuild; never publishes live artifacts.

Run: .venv/bin/python research/rebuild_history.py --version historical_v2_20260922_r1
Every version is create-only. Failed runs are retained with a failure manifest.
Raw cached inputs and configuration are copied into the version for reproducibility.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import socket
import subprocess
import sys
import traceback
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n")


def protected_paths():
    return sorted(
        {
            *(ROOT / "data/static/draft_2026").rglob("*"),
            ROOT / "data/static/experimental_2026_predictions.csv",
            *(ROOT / "data/outputs").glob("board*.json"),
            ROOT / "data/outputs/metric_backtest_predictions.parquet",
            ROOT / "src/patron/config/experimental_freeze_2026.yaml",
        }
        - {p for p in (ROOT / "data/static/draft_2026").rglob("*") if p.is_dir()}
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,100}", args.version):
        parser.error("version must be a simple directory name")
    version = ROOT / "data/research" / args.version
    version.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    protected = {str(p.relative_to(ROOT)): digest(p) for p in protected_paths() if p.is_file()}
    manifest = {
        "version": args.version,
        "schema_version": 2,
        "status": "building",
        "started_at": datetime.now(UTC).isoformat(),
        "research_only": True,
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "protected_artifact_sha256": protected,
        "policies": {
            "positions": "Historical weekly roster positions, FB/HB as RB; V1 unchanged",
            "transactions": "official dated sources only; uncertain states remain unknown",
            "contracts": "Canonical money requires signing and availability dates; "
            "term also requires effective start year",
            "routes": "matched dropback universe including scrambles; incomplete coverage null",
            "depth_charts": "no synthetic Week 1 publication dates; timestamped 2025+ only",
            "xfp": "quarantined from model inputs pending historical model-vintage evidence",
            "known_absences": "dated official evidence; team-game sets cap games before products",
        },
        "limitations": [
            "Reconstruction from current cached historical statistics, "
            "not original publication vintages.",
            "Official dated transaction archives may be incomplete; "
            "absence is not proof of active status.",
            "Historical market ADP is not same-date executable ESPN/FAAB price.",
            "2026 is a new research forecast only, not a replacement for the frozen forecast.",
        ],
    }
    write_json(version / "manifest.json", manifest)
    try:
        # Copies isolate even library cache expiry/cleanup from the user's source cache.
        shutil.copytree(ROOT / "data/cache/nflverse", version / "cache/nflverse")
        shutil.copytree(ROOT / "data/static", version / "static")
        evidence_cache = ROOT / "data/cache/historical_evidence"
        if evidence_cache.exists():
            shutil.copytree(evidence_cache, version / "cache/historical_evidence")
        derived = version / "cache/derived"
        derived.mkdir()
        for pattern in (
            "nfl_transactions_cutoff_v1_*.parquet",
            "nfl_official_transactions_cutoff_v1_*.parquet",
        ):
            sources = sorted((ROOT / "data/cache/derived").glob(pattern))
            if len(sources) != 1:
                raise RuntimeError(
                    f"Expected one unambiguous raw archive for {pattern}, got {sources}"
                )
            shutil.copy2(sources[0], derived / sources[0].name)
        manifest["source_sha256"] = {
            str(p.relative_to(version)): digest(p)
            for folder in (
                version / "cache/nflverse",
                derived,
                version / "static",
                version / "cache/historical_evidence",
            )
            for p in sorted(folder.rglob("*"))
            if p.is_file()
        }
        shutil.copytree(
            ROOT / "src/patron",
            version / "implementation/patron",
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        shutil.copytree(
            ROOT / "research",
            version / "implementation/research",
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        manifest["implementation_sha256"] = {
            str(p.relative_to(version / "implementation")): digest(p)
            for p in sorted((version / "implementation").rglob("*"))
            if p.is_file()
        }
        os.environ["DATA_DIR"] = str(version)
        os.environ["NFLVERSE_CACHE_DURATION"] = "3153600000"

        # Fail loudly on a cache miss. No remote refresh or live-season outcomes.
        def no_network(*args, **kwargs):
            raise RuntimeError("Offline rebuild: required source is not in the frozen cache")

        socket.socket.connect = no_network
        socket.socket.connect_ex = no_network

        from patron import pipeline
        from patron.config.settings import Settings, get_settings
        from patron.data import nflverse
        from patron.metrics.backtest import MetricReportConfig

        get_settings.cache_clear()
        settings = Settings(data_dir=version, nflverse_cache_duration=3153600000)
        nflverse.configure_cache(settings)
        config = MetricReportConfig.from_config()
        manifest["report_config"] = asdict(config)
        write_json(version / "manifest.json", manifest)
        inputs = pipeline.build_metric_report(
            settings=settings, report_config=config, analyze=False
        )
        inputs.predictions.write_parquet(settings.outputs_dir / "historical_inputs.parquet")
        from patron.metrics.availability import coverage_audit

        coverage, review = coverage_audit(inputs.predictions)
        write_json(settings.outputs_dir / "availability_coverage.json", coverage)
        review.write_parquet(settings.outputs_dir / "availability_review_queue.parquet")
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "research/data_integrity_audit.py"),
                "--data-dir",
                str(version),
                "--inputs-only",
                "--require-clean",
            ],
            cwd=ROOT,
            check=True,
        )
        # Save unresolved records as a review queue, not silently inferred truths.
        inputs.predictions.filter(
            pl.col("cutoff_state_resolution").is_not_null()
            & (pl.col("cutoff_state_resolution") != "observed")
        ).select(
            "forecast_season",
            "player_id",
            "player_display_name",
            "cutoff_preseason_team",
            "cutoff_state_resolution",
            "cutoff_source_url",
            "cutoff_evidence_clause",
            "cutoff_evidence",
        ).write_parquet(settings.outputs_dir / "transaction_review_queue.parquet")
        logging.info(
            "Historical inputs rebuilt: %s rows; fitting walk-forward models",
            inputs.predictions.height,
        )
        report, predictions = pipeline._analyze_with_fit(inputs.predictions, config)
        report["historical_rebuild"] = {
            "version": args.version,
            "schema_version": 2,
            "research_only": True,
            "policies": manifest["policies"],
            "limitations": manifest["limitations"],
        }
        result = pipeline.MetricReportBuildResult(report=report, predictions=predictions)
        pipeline.export_metric_report(result, settings.outputs_dir)
        manifest.update(
            status="rebuilt_pending_audit",
            rows=predictions.height,
            completed_at=datetime.now(UTC).isoformat(),
        )
        manifest["output_sha256"] = {
            str(p.relative_to(version)): digest(p)
            for p in sorted(settings.outputs_dir.glob("*"))
            if p.is_file()
        }
    except BaseException as error:
        manifest.update(status="failed", error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        after = {name: digest(ROOT / name) for name in protected}
        manifest["protected_unchanged"] = after == protected
        write_json(version / "manifest.json", manifest)
        if after != protected:
            raise RuntimeError(
                "Protected artifact changed during rebuild; investigate before acceptance"
            )
    print(f"Rebuilt {version}; audit required before acceptance", flush=True)


if __name__ == "__main__":
    main()
