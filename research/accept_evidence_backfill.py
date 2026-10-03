"""Audit a completed backfill rebuild, report coverage changes, and accept for research."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
from artifact_inputs import require_audited_version

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n")


def coverage(frame):
    resolution = pl.col("cutoff_state_resolution")
    return {
        "rows": frame.height,
        "missing_names": frame["player_display_name"].null_count(),
        "missing_transaction_features": frame["cutoff_state_resolution"].null_count(),
        "dated_transaction_match": frame.filter(pl.col("cutoff_transaction_count") > 0).height,
        "observed_roster_state": frame.filter(resolution == "observed").height,
        "team_observed_status_unknown": frame.filter(
            resolution == "team_observed_status_unknown"
        ).height,
        "inferred_prior_team": frame.filter(resolution == "inferred_prior_team").height,
        "no_prior_team": frame.filter(resolution == "no_prior_team").height,
        "unresolved_action": frame.filter(resolution == "unresolved_action").height,
        "same_day_conflict": frame.filter(resolution == "conflicting_same_day").height,
        "known_absence": frame.filter(
            pl.col("availability_evidence_status") == "known_absence"
        ).height,
        "absence_unknown": frame.filter(pl.col("availability_evidence_status") == "unknown").height,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--collection", type=Path, required=True)
    args = parser.parse_args()
    version, previous, collection = args.version, args.baseline, args.collection
    manifest = json.loads((version / "manifest.json").read_text())
    if manifest["status"] != "rebuilt_pending_audit":
        raise ValueError("Rebuild must finish successfully before acceptance")
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "research/data_integrity_audit.py"),
            "--data-dir",
            str(version),
            "--require-clean",
        ],
        cwd=ROOT,
        check=True,
    )
    provenance = require_audited_version(version)
    for filename, expected in manifest["output_sha256"].items():
        assert digest(version / filename) == expected, f"Build output changed: {filename}"
    for filename, expected in manifest["protected_artifact_sha256"].items():
        assert digest(ROOT / filename) == expected, f"Protected artifact changed: {filename}"
    before = pl.read_parquet(previous / "outputs/metric_backtest_predictions.parquet")
    after = pl.read_parquet(version / "outputs/metric_backtest_predictions.parquet")
    keys = ["player_id", "forecast_season"]
    assert before.select(keys).join(after.select(keys), on=keys, how="anti").is_empty()
    assert after.select(keys).join(before.select(keys), on=keys, how="anti").is_empty()
    joined = before.join(after, on=keys, suffix="_after", validate="1:1")
    outcome_changes = {
        column: joined.filter(~pl.col(column).eq_missing(pl.col(column + "_after"))).height
        for column in ["actual_games", "actual_season_points", "outcome_complete"]
    }
    assert not any(outcome_changes.values()), outcome_changes
    by_group = []
    for year, population in (
        after.select("forecast_season", "player_population")
        .unique()
        .sort("forecast_season", "player_population")
        .iter_rows()
    ):
        condition = (pl.col("forecast_season") == year) & (
            pl.col("player_population") == population
        )
        by_group.append(
            {
                "forecast_season": year,
                "player_population": population,
                "before": coverage(before.filter(condition)),
                "after": coverage(after.filter(condition)),
            }
        )
    evidence_dir = version / "outputs/evidence_collection"
    evidence_dir.mkdir(exist_ok=False)
    shutil.copy2(ROOT / "research/data_integrity_audit.py", evidence_dir / "audit_procedure.py")
    shutil.copy2(Path(__file__), evidence_dir / "acceptance_procedure.py")
    for filename in [
        "collection.json",
        "archive_inventory.json",
        "article_index_inventory.json",
        "article_index.parquet",
        "absence_article_review.jsonl",
        "absence_legacy_review.jsonl",
        "absence_supplement_review.json",
        "approved_absence_reviews.json",
    ]:
        shutil.copy2(collection / filename, evidence_dir / filename)
    approved = json.loads((collection / "approved_absence_reviews.json").read_text())
    approved_urls = {row["source_url"] for row in approved}
    articles = {
        row["url"]: row
        for filename in ["absence_article_review.jsonl", "absence_legacy_review.jsonl"]
        for line in (collection / filename).read_text().splitlines()
        for row in [json.loads(line)]
    }
    articles.update(
        {
            r["url"]: r
            for r in json.loads((collection / "absence_supplement_review.json").read_text())
        }
    )
    review_queue = []
    for url, article in articles.items():
        decision = (
            "approved_explicit_fact" if url in approved_urls else "unreviewed_or_not_actionable"
        )
        reason = None
        if "dennis-pitta-out-for-season-john-harbaugh-confirms" in url:
            decision = "not_applied_superseded_outlook"
            reason = "Later August reporting reopened a possible return before the forecast cutoff."
        review_queue.append(
            {
                "url": url,
                "title": article["title"],
                "published_on": article["published_on"],
                "source_modified_at": article["modified_at"],
                "status": decision,
                "reason": reason,
                "capture_file": article["capture_file"],
                "source_sha256": article["sha256"],
            }
        )
    write(evidence_dir / "absence_review_status.json", review_queue)
    after.filter(
        pl.col("cutoff_state_resolution").is_in(
            [
                "unresolved_action",
                "conflicting_same_day",
                "no_prior_team",
                "inferred_prior_team",
            ]
        )
    ).select(
        *keys,
        "player_display_name",
        "player_population",
        "position",
        "forecast_cutoff_date",
        "cutoff_state_resolution",
        "cutoff_preseason_team",
        "cutoff_source_url",
        "cutoff_evidence_clause",
    ).write_parquet(version / "outputs/roster_evidence_review_queue.parquet")
    example_filter = (
        ((pl.col("player_display_name") == "Kareem Hunt") & (pl.col("forecast_season") == 2019))
        | ((pl.col("player_display_name") == "Julio Jones") & (pl.col("forecast_season") == 2021))
        | ((pl.col("player_display_name") == "Nick Chubb") & (pl.col("forecast_season") == 2024))
        | ((pl.col("player_display_name") == "Josh Oliver") & (pl.col("forecast_season") == 2022))
        | (
            (pl.col("player_display_name") == "Damien Williams")
            & (pl.col("forecast_season") == 2020)
        )
        | ((pl.col("player_display_name") == "Tom Brady") & (pl.col("forecast_season") == 2015))
    )
    columns = [
        "cutoff_preseason_team",
        "cutoff_state_resolution",
        "cutoff_availability_class",
        "known_absence_games",
        "known_available_games_cap",
        "fitted_games",
        "fitted_season_points",
    ]
    examples = (
        joined.filter(example_filter)
        .select(
            *keys,
            "player_display_name",
            "forecast_cutoff_date",
            *columns,
            *[column + "_after" for column in columns],
        )
        .to_dicts()
    )
    joined.filter(pl.col("player_display_name").is_null()).select(
        *keys, pl.col("player_display_name_after").alias("resolved_name")
    ).write_parquet(version / "outputs/backfilled_identity_names.parquet")
    comparison = {
        "version": version.name,
        "baseline": previous.name,
        "baseline_sha256": digest(previous / "outputs/metric_backtest_predictions.parquet"),
        "prediction_sha256": provenance["prediction_sha256"],
        "before": coverage(before),
        "after": coverage(after),
        "by_season_population": by_group,
        "outcome_changes": outcome_changes,
        "examples": examples,
        "source_collection": json.loads((collection / "collection.json").read_text()),
        "archive_status": dict(
            Counter(
                r["archive_status"]
                for r in json.loads((collection / "archive_inventory.json").read_text())
            )
        ),
        "captured_unique_absence_articles": len(articles),
        "reviewed_source_urls_applied": len(approved_urls),
        "news_review_status": dict(Counter(r["status"] for r in review_queue)),
    }
    write(version / "outputs/backfill_coverage_changes.json", comparison)
    audit = json.loads((version / "outputs/data_integrity_audit_2026-09-22.json").read_text())
    outputs = [
        version / "outputs/backfill_coverage_changes.json",
        version / "outputs/roster_evidence_review_queue.parquet",
        version / "outputs/backfilled_identity_names.parquet",
        version / "outputs/research_final_transaction_replay.parquet",
        *evidence_dir.iterdir(),
    ]
    acceptance = {
        "schema_version": 1,
        "version": version.name,
        "status": "accepted_for_research_not_promoted",
        "generated_at": datetime.now(UTC).isoformat(),
        "research_only": True,
        "audited_input": provenance,
        "acceptance_checks": audit["acceptance_checks"],
        "forecast_rows": after.height,
        "coverage_complete": False,
        "protected_artifacts_unchanged": True,
        "protected_artifact_count": len(manifest["protected_artifact_sha256"]),
        "verified_source_files": len(manifest["source_sha256"]),
        "research_output_sha256": {str(p.relative_to(version)): digest(p) for p in outputs},
        "limitations": [
            "Retrospective dated sources, not original publication-time snapshots.",
            "Collection and absence review remain incomplete; unknown is not available.",
            "Transaction matching covers every candidate population; "
            "roster completeness is unproven.",
            "No performance or model-promotion claim follows from improved source coverage.",
        ],
    }
    with (version / "acceptance.json").open("x") as stream:
        stream.write(json.dumps(acceptance, indent=2) + "\n")
    from engine.api.research_sources import accepted_version

    accepted_version(version.name)
    print(
        json.dumps(
            {
                "accepted": version.name,
                "before": coverage(before),
                "after": coverage(after),
                "examples": examples,
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
