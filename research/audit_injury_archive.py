"""Seal and independently audit provenance, temporal replay, and forecast effects.

No models are fitted. Existing immutable forecasts are used for rule checks.
"""

import gzip
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import polars as pl
from build_injury_archive import CACHE, GOLD, OUT, ROOT, digest
from experiments.future_player_lab.notebooks import rb_availability as availability
from experiments.future_player_lab.notebooks import workbench as data

from patron.data.historical_evidence import article_content


@lru_cache(maxsize=128)
def captured_content(filename):
    return article_content(gzip.decompress((CACHE / filename).read_bytes()).decode())


def main():
    manifest_path = OUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("sealed"):
        raise ValueError("Already sealed; do not mutate completed archive")
    ledger = json.loads((OUT / "absence_events.json").read_text())
    source_reviews = {
        r["evidence_id"]: r for r in json.loads((OUT / "absence_source_reviews.json").read_text())
    }
    for e in ledger["events"]:
        if e["evidence_id"] not in source_reviews:
            continue
        r = source_reviews[e["evidence_id"]]
        e.update(
            source_capture_file=r["capture_file"],
            source_sha256=r["sha256"],
            source_capture_status=r["status_code"],
            source_modified_at=r["modified_at"],
            source_published_at=r["published_at"],
        )
        if r["status_code"] != 200:
            e["certainty"] = "unverified_source"
    base_ids = {e["evidence_id"] for e in ledger["events"]}
    ledger["events"].extend(
        e
        for e in json.loads((OUT / "reviewed_additions.json").read_text())
        if e["evidence_id"] not in base_ids
    )
    (OUT / "absence_events.json").write_text(json.dumps(ledger, indent=2))
    recovered = pl.concat(
        [
            pl.read_parquet(OUT / "corroborated_reports.parquet"),
            pl.read_parquet(OUT / "manual_corroborated_reports.parquet"),
        ],
        how="diagonal_relaxed",
    ).unique("record_id")
    recovered.write_parquet(OUT / "corroborated_reports.parquet")

    # Pass 1: every retained source reference resolves to exactly the captured bytes.
    captures = {}

    def register(r):
        filename = r.get("source_capture_file", r.get("capture_file"))
        expected = r.get("source_sha256", r.get("sha256"))
        if filename and expected:
            if filename in captures and captures[filename] != expected:
                raise AssertionError("Conflicting hash for one capture")
            captures[filename] = expected

    for p in OUT.glob("*.jsonl"):
        for line in p.read_text().splitlines():
            register(json.loads(line))
    for p in OUT.glob("*.json"):
        value = json.loads(p.read_text())
        if isinstance(value, list):
            for r in value:
                register(r)
        elif "events" in value:
            for r in value["events"]:
                register(r)
        elif "capture_file" in value:
            register(value)
    for r in recovered.to_dicts():
        register(r)
    for filename, expected in captures.items():
        assert Path(filename).name == filename
        raw = gzip.decompress((CACHE / filename).read_bytes())
        assert hashlib.sha256(raw).hexdigest() == expected, filename
    original = pl.read_parquet(OUT / "injury_reports.parquet")
    pinned = pl.read_parquet(GOLD / "tables/nfl_injuries.parquet")
    assert original.equals(pinned), "Original timestamps/observations changed"
    lookup = {str(r["record_id"]): r for r in original.to_dicts()}
    for r in recovered.to_dicts():
        old = lookup[str(r["original_record_id"])]
        assert old["date_modified"] is None
        for key in ["season", "week", "gsis_id", "report_status", "report_primary_injury"]:
            assert old[key] == r[key], (key, r)
        content = captured_content(r["source_capture_file"])
        assert content["published_at"] == r["source_published_at"]
        assert content["modified_at"] == r["source_modified_at"]
        assert r["date_modified"] == max(
            datetime.fromisoformat(content[c].replace("Z", "+00:00"))
            for c in ["published_at", "modified_at"]
        )
        assert r["practice_status"] is None
    coverage = []
    early = pl.read_parquet(OUT / "early_official_reports.parquet")
    for year in range(2001, 2026):
        rows = original.filter(pl.col("season") == year)
        fixed = recovered.filter(pl.col("season") == year)
        older = early.filter(pl.col("season") == year)
        coverage.append(
            dict(
                season=year,
                original_reports=rows.height,
                original_undated=rows["date_modified"].null_count(),
                corroborated_undated_records=fixed["original_record_id"].n_unique(),
                corroborated_rb_records=fixed.filter(pl.col("position") == "RB")[
                    "original_record_id"
                ].n_unique(),
                added_early_official_rows=older.height,
                early_team_weeks=older.select("team", "week").n_unique(),
            )
        )
    pl.DataFrame(coverage).write_csv(OUT / "coverage_by_season.csv")
    early.join(pl.read_parquet(GOLD / "tables/players.parquet"), on="gsis_id", how="left").filter(
        pl.col("rookie_season") > pl.col("season")
    ).write_csv(OUT / "early_identity_conflicts.csv")
    assert pl.read_csv(OUT / "early_identity_conflicts.csv").height == 0
    print(
        f"Pass 1: {len(captures)} capture hashes verified; original injury rows unchanged",
        flush=True,
    )

    # Bind inputs before loading through the same verified path the notebook uses.
    manifest["files"] = {
        p.name: digest(p) for p in OUT.iterdir() if p.is_file() and p.name != "manifest.json"
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    events, schedule, preseason, injuries = availability.load_inputs(ROOT, allow_unsealed=True)
    actual_weeks = data.target_weeks()
    replay, zero_rows, annotations_by_horizon = [], [], {}
    for horizon in ["season", "next_week", "next_four", "remaining"]:
        panel, _ = data.task_data("RB", horizon)
        annotations = availability.annotate(panel, events, schedule, preseason, injuries)
        annotations_by_horizon[horizon] = (panel, annotations)
        # Adding future evidence must not change any historical row.
        future = dict(
            events[0],
            evidence_id="audit-future",
            absence_id="audit-future",
            source_published_on="2099-01-01",
            known_on="2099-01-01",
        )
        assert annotations.equals(
            availability.annotate(panel, events + [future], schedule, preseason, injuries)
        )
        forced = annotations["availability_force_zero"].to_numpy()
        for target in data.POSITION_TARGETS["RB"]:
            labeled = data.stat_panel(panel, actual_weeks, target)
            f = labeled.select(*data.KEYS, "name", "actual", "persistence_prediction").with_columns(
                pl.lit(123.0).alias("prediction")
            )
            adjusted = availability.adjust_forecasts(f, annotations)
            assert adjusted["actual"].equals(labeled["actual"])
            assert (adjusted.filter(pl.col("availability_force_zero"))["prediction"] == 0).all()
            assert (
                adjusted.filter(pl.col("availability_force_zero"))["persistence_prediction"] == 0
            ).all()
            assert np.array_equal(
                adjusted["prediction"].to_numpy()[~forced], f["prediction"].to_numpy()[~forced]
            )
            zero_rows.append(
                adjusted.filter(pl.col("availability_force_zero")).with_columns(
                    pl.lit(target).alias("stat")
                )
            )
        replay.append(
            dict(
                horizon=horizon,
                rows=panel.height,
                forced_zero_players=int(forced.sum()),
                evidence_rows=annotations.filter(
                    pl.col("availability_evidence_ids").list.len() > 0
                ).height,
                dated_report_rows=annotations.filter(
                    pl.col("availability_dated_reports") > 0
                ).height,
                conflicts=annotations.filter(
                    pl.col("availability_conflicts").list.len() > 0
                ).height,
            )
        )
    all_zeros = pl.concat(zero_rows)
    contradictions = all_zeros.filter(pl.col("actual").is_finite() & (pl.col("actual") != 0))
    all_zeros.write_parquet(OUT / "forced_zero_replay.parquet")
    contradictions.write_parquet(OUT / "absence_outcome_disagreements.parquet")
    # Disagreement is retained and flagged; neither labels nor evidence are silently changed.
    (OUT / "temporal_replay.json").write_text(json.dumps(replay, indent=2))
    print(
        f"Pass 2: {sum(r['rows'] for r in replay)} historical rows replayed; "
        f"{contradictions.height} outcome disagreements",
        flush=True,
    )

    # Pass 3: rule-only comparison using existing immutable cached predictions.
    comparisons = []
    panel, annotations = annotations_by_horizon["remaining"]
    joined_annotations = panel.select(data.KEYS).hstack(annotations)
    # The per-stat artifact lives in the notebook trial cache; locate from its metadata.
    for path in (ROOT / "experiments/future_player_lab/runs").glob(
        "notebook_tree_*/forecasts.parquet"
    ):
        forecast = pl.read_parquet(path)
        if not set(data.KEYS + ["actual", "prediction", "persistence_prediction"]) <= set(
            forecast.columns
        ):
            continue
        forecast = forecast.filter(
            (pl.col("position") == "RB") & (pl.col("horizon") == "remaining")
        )
        if not forecast.height:
            continue
        matched = (
            forecast.with_row_index("_order")
            .join(joined_annotations, on=data.KEYS, how="left", validate="m:1")
            .sort("_order")
        )
        assert matched["availability_force_zero"].null_count() == 0
        ann = matched.select(pl.col("^availability_.*$"))
        adjusted = availability.adjust_forecasts(forecast, ann)
        raw_error = np.abs(adjusted["actual"].to_numpy() - adjusted["raw_prediction"].to_numpy())
        new_error = np.abs(adjusted["actual"].to_numpy() - adjusted["prediction"].to_numpy())
        comparisons.append(
            dict(
                artifact=str(path.relative_to(ROOT)),
                rows=forecast.height,
                adjusted_players=adjusted.filter(pl.col("availability_force_zero")).height,
                raw_mae=float(np.nanmean(raw_error)),
                adjusted_mae=float(np.nanmean(new_error)),
            )
        )
    (OUT / "existing_forecast_audit.json").write_text(json.dumps(comparisons, indent=2))
    print(f"Pass 3: {len(comparisons)} existing forecast artifacts checked, no fitting", flush=True)
    news = [json.loads(s) for s in (OUT / "rb_news.jsonl").read_text().splitlines()]
    summary = dict(
        checked_at=datetime.now(UTC).isoformat(),
        verified_capture_files=len(captures),
        original_reports=original.height,
        original_missing_timestamp=original["date_modified"].null_count(),
        corroborated_original_records=recovered["original_record_id"].n_unique(),
        corroborated_rb_records=recovered.filter(pl.col("position") == "RB")[
            "original_record_id"
        ].n_unique(),
        early_official_rows=early.height,
        early_unresolved_identities=early.filter(pl.col("gsis_id").is_null()).height,
        news_candidates=len(news),
        news_http_statuses=dict(Counter(str(n["status_code"]) for n in news)),
        news_with_publication_date=sum(bool(n["published_at"]) for n in news),
        reviewed_absence_events=len(events),
        outcome_disagreements=contradictions.height,
        temporal_replay=replay,
        existing_forecast_artifacts=len(comparisons),
        models_trained=0,
    )
    (OUT / "audit_summary.json").write_text(json.dumps(summary, indent=2))
    manifest.update(sealed=True, audited_at=summary["checked_at"])
    manifest["files"] = {
        p.name: digest(p) for p in OUT.iterdir() if p.is_file() and p.name != "manifest.json"
    }
    manifest["captures"] = captures
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
