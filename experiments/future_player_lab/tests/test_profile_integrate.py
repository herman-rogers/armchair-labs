"""Annual cluster numbers must remain local in conditional forecast diagnostics."""

import json

import polars as pl
from experiments.future_player_lab.gpu_lab.profile_integrate import (
    aggregate_scoped_cohorts,
    analysis_ready,
)


def test_existing_prediction_files_do_not_imply_analysis_is_complete(tmp_path):
    folder = tmp_path / "analysis"
    folder.mkdir()
    (folder / "predictions.parquet").touch()
    assert not analysis_ready(tmp_path)
    path = folder / "manifest.json"
    path.write_text(json.dumps({"status": "running", "files": {}}))
    assert not analysis_ready(tmp_path)
    path.write_text(
        json.dumps(
            {
                "status": "complete",
                "files": {"predictions.parquet": "hash", "distributions.parquet": "hash"},
            }
        )
    )
    assert analysis_ready(tmp_path)
    path.write_text(
        json.dumps(
            {
                "status": "core_complete",
                "files": {"predictions.parquet": "hash", "distributions.parquet": "hash"},
            }
        )
    )
    assert analysis_ready(tmp_path)


def test_summary_counts_year_scoped_cohorts_without_merging_numeric_profile_labels():
    rows = pl.DataFrame(
        {
            "variant": ["standard_gmm"] * 2,
            "horizon": ["season"] * 2,
            "year": [2023, 2024],
            "profile": [0, 0],
            "target": ["points"] * 2,
            "recipe": ["ensemble_top3"] * 2,
            "n": [10, 30],
            "mse": [4.0, 16.0],
            "baseline_mse": [16.0, 4.0],
            "mse_improvement_pct": [75.0, -300.0],
            "coverage80": [0.9, 0.6],
            "interval_rows": [10, 30],
        }
    )
    summary = aggregate_scoped_cohorts(rows).row(0, named=True)
    assert summary["annual_cohort_cells"] == 2
    assert summary["cohort_cells_n20"] == 1
    assert summary["positive_skill_cells_n20"] == 0
    assert summary["matched_player_origins"] == 40
    assert summary["overall_rmse"] == 13**0.5
    assert summary["median_cohort_skill_n20"] == -300
    assert summary["median_cohort_coverage80_n20"] == 0.6
    assert "profile" not in summary
