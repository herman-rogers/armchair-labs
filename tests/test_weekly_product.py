import polars as pl
import pytest

from engine.data import weekly_points
from engine.data.releases import digest, reference, write_json


def test_weekly_product_verifies_artifacts_and_analysis_binding(tmp_path, monkeypatch):
    assert weekly_points.load_weekly_points(tmp_path) is None
    analysis = tmp_path / "research/analysis"
    analysis.mkdir(parents=True)
    write_json(analysis / "manifest.json", {"version": "analysis"})
    monkeypatch.setattr(weekly_points, "load_analysis", lambda _: (analysis, {}))
    root = tmp_path / "research/weekly"
    root.mkdir()
    write_json(root / "report.json", {"positions": []})
    pl.DataFrame({"prediction": [12.0]}).write_parquet(root / "predictions.parquet")
    write_json(
        root / "manifest.json",
        dict(
            version="weekly",
            kind="nextgen_weekly_points",
            status="complete",
            analysis=reference(analysis),
            files={name: digest(root / name) for name in ("report.json", "predictions.parquet")},
        ),
    )
    (tmp_path / "weekly").mkdir()
    write_json(tmp_path / "weekly/current.json", reference(root))
    assert weekly_points.load_weekly_points(tmp_path)["predictions"] == [{"prediction": 12.0}]
    # Warm verification cannot hide changed content.
    original = (root / "predictions.parquet").read_bytes()
    pl.DataFrame({"prediction": [999.0]}).write_parquet(root / "predictions.parquet")
    with pytest.raises(ValueError, match="artifact changed"):
        weekly_points.load_weekly_points(tmp_path)
    (root / "predictions.parquet").write_bytes(original)
    write_json(analysis / "manifest.json", {"version": "different-analysis"})
    with pytest.raises(ValueError, match="current analysis"):
        weekly_points.load_weekly_points(tmp_path)
