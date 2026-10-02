"""A retirement review cannot silently promote forecasts or drop pending entries."""

from copy import deepcopy

import polars as pl
import pytest
from research.review_revalidation import apply_decisions, stat_decision

from patron.data.nextgen import eligible


def pending(kind="legacy_model"):
    return dict(
        id="pending",
        kind=kind,
        validity="revalidation_required",
        serving="archive",
        allowed_uses=[],
        evidence="untested",
        reason="Old reason",
    )


def decision(**kwargs):
    return dict(
        category="retired", reason="Retired with evidence", reconsideration="New run", **kwargs
    )


def test_review_preserves_serving_and_unreviewed_evidence():
    rows = [pending(), dict(id="shadow", validity="verified", serving="shadow", allowed_uses=[])]
    before = deepcopy(rows)
    result = apply_decisions(rows, {"pending": decision()}, "review", "2026-09-24")
    assert rows == before
    assert result[1] == before[1]
    assert result[0]["validity"] == "archived"
    assert not eligible(result[0], use="forecast")[0]


def test_dependency_revalidation_cannot_approve_predictions():
    review = dict(
        category="verified_dependency",
        validity="verified",
        reason="Checked",
        reconsideration="Changed inputs",
    )
    result = apply_decisions([pending("study")], {"pending": review}, "review", "2026-09-24")
    assert result[0]["validity"] == "verified"
    assert not eligible(result[0], use="forecast")[0]
    with pytest.raises(ValueError, match="cannot promote"):
        apply_decisions([pending()], {"pending": review}, "review", "2026-09-24")


@pytest.mark.parametrize("decisions", [{}, {"pending": decision(), "extra": decision()}])
def test_review_requires_exhaustive_exact_coverage(decisions):
    with pytest.raises(ValueError, match="exactly"):
        apply_decisions([pending()], decisions, "review", "2026-09-24")


def test_review_rejects_duplicate_ids_and_existing_permissions():
    with pytest.raises(ValueError, match="Duplicate"):
        apply_decisions([pending(), pending()], {"pending": decision()}, "review", "2026-09-24")
    with pytest.raises(ValueError, match="serving permissions"):
        apply_decisions(
            [{**pending(), "allowed_uses": ["forecast"]}],
            {"pending": decision()},
            "review",
            "2026-09-24",
        )


@pytest.mark.parametrize(
    "status,values",
    [("constant", [0.0, 0.0, None]), ("unavailable_in_gold", [None, float("nan"), None])],
)
def test_recheck_stat_missingness_and_variation(status, values):
    frame = pl.DataFrame({"stat": values})
    finite = sum(v is not None and v == v for v in values)
    evidence = {
        "stat": dict(
            status=status,
            finite_rows=finite,
            total_rows=3,
            reason="Saved reason",
            definition_ref="source",
        )
    }
    entry = dict(id="stat:stat", reason="Saved reason")
    result = stat_decision(entry, evidence, frame)
    assert result["evidence_details"]["finite_rows"] == finite
    with pytest.raises(ValueError, match="coverage changed"):
        stat_decision(entry, evidence, pl.DataFrame({"stat": [1.0, 2.0, 3.0]}))


def test_constant_with_new_variation_requires_review():
    evidence = {
        "stat": dict(
            status="constant",
            finite_rows=2,
            total_rows=2,
            reason="Saved reason",
            definition_ref="source",
        )
    }
    with pytest.raises(ValueError, match="no longer constant"):
        stat_decision(
            dict(id="stat:stat", reason="Saved reason"),
            evidence,
            pl.DataFrame({"stat": [0.0, 1.0]}),
        )
