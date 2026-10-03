"""Carry retired, non-serving registry decisions across a weekly data refresh."""

from __future__ import annotations

from copy import deepcopy


def carry_retirements(registry, previous, *, evidence_unchanged, legacy_unchanged, source):
    old = {row["id"]: row for row in previous}
    result = deepcopy(registry)
    for row in result:
        prior = old.get(row["id"], {})
        if (
            row["validity"] != "revalidation_required"
            or prior.get("validity") != "archived"
            or row.get("allowed_uses")
            or prior.get("allowed_uses")
            or row.get("serving") != "archive"
            or prior.get("serving") != "archive"
        ):
            continue
        kind = row["kind"]
        same = (
            (kind == "research_stat" and evidence_unchanged)
            or (kind == "legacy_model" and legacy_unchanged)
            or (
                kind in {"study", "data_release"}
                and row.get("source_manifest_sha256")
                and row["source_manifest_sha256"] == prior.get("source_manifest_sha256")
            )
            or (kind == "forecast" and row["id"].startswith("unavailable:"))
        )
        if same:
            row.update(
                validity="archived",
                reason=prior["reason"],
                reconsideration=prior.get("reconsideration"),
                retirement_source=source,
            )
    return result
