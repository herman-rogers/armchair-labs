"""Matched bridge-control total uncertainty evidence, with explicit sparse support."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .uncertainty import evaluate_totals


def evaluate_bridge(path):
    """Return score/support tables for a newly calibrated v3.2 bridge control.

    The bridge itself served no long-horizon distributions. These distributions
    are a research control fitted exclusively to its earlier matched total errors.
    """
    original = pd.read_parquet(Path(path))
    # Definitions must agree within a paired score. Keep every exclusion visible
    # in the support table rather than silently mixing different actual targets.
    paired = original[original.label_matches.eq(True)].copy()
    base = paired.copy()
    base["prediction"], base["method"] = base.prediction_v32, "v32_bridge_calibrated_control"
    combined = pd.concat([paired, base], ignore_index=True)
    scores, support = [], []
    for (position, origin), raw in original.groupby(["position", "origin_week"]):
        group = combined[combined.position.eq(position) & combined.origin_week.eq(origin)]
        for year in (2024, 2025):
            counts = group[group.season.lt(year)].groupby("method").size()
            test_rows = int(
                (
                    paired.position.eq(position)
                    & paired.origin_week.eq(origin)
                    & paired.season.eq(year)
                ).sum()
            )
            valid = len(counts) == 2 and counts.min() >= 100 and test_rows > 0
            support.append(
                dict(
                    position=position,
                    origin_week=int(origin),
                    season=year,
                    prior_rows_per_method=int(counts.min()) if len(counts) else 0,
                    test_rows_per_method=test_rows,
                    excluded_prior_label_mismatches=int(
                        (raw.season.lt(year) & ~raw.label_matches.eq(True)).sum()
                    ),
                    excluded_test_label_mismatches=int(
                        (raw.season.eq(year) & ~raw.label_matches.eq(True)).sum()
                    ),
                    supported=bool(valid),
                    minimum_prior_rows=100,
                    note="New research-only total distributions; original bridge "
                    "served none. 2023 seeds later evaluation.",
                )
            )
            if valid:
                scores.append(evaluate_totals(group, evaluation_years=(year,)))
    return (
        pd.concat(scores, ignore_index=True) if scores else pd.DataFrame(),
        pd.DataFrame(support),
    )
