"""Descriptive same-position career-stage neighbors; not predictions or talent grades."""

import numpy as np
import polars as pl

from patron.metrics.career_stat_evidence import apply_zero_reviews

MIN_COMPARISON_WEEKS = 8
# A descriptive coverage floor, not a calibrated statistical confidence level.
MIN_COVERAGE = 0.8

METRICS = {
    "QB": ["attempts", "passing_yards", "carries", "rushing_yards"],
    "RB": ["carries", "rushing_yards", "targets", "receiving_yards"],
    "WR": ["targets", "receptions", "receiving_yards"],
    "TE": ["targets", "receptions", "receiving_yards"],
}


def career_neighbors(identities, weeks, player_id, complete_through, limit=5):
    people = {r["player_id"]: r for r in identities.to_dicts()}
    identity = people.get(player_id, {})
    metrics = METRICS.get(identity.get("position"), [])
    result = dict(
        matches=[],
        metrics=metrics,
        seasons_compared=0,
        eligible_peers=0,
        complete_through=complete_through,
        coverage=[],
        method_version=2,
        minimum_comparison_weeks=MIN_COMPARISON_WEEKS,
        minimum_coverage=MIN_COVERAGE,
        method="Same position and first N NFL career seasons; workload "
        "and yardage per week with all comparison measures available, "
        "compared season by season. Source-verified zero-stat appearances are included. "
        "All measures share the same covered-week denominator within a season. "
        "Each metric/year is scaled by "
        "the comparison pool's standard deviation. Lower RMS distance is closer.",
        limitations="Descriptive statistical resemblance, not a ranking, forecast or "
        "probability. Not adjusted for era, teammates, scheme, age or injuries. "
        "Requires at least eight covered weeks and 80% coverage of observed weeks "
        "in every compared season. Unverified missing stats stay unknown and their "
        "weeks are omitted from rates, which can overstate workload when omitted "
        "weeks had little activity. The 80% floor is a coverage policy, not a "
        "validated confidence level. Incomplete early careers are excluded.",
    )

    def unavailable(reason):
        return {**result, "reason": reason}

    if (
        not metrics
        or identity.get("rookie_season") is None
        or identity.get("history_left_truncated")
    ):
        return unavailable("A verified position and complete early NFL history are required.")
    observed = weeks.filter(
        (pl.col("season") <= complete_through)
        & ((pl.col("stat_recorded") == True) | (pl.col("offense_snaps") > 0)).fill_null(False)  # noqa: E712
    )
    target = observed.filter(pl.col("player_id") == player_id)
    if target.is_empty():
        return unavailable(
            "No completed NFL season is available yet. "
            "Partial rookie seasons are not forced into career comparisons."
        )
    count = int(target["season"].max()) - int(identity["rookie_season"]) + 1
    if count < 1:
        return unavailable("Recorded entry year does not precede the captured career history.")
    result["seasons_compared"] = count
    pool_ids = [
        pid
        for pid, p in people.items()
        if p.get("position") == identity["position"]
        and p.get("rookie_season") is not None
        and not p.get("history_left_truncated")
    ]
    observed, reviews = apply_zero_reviews(
        observed.filter(pl.col("player_id").is_in(pool_ids)), metrics
    )
    observed = observed.with_columns(
        pl.all_horizontal([pl.col(m).is_finite().fill_null(False) for m in metrics]).alias(
            "covered"
        )
    )
    annual = observed.group_by("player_id", "season").agg(
        pl.len().alias("observed_weeks"),
        pl.col("covered").sum().alias("comparison_weeks"),
        pl.col("week").filter(~pl.col("covered")).sort().alias("missing_weeks"),
        pl.col("verified_zero").sum().alias("verified_zero_weeks"),
        *[pl.col(m).filter(pl.col("covered")).mean().alias(m) for m in metrics],
    )
    groups = {}
    for row in annual.to_dicts():
        row["coverage_fraction"] = row["comparison_weeks"] / row["observed_weeks"]
        groups.setdefault(row["player_id"], {})[row["season"]] = row
    start = int(identity["rookie_season"])
    result["coverage"] = [
        groups.get(player_id, {}).get(
            year,
            dict(
                season=year,
                observed_weeks=0,
                comparison_weeks=0,
                coverage_fraction=0.0,
                missing_weeks=[],
                verified_zero_weeks=0,
            ),
        )
        for year in range(start, start + count)
    ]
    candidates = []
    for pid in pool_ids:
        p = people[pid]
        start = int(p["rookie_season"])
        if start + count - 1 > complete_through:
            continue
        years = [groups.get(pid, {}).get(start + n) for n in range(count)]
        if any(
            r is None
            or r["comparison_weeks"] < MIN_COMPARISON_WEEKS
            or r["coverage_fraction"] < MIN_COVERAGE
            or any(r[m] is None for m in metrics)
            for r in years
        ):
            continue
        vector = [r[m] for r in years for m in metrics]
        if not np.isfinite(vector).all():
            continue
        candidates.append(
            dict(
                player_id=pid,
                player_display_name=p["player_display_name"],
                from_season=start,
                through_season=start + count - 1,
                observed_weeks=sum(r["observed_weeks"] for r in years),
                comparison_weeks=sum(r["comparison_weeks"] for r in years),
                verified_zero_weeks=sum(r["verified_zero_weeks"] for r in years),
                zero_stat_reviews=[
                    r
                    for r in reviews
                    if r["player_id"] == pid and start <= r["season"] < start + count
                ],
                seasons=[{**r, "career_year": i + 1} for i, r in enumerate(years)],
                vector=vector,
            )
        )
    target_index = next((i for i, c in enumerate(candidates) if c["player_id"] == player_id), None)
    result["eligible_peers"] = len(candidates) - int(target_index is not None)
    if target_index is None:
        gaps = [
            f"{r['season']}: {r['comparison_weeks']}/{r['observed_weeks']} weeks covered"
            for r in result["coverage"]
            if r["comparison_weeks"] < MIN_COMPARISON_WEEKS or r["coverage_fraction"] < MIN_COVERAGE
        ]
        return unavailable(
            "Career statistics are present, but comparison coverage is insufficient. "
            "Each season needs at least eight covered weeks and 80% coverage. "
            + "; ".join(gaps)
            + "."
        )
    result["target"] = {k: v for k, v in candidates[target_index].items() if k != "vector"}
    if len(candidates) < 6:
        return unavailable(
            "Fewer than five comparable careers have sufficient coverage at this cutoff."
        )
    values = np.asarray([c["vector"] for c in candidates])
    scale = values.std(axis=0)
    scale[scale == 0] = 1
    distances = np.sqrt(np.mean(((values - values[target_index]) / scale) ** 2, axis=1))
    for candidate, distance in zip(candidates, distances, strict=True):
        candidate.pop("vector")
        candidate["distance"] = float(distance)
    result["target"] = candidates[target_index]
    result["matches"] = sorted(
        (c for c in candidates if c["player_id"] != player_id),
        key=lambda c: (c["distance"], c["player_id"]),
    )[:limit]
    result["reason"] = None
    return result
