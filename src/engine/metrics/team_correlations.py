"""Descriptive pair relationships and coherent complete-case lineup variance.

Every selected-player covariance uses the same games and denominator. Pairwise
correlations from different samples are never stitched into the lineup covariance.
"""

from itertools import combinations

import numpy as np
import polars as pl


def covariance(
    values: np.ndarray, seasons: np.ndarray, basis: str
) -> tuple[np.ndarray | None, int]:
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    centered = values.copy()
    if basis == "season":
        for year in np.unique(seasons):
            mask = seasons == year
            centered[mask] -= centered[mask].mean(axis=0)
        df = len(values) - len(np.unique(seasons))
    else:
        centered -= centered.mean(axis=0) if len(values) else 0
        df = len(values) - 1
    return (centered.T @ centered / df if df >= 2 else None), max(df, 0)


def correlation(cov: np.ndarray | None) -> float | None:
    if cov is None or cov[0, 0] <= 1e-12 or cov[1, 1] <= 1e-12:
        return None
    return float(np.clip(cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]), -1, 1))


def risk_intervals(values: np.ndarray, seasons: np.ndarray, basis: str, df: int) -> dict:
    """Exploratory paired-game bootstrap, retaining each season's sample size."""
    rng = np.random.default_rng(20261003)
    indices = np.tile(np.arange(len(values)), (2000, 1))
    for season in np.unique(seasons):
        subset = np.flatnonzero(seasons == season)
        indices[:, subset] = rng.choice(subset, size=(len(indices), len(subset)))
    draws = values[indices]
    if basis == "season":
        for season in np.unique(seasons):
            mask = seasons == season
            draws[:, mask] -= draws[:, mask].mean(axis=1, keepdims=True)
    else:
        draws -= draws.mean(axis=1, keepdims=True)
    variance = np.square(draws.sum(axis=2)).sum(axis=1) / df
    independent = np.square(draws).sum(axis=(1, 2)) / df
    valid = independent > 1e-12
    return dict(
        sd_interval=np.quantile(np.sqrt(variance), [0.025, 0.975]).tolist(),
        variance_change_interval=np.quantile(
            100 * (variance[valid] / independent[valid] - 1), [0.025, 0.975]
        ).tolist()
        if valid.sum() >= 1000
        else None,
    )


def player_summary(panel: pl.DataFrame, sample: str, basis: str, eligible: set[str]) -> list[dict]:
    players = []
    for (pid,), active in panel.group_by("player_id", maintain_order=True):
        starts = active.filter(pl.col("starter_game"))
        if pid not in eligible:
            continue
        rows = starts if sample == "starter" else active
        if rows.is_empty():
            continue
        values = rows["league_points"].to_numpy()
        cov, _ = covariance(values, rows["season"].to_numpy(), basis)
        players.append(
            dict(
                player_id=pid,
                name=active["name"][0],
                position=active["position"][0],
                games=len(rows),
                starter_games=len(starts),
                mean=float(values.mean()),
                sd=float(np.sqrt(cov[0, 0])) if cov is not None else None,
                first_season=int(active["season"].min()),
                last_season=int(active["season"].max()),
                last_game=active["gameday"].max(),
            )
        )
    return sorted(
        players,
        key=lambda p: (
            {"QB": 0, "WR": 1, "TE": 2}[p["position"]],
            -p["last_season"],
            -p["games"],
            p["name"],
        ),
    )


def shared(panel: pl.DataFrame, ids: list[str]) -> pl.DataFrame:
    frame = panel.filter(pl.col("player_id").is_in(ids)).pivot(
        on="player_id", index=["game_id", "season", "week", "gameday"], values="league_points"
    )
    for pid in ids:
        if pid not in frame.columns:
            frame = frame.with_columns(pl.lit(None, dtype=pl.Float64).alias(pid))
    return frame.drop_nulls(ids).sort("season", "week")


def pair_summary(panel: pl.DataFrame, a: str, b: str, basis: str) -> dict:
    data = shared(panel, [a, b])
    cov, df = covariance(data.select(a, b).to_numpy(), data["season"].to_numpy(), basis)
    r = correlation(cov)
    interval = None
    if r is not None and df > 2 and abs(r) < 1:
        interval = np.tanh(np.arctanh(r) + np.array([-1, 1]) * 1.96 / np.sqrt(df - 2)).tolist()
    return dict(
        a=a,
        b=b,
        n=len(data),
        df=df,
        r=r,
        interval=interval,
        covariance=float(cov[0, 1]) if cov is not None else None,
        years=sorted(data["season"].unique().to_list()),
        points=[
            dict(game_id=x["game_id"], season=x["season"], week=x["week"], a=x[a], b=x[b])
            for x in data.to_dicts()
        ],
    )


def lineup_risk(panel: pl.DataFrame, ids: list[str], basis: str) -> dict:
    result = dict(
        player_ids=ids,
        n=0,
        df=0,
        status="empty_selection",
        mean=None,
        sd=None,
        variance=None,
        independent_variance=None,
        independent_sd=None,
        covariance_effect=None,
        variance_change_pct=None,
        quantiles=None,
        contributions=[],
        covariance=[],
        games=[],
        sd_interval=None,
        variance_change_interval=None,
    )
    if not ids:
        return result
    data = shared(panel, ids)
    values = data.select(ids).to_numpy()
    totals = values.sum(axis=1)
    cov, df = covariance(values, data["season"].to_numpy(), basis)
    result.update(
        n=len(data),
        df=df,
        status="insufficient_overlap",
        games=[
            dict(game_id=row["game_id"], season=row["season"], week=row["week"], total=float(total))
            for row, total in zip(data.to_dicts(), totals, strict=True)
        ],
    )
    if cov is None or len(data) < 3:
        return result
    independent = float(np.trace(cov))
    variance = max(0.0, float(cov.sum()))
    result.update(
        status="available",
        mean=float(totals.mean()),
        sd=float(np.sqrt(variance)),
        variance=variance,
        independent_variance=independent,
        independent_sd=float(np.sqrt(independent)),
        covariance_effect=variance - independent,
        variance_change_pct=100 * (variance / independent - 1) if independent > 1e-12 else None,
        quantiles=dict(
            zip(
                ["p10", "p25", "p50", "p75", "p90"],
                np.quantile(totals, [0.1, 0.25, 0.5, 0.75, 0.9]).tolist(),
                strict=True,
            )
        ),
        covariance=cov.tolist(),
        contributions=[
            dict(
                player_id=pid,
                mean=float(values[:, i].mean()),
                variance=float(cov[i, i]),
                covariance=float(cov[i].sum() - cov[i, i]),
                total=float(cov[i].sum()),
            )
            for i, pid in enumerate(ids)
        ],
    )
    result.update(risk_intervals(values, data["season"].to_numpy(), basis, df))
    return result


def analyze_team(
    panel: pl.DataFrame,
    *,
    team: str,
    start: int,
    end: int,
    qb: str,
    sample: str,
    basis: str,
    selected: list[str] | None,
) -> dict:
    panel = panel.filter((pl.col("team") == team) & pl.col("season").is_between(start, end))
    # Establish roles before restricting QB starts. A new QB's short history
    # should expose small n, not remove established teammates from the selector.
    roster = panel.group_by("player_id").agg(
        pl.col("starter_game").sum().alias("starts"),
        (pl.col("position") == "QB").any().alias("quarterback"),
    )
    role_ids = set(roster.filter((pl.col("starts") >= 3) | pl.col("quarterback"))["player_id"])
    latest_season = panel["season"].max() or end
    quarterbacks = (
        panel.filter(pl.col("position") == "QB")
        .group_by("player_id", "name")
        .agg(
            pl.len().alias("games"),
            pl.col("gameday").max().alias("last_game"),
            (pl.col("season") == latest_season).sum().alias("recent_starts"),
        )
    )
    quarterbacks = quarterbacks.sort(
        ["recent_starts", "last_game", "games"], descending=True
    ).to_dicts()
    actual_qb = quarterbacks[0]["player_id"] if qb == "current" and quarterbacks else qb
    if actual_qb not in {"all", "current"} and actual_qb not in {
        row["player_id"] for row in quarterbacks
    }:
        raise ValueError("Quarterback is not available for this team and period")
    if actual_qb != "all":
        panel = panel.filter(pl.col("starting_qb_id") == actual_qb)
    players = player_summary(panel, sample, basis, role_ids)
    allowed = {p["player_id"] for p in players}
    if selected is not None and (
        len(selected) != len(set(selected)) or not set(selected) <= allowed
    ):
        raise ValueError("Selected players must be unique members of this team sample")
    if (
        selected is not None
        and sum(p["position"] == "QB" for p in players if p["player_id"] in selected) > 1
    ):
        raise ValueError("Select at most one starting quarterback")
    if selected is None:
        # Most recent QB and two most recently used WRs; no same-week points ranking.
        ordered = sorted(
            players,
            key=lambda p: (p["last_season"], p["starter_games"], p["last_game"]),
            reverse=True,
        )
        selected = [
            p["player_id"]
            for pos, limit in [("QB", 1), ("WR", 2)]
            for p in [r for r in ordered if r["position"] == pos][:limit]
        ]
    eligible = panel.filter(pl.col("starter_game")) if sample == "starter" else panel
    pairs = [
        pair_summary(eligible, a["player_id"], b["player_id"], basis)
        for a, b in combinations(players, 2)
    ]
    return dict(
        team=team,
        start=start,
        end=end,
        qb=actual_qb,
        sample=sample,
        basis=basis,
        quarterbacks=quarterbacks,
        players=players,
        pairs=pairs,
        team_games=panel["game_id"].n_unique(),
        risk=lineup_risk(eligible, selected, basis),
    )
