"""Accepted descriptive datasets; statistical definitions stay in the metrics module."""

from itertools import combinations

import numpy as np
import polars as pl

from engine.data.team_analysis import TABLES, _panel
from engine.metrics.team_correlations import correlation, covariance
from engine.tables.build import BuildContext
from engine.tables.storage import content_hash


def team_player_games(context: BuildContext) -> pl.DataFrame:
    cutoff = context.observation_cutoff
    if not cutoff:
        raise ValueError("team_player_games requires a common observation cutoff")
    names = [name.replace("_schedule", "_observed_schedule") for name in TABLES]
    paths = tuple(str(context.dependencies[name]) for name in names)
    panel = _panel(
        paths,
        content_hash(dict(zip(names, paths, strict=True))),
        cutoff["season"],
        cutoff["through_week"],
    )
    return panel.filter(pl.col("season") >= context.parameters["start_season"])


PAIR_SCHEMA = pl.Schema(
    {
        "team": pl.String(),
        "scope": pl.String(),
        "start_season": pl.Int64(),
        "end_season": pl.Int64(),
        "qb_scope": pl.String(),
        "sample": pl.String(),
        "basis": pl.String(),
        "player_a_id": pl.String(),
        "player_b_id": pl.String(),
        "player_a": pl.String(),
        "player_b": pl.String(),
        "position_a": pl.String(),
        "position_b": pl.String(),
        "n": pl.Int64(),
        "degrees_of_freedom": pl.Int64(),
        "correlation": pl.Float64(),
        "covariance": pl.Float64(),
        "ci_low": pl.Float64(),
        "ci_high": pl.Float64(),
        "shared_seasons": pl.List(pl.Int64),
    }
)


def player_pair_correlations(context: BuildContext) -> pl.DataFrame:
    panel = context.read("team_player_games")
    rows = []
    for (team,), team_panel in panel.group_by("team", maintain_order=True):
        years = sorted(team_panel["season"].unique().to_list())
        periods = [("pooled", years[0], years[-1]), *[("season", y, y) for y in years]]
        for scope, start, end in periods:
            period = team_panel.filter(pl.col("season").is_between(start, end))
            # Pooled results retain all QB regimes; the game table supports custom conditioning.
            players = (
                period.group_by("player_id", maintain_order=True)
                .agg(
                    pl.col("name").first(),
                    pl.col("position").first(),
                    pl.col("starter_game").sum().alias("starts"),
                )
                .filter(pl.col("starts") >= 3)
                .sort("player_id")
                .to_dicts()
            )
            for sample in ("starter", "active"):
                eligible = period.filter(pl.col("starter_game")) if sample == "starter" else period
                if not players:
                    continue
                # Pivot once per sample, not once per pair. Every covariance still uses
                # only that pair's shared appearances and the existing metric definition.
                wide = eligible.pivot(
                    on="player_id", index=["game_id", "season"], values="league_points"
                )
                years_array = wide["season"].to_numpy()
                values = {p["player_id"]: wide[p["player_id"]].to_numpy() for p in players}
                for a, b in combinations(players, 2):
                    pair = np.column_stack([values[a["player_id"]], values[b["player_id"]]])
                    present = np.isfinite(pair).all(axis=1)
                    pair = pair[present]
                    shared_years = years_array[present]
                    n = len(pair)
                    if n < 3:
                        continue
                    for basis in ("raw", "season"):
                        cov, df = covariance(pair, shared_years, basis)
                        r = correlation(cov)
                        interval = [None, None]
                        if r is not None and df > 2 and abs(r) < 1:
                            interval = np.tanh(
                                np.arctanh(r) + np.array([-1, 1]) * 1.96 / np.sqrt(df - 2)
                            ).tolist()
                        rows.append(
                            dict(
                                team=team,
                                scope=scope,
                                start_season=start,
                                end_season=end,
                                qb_scope="all",
                                sample=sample,
                                basis=basis,
                                player_a_id=a["player_id"],
                                player_b_id=b["player_id"],
                                player_a=a["name"],
                                player_b=b["name"],
                                position_a=a["position"],
                                position_b=b["position"],
                                n=n,
                                degrees_of_freedom=df,
                                correlation=r,
                                covariance=float(cov[0, 1]) if cov is not None else None,
                                ci_low=interval[0],
                                ci_high=interval[1],
                                shared_seasons=sorted(set(shared_years.tolist())),
                            )
                        )
    return pl.DataFrame(rows, schema=PAIR_SCHEMA)
