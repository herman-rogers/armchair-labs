"""Score frozen prospective offense lineups using their original joint distribution."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..weekly_v3.lineups import aggregate, interval_score
from ..weekly_v31.artifacts import verify
from ..weekly_v32.uncertainty import estimates
from .artifacts import check


def score_lineups(directory, actual_series, integration):
    directory = Path(directory)
    frozen = pd.read_parquet(directory / "lineups.parquet")
    if frozen.empty:
        return frozen
    receipt = verify(directory)
    contract = check(integration)
    dependence_path = Path(contract["dependencies"]["baseline"]["path"]) / "dependence.json"
    dependence = estimates(json.loads(dependence_path.read_text())[receipt["forecast_horizon"]])
    names = {"v3.2": "baseline/players", "v4": "players", "v4_retrained": "retrained"}
    frames = {
        key: pd.read_parquet(directory / f"{name}.parquet").set_index("player_id").to_dict("index")
        for key, name in names.items()
    }
    rows = []
    for row in frozen.to_dict("records"):
        ids = row["player_ids"].split(",")
        actuals = actual_series.reindex(ids)
        if actuals.isna().any():
            rows.append(
                dict(
                    row,
                    actual=None,
                    target_points=None,
                    scored=False,
                    pending_players=int(actuals.isna().sum()),
                )
            )
            continue
        actual = float(actuals.sum())
        if row["method"] not in frames:
            raise ValueError("Unknown frozen lineup method")
        roster = [dict(player_id=pid, **frames[row["method"]][pid]) for pid in ids]
        for player in roster:
            player["samples"] = list(player["samples"])
        values = aggregate(
            roster,
            dependence,
            seed=int(row["season"]) * 10000 + int(row["week"]) * 100 + int(row["team_id"]),
            actual=actual,
        )
        for key in ("prediction", "q10", "q50", "q90", "std"):
            if not np.isclose(values[key], row[key], rtol=0, atol=1e-10):
                raise ValueError(f"Frozen lineup distribution changed: {key}")
        rows.append(
            dict(
                row,
                **{
                    k: v
                    for k, v in values.items()
                    if k not in row and k not in {"crps", "independent_crps"}
                },
                crps=values["crps"],
                independent_crps=values["independent_crps"],
                actual=actual,
                target_points=actual,
                scored=True,
                pending_players=0,
                squared_error=(values["prediction"] - actual) ** 2,
                absolute_error=abs(values["prediction"] - actual),
                coverage80=float(values["q10"] <= actual <= values["q90"]),
                width80=values["q90"] - values["q10"],
                interval_score80=interval_score(actual, values["q10"], values["q90"]),
            )
        )
    return pd.DataFrame(rows)
