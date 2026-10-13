"""Capture actual pregame offense rosters, independently of artificial lineups."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from engine.names import normalize

from ..weekly_v3.lineups import NONSTARTERS, POSITIONS, aggregate
from . import calibration
from .artifacts import timestamp, write


def freeze_inputs(directory, root, season, week, captured_at):
    path = Path(root) / "data/outputs/league_snapshot.json"
    if not path.exists():
        return "No local league snapshot; player/artificial-lineup capture continues"
    payload = json.loads(path.read_text())
    if payload["season"] != season or payload["week"] != week:
        return "Local league snapshot is for another week; no league forecast captured"
    if timestamp(payload["captured_at"]) > captured_at:
        raise ValueError("League snapshot timestamp is after capture")
    from engine.tables import observations
    from engine.tables.application import connect, query_scope

    with query_scope(Path(root) / "data"), connect() as db:
        identities = observations.espn_ids(db)
    write(Path(directory) / "league_inputs.json", dict(snapshot=payload, identities=identities))
    return "Captured actual offense rosters; K/DST excluded from this experiment"


def predict(directory, players, dependence, calibrator):
    path = Path(directory) / "league_inputs.json"
    if not path.exists():
        return pd.DataFrame(), []
    inputs = json.loads(path.read_text())
    snapshot = inputs["snapshot"]
    identities = {int(k): v for k, v in inputs["identities"].items()}
    by_id = {r["player_id"]: r for r in players.to_dict("records")}
    by_name = {}
    for row in by_id.values():
        by_name.setdefault((normalize(row.get("player_name") or ""), row["position"]), []).append(
            row
        )
    estimates = {
        (r["relationship"], *sorted([r["position_a"], r["position_b"]])): r
        for r in dependence["correlations"]
    }
    results, unavailable = [], []
    seen = set()
    for game in snapshot["week_lineups"]:
        if game["week"] != snapshot["week"]:
            continue
        for side in ("home", "away"):
            team_id = game[f"{side}_team_id"]
            if team_id in seen:
                raise ValueError("Duplicate league team in forecast week")
            seen.add(team_id)
            starters = [
                r
                for r in game[f"{side}_lineup"]
                if r["slot"] not in NONSTARTERS and r["position"] in POSITIONS
            ]
            roster = []
            for player in starters:
                match = by_id.get(identities.get(player["espn_id"]))
                if match is None:
                    matches = by_name.get(
                        (normalize(player["player_display_name"]), player["position"]), []
                    )
                    match = matches[0] if len(matches) == 1 else None
                if match is not None and match["position"] == player["position"]:
                    roster.append(match)
            counts = pd.Series([r["position"] for r in roster]).value_counts()
            shape_ok = (
                len(roster) == 8
                and counts.get("QB", 0) == 1
                and counts.get("RB", 0) >= 2
                and counts.get("WR", 0) >= 3
                and counts.get("TE", 0) >= 1
            )
            if (
                len(roster) != len(starters)
                or not shape_ok
                or len({r["player_id"] for r in roster}) != len(roster)
            ):
                unavailable.append(
                    dict(
                        team_id=team_id,
                        starters=len(starters),
                        matched=len(roster),
                        reason="Incomplete identities or unsupported roster shape",
                    )
                )
                continue
            value = aggregate(
                roster,
                estimates,
                seed=snapshot["season"] * 10000 + snapshot["week"] * 100 + team_id,
            )
            results.append(
                dict(
                    value,
                    season=snapshot["season"],
                    week=snapshot["week"],
                    position="LINEUP",
                    design="league_offense",
                    team_id=team_id,
                    league_id=snapshot["league_id"],
                    target_points=np.nan,
                    player_ids=",".join(r["player_id"] for r in roster),
                    has_qb_receiver_stack=any(
                        a["position"] == "QB"
                        and b["position"] in ("WR", "TE")
                        and a.get("team")
                        and a["team"] == b.get("team")
                        for a in roster
                        for b in roster
                    ),
                )
            )
    frame = pd.DataFrame(results)
    if not frame.empty:
        frame = calibration.apply(frame, calibrator)
    return frame, unavailable
