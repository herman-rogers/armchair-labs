"""Explicit QB efficiency and opportunity computed exclusively before each origin."""

from collections import defaultdict

import numpy as np
import polars as pl

from ..notebooks.raw_distribution_data import TEAM_CODES


def augment_qb(panel, tables):
    players, seasons, schedules = defaultdict(list), defaultdict(list), defaultdict(set)
    for row in tables["box"].sort("season", "week").iter_rows(named=True):
        players[row["player_id"]].append(row)
        seasons[row["season"]].append(row)
    for row in tables["schedule"].iter_rows(named=True):
        for field in ("home_team", "away_team"):
            team = TEAM_CODES.get(row[field], row[field])
            schedules[row["season"], team].add(row["week"])
    league_rates = {}
    for year in set(panel["year"]):
        old = [r for y, rows in seasons.items() if y < year for r in rows]
        league_rates[year] = sum(r.get("passing_yards") or 0 for r in old) / max(
            sum(r.get("attempts") or 0 for r in old), 1
        )
    output = []
    for row in panel.iter_rows(named=True):
        year, origin, pid = row["year"], row["origin"], row["player_id"]
        history = [
            r
            for r in players[pid]
            if r["season"] < year or (r["season"] == year and r["week"] <= origin)
        ]
        prior = [r for r in history if r["season"] == year - 1]
        current = [r for r in history if r["season"] == year]
        career = [r for r in history if r["season"] < year]
        passing = [r for r in history if (r.get("attempts") or 0) > 0]
        league = league_rates[year]

        def total(rows, field):
            return sum(r.get(field) or 0 for r in rows)

        def rate(rows, prior_attempts=400, half=None, year=year, league=league):
            a = y = 0.0
            for r in rows:
                if r.get("attempts") is None or r.get("passing_yards") is None:
                    continue
                if r["attempts"] <= 0:
                    continue
                weight = 2 ** (-(year - r["season"]) / half) if half else 1
                a += weight * r["attempts"]
                y += weight * r["passing_yards"]
            return (y + prior_attempts * league) / (a + prior_attempts)

        features = {
            "league_ypa": league,
            "career_ypa_h2_p400": rate(passing, 400, 2),
            "career_ypa_h0p5_p150": rate(passing, 150, 0.5),
            "last8_ypa_p400": rate(passing[-8:]),
            "current_ypa_p400": rate(current),
            "prior_ypa_p400": rate(prior),
            "prior_attempts": total(prior, "attempts"),
            "prior_yards": total(prior, "passing_yards"),
            "prior_last4_attempts": np.mean([r.get("attempts") or 0 for r in prior[-4:]])
            if prior
            else None,
        }
        for name, subset in (
            ("career", passing),
            ("recent16", passing[-16:]),
            ("current", [r for r in passing if r["season"] == year]),
        ):
            features[f"{name}_log_attempts"] = np.log1p(total(subset, "attempts"))
            for field in (
                "completions",
                "passing_tds",
                "passing_interceptions",
                "passing_epa",
                "carries",
                "rushing_yards",
                "rushing_tds",
            ):
                observed = [r for r in subset if r.get(field) is not None]
                a = total(observed, "attempts")
                features[f"{name}_{field}_per_attempt"] = total(observed, field) / a if a else None
        team = TEAM_CODES.get(row["team"], row["team"])
        elapsed = sorted(w for w in schedules[year, team] if w <= origin)
        by_week = {r["week"]: r for r in current}
        last3 = [by_week.get(w, {}) for w in elapsed[-3:]]
        team_rows = [
            r
            for r in seasons[year]
            if r["week"] <= origin and TEAM_CODES.get(r["team"], r["team"]) == team
        ]
        team_a, player_a = total(team_rows, "attempts"), total(current, "attempts")
        player_team_a = total([r for r in team_rows if r["player_id"] == pid], "attempts")
        primary = [r for r in career if (r.get("attempts") or 0) >= 15]
        reserve = [r for r in career if 0 < (r.get("attempts") or 0) < 15]
        first = min((r["season"] for r in career), default=year)
        exposure = sum(
            (17 if y >= 2021 else 16) * 2 ** (-(year - y) / 3) for y in range(first, year)
        )
        features.update(
            {
                "current_team_attempt_share": player_team_a / team_a if team_a else None,
                "competing_attempts_per_game": (team_a - player_team_a) / max(len(elapsed), 1),
                "elapsed_team_games": len(elapsed),
                "current_attempts_per_team_game": player_a / max(len(elapsed), 1),
                "last_attempts": (last3[-1].get("attempts") or 0) if last3 else None,
                "last3_attempts": total(last3, "attempts") / len(last3) if last3 else None,
                "last3_primary_share": sum((r.get("attempts") or 0) >= 15 for r in last3)
                / len(last3)
                if last3
                else None,
                "career_primary_share": (
                    sum(2 ** (-(year - r["season"]) / 3) for r in primary) + 0.3
                )
                / (exposure + 1),
                "career_reserve_share": (
                    sum(2 ** (-(year - r["season"]) / 3) for r in reserve) + 0.12
                )
                / (exposure + 1),
                "primary_attempts": (total(primary, "attempts") + 160) / (len(primary) + 5),
                "reserve_attempts": (total(reserve, "attempts") + 25) / (len(reserve) + 5),
            }
        )
        # Missing records and recorded unknown attempts are not observed zeros.
        # The same coverage rule holds for every historical/current origin.
        observed_weeks = {r["week"] for r in current if r.get("attempts") is not None}
        features["current_observation_known"] = float(bool(observed_weeks))
        features["current_attempt_coverage"] = (
            len(set(elapsed) & observed_weeks) / len(elapsed) if elapsed else None
        )
        if not elapsed or not set(elapsed).issubset(observed_weeks):
            for name in (
                "current_log_attempts",
                "current_team_attempt_share",
                "competing_attempts_per_game",
                "current_attempts_per_team_game",
            ):
                features[name] = None
        if not last3 or any(r.get("attempts") is None for r in last3):
            features["last3_attempts"] = None
            features["last3_primary_share"] = None
        if not last3 or last3[-1].get("attempts") is None:
            features["last_attempts"] = None
        output.append({"qb__" + k: v for k, v in features.items()})
    extra = pl.DataFrame(output, infer_schema_length=None).cast(pl.Float64)
    return panel.hstack(extra)
