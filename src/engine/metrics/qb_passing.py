"""Chronological QB passing forecasts: exposure evidence is separate from opportunity.

No player-name rules, injury-season deletions or future-start labels are used.
Historical origins reconstruct the information cutoff from preserved provider data;
they are not original historical snapshots.
"""

from __future__ import annotations

import re
from collections import defaultdict
from datetime import UTC, date, datetime, timedelta

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from engine.metrics.current_rankings import team_code
from engine.metrics.experimental import _normalized_name
from engine.metrics.nextgen import season_length
from engine.metrics.transaction_events import interpret_event, player_clause, resolve_day

HORIZONS = ("next_game", "next4", "rest_of_season")
REFERENCES = ("prior_season", "current_pace", "career_reference", "adaptive_ridge")
CHALLENGERS = ("direct_boost", "conditional_boost")
MODELS = REFERENCES + CHALLENGERS
BOOST = dict(
    max_iter=100,
    learning_rate=0.05,
    max_leaf_nodes=15,
    min_samples_leaf=60,
    l2_regularization=10,
    max_bins=63,
    early_stopping=False,
    random_state=20260924,
)


def news_known_at_issue(note, issued_at):
    """Allow same-day updates only when the captured source predates this issue."""
    known = note.get("known_on", "9999")
    if known > issued_at[:10]:
        return False
    captured = note.get("retrieved_at")
    if not captured:
        return known < issued_at[:10]
    try:
        capture_time = datetime.fromisoformat(captured.replace("Z", "+00:00"))
        issue_time = datetime.fromisoformat(issued_at.replace("Z", "+00:00"))
        if capture_time.tzinfo is None or issue_time.tzinfo is None:
            return False
        return capture_time.astimezone(UTC) <= issue_time.astimezone(UTC)
    except ValueError:
        return False


def compile_events(transactions, identities, candidates, supplements=()):
    """Resolve official transaction clauses to unique QB identities, retaining conflicts.

    ESPN event dates have known historical errors and are intentionally excluded.
    Reviewed news is supplementary, not a claim of complete roster-exit coverage.
    """
    ids = set(candidates["player_id"])
    aliases = defaultdict(set)
    for row in identities.to_dicts():
        if row["gsis_id"] in ids:
            aliases[_normalized_name(row["display_name"])].add(row["gsis_id"])
    for row in candidates.select("player_id", "player_display_name").unique().to_dicts():
        aliases[_normalized_name(row["player_display_name"])].add(row["player_id"])
    unique = {name: next(iter(pids)) for name, pids in aliases.items() if len(pids) == 1}
    matcher = re.compile(
        r"\b(?:" + "|".join(map(re.escape, sorted(unique, key=len, reverse=True))) + r")\b"
    )
    events = []
    for row in transactions.filter(pl.col("category") != "espn").to_dicts():
        for name in set(matcher.findall(_normalized_name(row["description"]))):
            clause = player_clause(row["description"], {name}, _normalized_name)
            parsed = interpret_event(row, clause, None, matched_aliases={name})
            if parsed["action"] not in {"unparsed", "contract", "draft", "physical"}:
                events.append(
                    dict(
                        parsed,
                        player_id=unique[name],
                        source_sha256=row.get("source_sha256"),
                        date_basis=row.get("date_basis", "dated transaction"),
                    )
                )
    for row in supplements:
        pid = unique.get(_normalized_name(row["player_name"]))
        if pid is None:
            raise ValueError(f"Unresolved supplemental QB identity: {row['player_name']}")
        events.append(
            dict(
                player_id=pid,
                transaction_date=date.fromisoformat(row["known_on"]),
                team=row.get("team"),
                status="off" if row["action"] == "retire" else "active",
                action=row["action"],
                clause=row["action"],
                category="official",
                source_url=row["source_url"],
                source_sha256=row["sha256"],
                date_basis=row["date_basis"],
            )
        )
    grouped = defaultdict(list)
    for event in events:
        grouped[event["player_id"], event["transaction_date"]].append(event)
    resolved = []
    for (pid, day), group in sorted(grouped.items()):
        event, resolution = resolve_day(group)
        resolved.append(dict(event, player_id=pid, known_on=str(day), resolution=resolution))
    return resolved


def execution_evidence(history, season, league_ypa=7.0):
    """Attempt-weighted, recency-weighted career evidence; zero attempts add no YPA data."""
    attempts = yards = 0.0
    raw_attempts = raw_yards = 0.0
    for row in history:
        a, y = row.get("attempts"), row.get("passing_yards")
        if a is None or y is None or a <= 0:
            continue
        weight = 2 ** (-max(season - row["season"], 0) / 3)
        attempts += weight * a
        yards += weight * y
        raw_attempts += a
        raw_yards += y
    return dict(
        execution_ypa=(yards + 100 * league_ypa) / (attempts + 100),
        effective_attempts=attempts,
        evidence_attempts=raw_attempts,
        observed_career_ypa=raw_yards / raw_attempts if raw_attempts else None,
        prior_weight=100 / (attempts + 100),
    )


def _sum(rows, field):
    return sum((r.get(field) or 0) for r in rows)


def build_panel(
    features,
    weeks,
    schedule,
    identities,
    events,
    *,
    season,
    through_week,
    current_issue_date,
    absences=(),
):
    """Freeze candidates, features, schedule and targets separately at every origin."""
    feature_names = [
        "player_id",
        "player_display_name",
        "position",
        "forecast_season",
        "forecast_cutoff_date",
        "player_population",
        "cutoff_preseason_team",
        "cutoff_preseason_status",
        "cutoff_state_resolution",
    ]
    candidates = defaultdict(dict)
    for r in features.filter(pl.col("position") == "QB").select(feature_names).to_dicts():
        candidates[r["forecast_season"]][r["player_id"]] = r
    # A later position label cannot delete a candidate's passing observations.
    qb_ids = set(features.filter(pl.col("position") == "QB")["player_id"]) | set(
        weeks.filter(pl.col("position") == "QB")["player_id"]
    )
    week_rows = weeks.filter(pl.col("player_id").is_in(qb_ids) & (pl.col("season_type") == "REG"))
    by_player, by_season, event_map = defaultdict(list), defaultdict(list), defaultdict(list)
    for r in week_rows.sort("season", "week").to_dicts():
        if r["season"] > season or r["season"] == season and r["week"] > through_week:
            continue
        by_player[r["player_id"]].append(r)
        by_season[r["season"]].append(r)
    for event in events:
        event_map[event["player_id"]].append(event)
    people = {r["gsis_id"]: r for r in identities.to_dicts()}
    games, dates = defaultdict(dict), defaultdict(list)
    for r in schedule.filter(pl.col("game_type") == "REG").to_dicts():
        day = date.fromisoformat(str(r["gameday"])[:10])
        dates[r["season"], r["week"]].append(day)
        for team in (r["home_team"], r["away_team"]):
            games[r["season"], team_code(team)][r["week"]] = day
    panel = []
    for year in sorted(y for y in candidates if 2004 <= y <= season):
        prior_league = [r for y, rows in by_season.items() if y < year for r in rows]
        league_ypa = _sum(prior_league, "passing_yards") / max(_sum(prior_league, "attempts"), 1)
        max_week = max(w for y, w in dates if y == year)
        origins = range(0, through_week + 1 if year == season else max_week)
        for origin in origins:
            pool = dict(candidates[year])
            for r in by_season[year]:
                if r["position"] == "QB" and r["week"] <= origin and r["player_id"] not in pool:
                    pool[r["player_id"]] = dict(
                        player_id=r["player_id"],
                        player_display_name=r["player_display_name"],
                        player_population="observed_by_cutoff",
                        forecast_cutoff_date=f"{year}-08-25",
                    )
            for pid, c in sorted(pool.items()):
                issue = (
                    date.fromisoformat(str(c["forecast_cutoff_date"])[:10])
                    if origin == 0
                    else max(dates[year, origin]) + timedelta(days=1)
                )
                if year == season and origin == through_week:
                    issue = date.fromisoformat(current_issue_date)
                all_history = by_player[pid]
                history = [
                    r
                    for r in all_history
                    if r["season"] < year or r["season"] == year and r["week"] <= origin
                ]
                prior = [r for r in history if r["season"] == year - 1]
                current = [r for r in history if r["season"] == year]
                career = [r for r in history if r["season"] < year]
                last_seen = history[-1] if history else None
                latest_game = (
                    games.get((last_seen["season"], team_code(last_seen["team"])), {}).get(
                        last_seen["week"], date.min
                    )
                    if last_seen
                    else date.min
                )
                dated = [e for e in event_map[pid] if e["known_on"] < str(issue)]
                event = max(dated, key=lambda e: e["known_on"]) if dated else None
                # A subsequently observed game supersedes an older inactive event.
                effective = event if event and event["known_on"] > str(latest_game) else None
                team = team_code(current[-1]["team"] if current else c.get("cutoff_preseason_team"))
                if effective and effective.get("team"):
                    team = team_code(effective["team"])
                if not team and last_seen:
                    team = team_code(last_seen["team"])
                team_schedule = games.get((year, team), {})
                scheduled = sorted(team_schedule) if team_schedule else list(range(1, max_week + 1))
                elapsed = [w for w in scheduled if w <= origin]
                by_week = {r["week"]: r for r in current}
                current_games = [by_week.get(w, dict(attempts=0, passing_yards=0)) for w in elapsed]
                last3 = current_games[-3:]
                ev = execution_evidence(history, year, league_ypa)
                person = people.get(pid, {})
                draft_year = person.get("draft_year")
                draft_known = draft_year is not None and draft_year <= year
                birth = person.get("birth_date")
                primary = [r for r in career if (r.get("attempts") or 0) >= 15]
                reserve = [r for r in career if 0 < (r.get("attempts") or 0) < 15]
                first = min((r["season"] for r in career), default=year)
                exposure = sum(
                    season_length(y) * 2 ** (-(year - y) / 3) for y in range(first, year)
                )
                weighted_primary = sum(2 ** (-(year - r["season"]) / 3) for r in primary)
                weighted_reserve = sum(2 ** (-(year - r["season"]) / 3) for r in reserve)
                career_primary = (weighted_primary + 0.3) / (exposure + 1)
                career_reserve = (weighted_reserve + 0.12) / (exposure + 1)
                p_primary = (
                    2 * career_primary + sum((r.get("attempts") or 0) >= 15 for r in last3)
                ) / (2 + len(last3))
                p_reserve = (
                    2 * career_reserve + sum(0 < (r.get("attempts") or 0) < 15 for r in last3)
                ) / (2 + len(last3))
                primary_attempts = (_sum(primary, "attempts") + 32 * 5) / (len(primary) + 5)
                reserve_attempts = (_sum(reserve, "attempts") + 5 * 5) / (len(reserve) + 5)
                state = effective.get("status", "unknown") if effective else "unknown"
                retired = bool(
                    effective
                    and effective["action"] == "retire"
                    and effective["resolution"] != "conflicting_same_day"
                )
                role_group = (
                    "preseason"
                    if not origin
                    else "current_substantial"
                    if last3 and (last3[-1].get("attempts") or 0) >= 15
                    else "current_low"
                )
                sample_group = "sparse_history" if ev["evidence_attempts"] < 100 else "established"
                base = dict(
                    player_id=pid,
                    player_display_name=c["player_display_name"],
                    season=year,
                    through_week=origin,
                    issue_date=str(issue),
                    team=team,
                    population=c["player_population"],
                    role_group=role_group,
                    sample_group=sample_group,
                    schedule_known=bool(team_schedule),
                    retired_known=retired,
                    roster_state=state,
                    roster_source=effective.get("source_url") if effective else None,
                    roster_known_on=effective.get("known_on") if effective else None,
                    medical_status="unknown",
                    current_yards=_sum(current, "passing_yards"),
                    current_attempts=_sum(current, "attempts"),
                    prior_attempts=_sum(prior, "attempts"),
                    prior_yards=_sum(prior, "passing_yards"),
                    **ev,
                    prior_rate=_sum(prior, "passing_yards") / season_length(year - 1)
                    if prior
                    else None,
                    current_rate=_sum(current, "passing_yards") / len(elapsed) if elapsed else None,
                    reference_primary=p_primary,
                    reference_reserve=p_reserve,
                    reference_primary_attempts=primary_attempts,
                    reference_reserve_attempts=reserve_attempts,
                    x_execution=ev["execution_ypa"],
                    x_log_exposure=np.log1p(ev["effective_attempts"]),
                    x_prior_attempts=_sum(prior, "attempts"),
                    x_prior_yards=_sum(prior, "passing_yards"),
                    x_career_primary=career_primary,
                    x_career_reserve=career_reserve,
                    x_primary_attempts=primary_attempts,
                    x_reserve_attempts=reserve_attempts,
                    x_prior_primary=sum((r.get("attempts") or 0) >= 15 for r in prior)
                    / season_length(year - 1),
                    x_prior_ypa=_sum(prior, "passing_yards") / _sum(prior, "attempts")
                    if _sum(prior, "attempts")
                    else None,
                    x_current_attempts=_sum(current_games, "attempts") / max(len(elapsed), 1),
                    x_current_yards=_sum(current_games, "passing_yards") / max(len(elapsed), 1),
                    x_last_attempts=(last3[-1].get("attempts") or 0) if last3 else None,
                    x_last3_attempts=_sum(last3, "attempts") / len(last3) if last3 else None,
                    x_last3_primary=sum((r.get("attempts") or 0) >= 15 for r in last3) / len(last3)
                    if last3
                    else None,
                    x_elapsed=len(elapsed),
                    x_origin=origin,
                    x_age=(date(year, 9, 1) - date.fromisoformat(str(birth)[:10])).days / 365.25
                    if birth
                    else None,
                    x_draft=person.get("draft_pick") if draft_known else None,
                    x_rookie=float(c["player_population"] == "rookie"),
                    x_off=float(state == "off"),
                    x_reserve=float(state in {"reserve", "practice"}),
                    x_state_known=float(state != "unknown"),
                )
                future_weeks = [w for w in scheduled if w > origin]
                if not future_weeks:
                    continue
                for horizon in HORIZONS:
                    selected = (
                        future_weeks[:1]
                        if horizon == "next_game"
                        else future_weeks[:4]
                        if horizon == "next4"
                        else future_weeks
                    )
                    end, count = selected[-1], len(selected)
                    future = [
                        r for r in all_history if r["season"] == year and origin < r["week"] <= end
                    ]
                    complete = year < season
                    future_primary = [r for r in future if (r.get("attempts") or 0) >= 15]
                    future_reserve = [r for r in future if 0 < (r.get("attempts") or 0) < 15]
                    known_absences = [
                        a
                        for a in absences
                        if a["player_id"] == pid
                        and a["season"] == year
                        and a["known_on"] < str(issue)
                    ]
                    # The most recent review can revoke an earlier announcement.
                    latest_absence = (
                        max(known_absences, key=lambda a: a["known_on"]) if known_absences else None
                    )
                    absent_slots = (
                        set(latest_absence["unavailable_games"]) if latest_absence else set()
                    )
                    unavailable = sum((scheduled.index(w) + 1) in absent_slots for w in selected)
                    allowed_fraction = 0.0 if retired else (count - unavailable) / count
                    row = dict(
                        base,
                        horizon=horizon,
                        end_week=end,
                        scheduled_games=count,
                        x_horizon_games=count,
                        x_known_unavailable_fraction=1 - allowed_fraction,
                        allowed_fraction=allowed_fraction,
                        actual=_sum(future, "passing_yards") if complete else None,
                        actual_attempts=_sum(future, "attempts") if complete else None,
                        actual_primary=len(future_primary) / count if complete else None,
                        actual_reserve=len(future_reserve) / count if complete else None,
                    )
                    for state_name, subset in (
                        ("primary", future_primary),
                        ("reserve", future_reserve),
                    ):
                        row[f"actual_{state_name}_attempts"] = (
                            _sum(subset, "attempts") / len(subset) if complete and subset else None
                        )
                        row[f"actual_{state_name}_yards"] = (
                            _sum(subset, "passing_yards") / len(subset)
                            if complete and subset
                            else None
                        )
                    panel.append(row)
    return panel


def constrain_totals(values, rows):
    """Hard zero for a fully unavailable horizon; no second partial-absence discount."""
    return np.maximum(np.asarray(values, dtype=float), 0) * np.array(
        [float(r["allowed_fraction"] > 0) for r in rows]
    )


def fit_fold(train, test):
    """A fixed matched model family; all fitted transforms use earlier seasons only."""
    columns = sorted(k for k in train[0] if k.startswith("x_"))

    def matrix(rows):
        return np.array(
            [[np.nan if r.get(c) is None else r[c] for c in columns] for r in rows], dtype=float
        )

    x, z = matrix(train), matrix(test)
    y = np.array([r["actual"] / r["scheduled_games"] for r in train])
    n = np.array([r["scheduled_games"] for r in test])
    # One completed candidate per year, within population, matches the legacy
    # no-history fallback without overweighting repeated weekly origins.
    population_rates = defaultdict(list)
    for r in train:
        if r["through_week"] == 0 and r["horizon"] == "rest_of_season":
            population_rates[r["population"]].append(r["actual"] / season_length(r["season"]))
    fallback = float(np.mean([v for rows in population_rates.values() for v in rows]))
    fallback_by_population = {k: float(np.mean(v)) for k, v in population_rates.items()}
    priors = np.array(
        [
            r["prior_rate"]
            if r["prior_rate"] is not None
            else fallback_by_population.get(r["population"], fallback)
            for r in test
        ]
    )
    current = np.array(
        [
            r["current_rate"] if r["current_rate"] is not None else p
            for r, p in zip(test, priors, strict=True)
        ]
    )
    career = np.array(
        [
            r["execution_ypa"]
            * (
                r["reference_primary"] * r["reference_primary_attempts"]
                + r["reference_reserve"] * r["reference_reserve_attempts"]
            )
            for r in test
        ]
    )
    for i, row in enumerate(test):
        if row["evidence_attempts"] == 0 and row["prior_rate"] is None:
            career[i] = priors[i]
    with threadpool_limits(limits=1):
        ridge = make_pipeline(
            SimpleImputer(add_indicator=True, keep_empty_features=True),
            StandardScaler(),
            Ridge(alpha=100),
        )
        ridge.fit(x, y)
        direct = HistGradientBoostingRegressor(**BOOST).fit(x, y)
        components = {}
        for state in ("primary", "reserve"):
            prob = HistGradientBoostingRegressor(**BOOST).fit(
                x, [r[f"actual_{state}"] for r in train]
            )
            p = np.clip(prob.predict(z), 0, 1)
            mask = np.array([r[f"actual_{state}_yards"] is not None for r in train])
            subset = [r for r, keep in zip(train, mask, strict=True) if keep]
            attempts = HistGradientBoostingRegressor(**BOOST).fit(
                x[mask], [r[f"actual_{state}_attempts"] for r in subset]
            )
            a = np.clip(
                attempts.predict(z),
                15 if state == "primary" else 1,
                80 if state == "primary" else 14,
            )
            residual = [
                r[f"actual_{state}_yards"] - r["execution_ypa"] * r[f"actual_{state}_attempts"]
                for r in subset
            ]
            adjustment = HistGradientBoostingRegressor(**BOOST).fit(x[mask], residual)
            yards = np.maximum(
                np.array([r["execution_ypa"] for r in test]) * a + adjustment.predict(z), 0
            )
            components[state] = dict(probability=p, attempts=a, yards=yards)
        denom = np.maximum(
            components["primary"]["probability"] + components["reserve"]["probability"], 1
        )
        for value in components.values():
            value["probability"] /= denom
        conditional = sum(v["probability"] * v["yards"] for v in components.values())
        rates = dict(
            prior_season=priors,
            current_pace=current,
            career_reference=career,
            adaptive_ridge=ridge.predict(z),
            direct_boost=direct.predict(z),
            conditional_boost=conditional,
        )
    # Constraints represent information, applied identically to every candidate.
    forecasts = {m: constrain_totals(values * n, test) for m, values in rates.items()}
    return forecasts, components


def choose_recipe(earlier, horizon, origin_type, models, default):
    rows = [
        r
        for r in earlier
        if r["horizon"] == horizon and (r["through_week"] == 0) == (origin_type == "preseason")
    ]
    years = sorted({r["season"] for r in rows})
    if len(years) < 3:
        return default
    return min(
        models,
        key=lambda m: np.mean(
            [np.mean([(r[m] - r["actual"]) ** 2 for r in rows if r["season"] == y]) for y in years]
        ),
    )


def residual_range(earlier, row, prediction):
    rows = [
        r
        for r in earlier
        if r["horizon"] == row["horizon"] and r["role_group"] == row["role_group"]
    ]
    if len(rows) < 100:
        rows = [r for r in earlier if r["horizon"] == row["horizon"]]
    if len(rows) < 100:
        return None, None, 0
    # Normalize by forecast horizon length, then restore this row's schedule.
    residuals = [(r["actual"] - r["reference"]) / r["scheduled_games"] for r in rows]
    low, high = np.quantile(residuals, [0.1, 0.9]) * row["scheduled_games"] + prediction
    return float(max(0, low)), float(max(0, high)), len(rows)


def summarize(rows, model, comparator="reference"):
    annual = []
    for year in sorted({r["season"] for r in rows}):
        subset = [r for r in rows if r["season"] == year]
        y = np.array([r["actual"] for r in subset])
        p = np.array([r[model] for r in subset])
        b = np.array([r[comparator] for r in subset])
        annual.append(
            dict(
                season=year,
                n=len(subset),
                mse=float(np.mean((p - y) ** 2)),
                mae=float(np.mean(np.abs(p - y))),
                bias=float(np.mean(p - y)),
                baseline_mse=float(np.mean((b - y) ** 2)),
                mse_gain=float(np.mean((b - y) ** 2 - (p - y) ** 2)),
            )
        )
    if not annual:
        return dict(n=0, years=0, annual=[])
    gains = np.array([r["mse_gain"] for r in annual])
    boot = np.random.default_rng(20260924).choice(gains, (5000, len(gains))).mean(axis=1)
    return dict(
        model=model,
        comparator=comparator,
        n=len(rows),
        years=len(annual),
        annual=annual,
        mse=float(np.mean([r["mse"] for r in annual])),
        rmse=float(np.sqrt(np.mean([r["mse"] for r in annual]))),
        mae=float(np.mean([r["mae"] for r in annual])),
        bias=float(np.mean([r["bias"] for r in annual])),
        mse_gain=float(gains.mean()),
        ci_low=float(np.quantile(boot, 0.025)),
        ci_high=float(np.quantile(boot, 0.975)),
        positive_years=int(sum(gains > 0)),
    )
