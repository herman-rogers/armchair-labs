"""Append-only weekly forecast captures and honest pregame accuracy accounting."""

import json
import os
import tempfile
from datetime import UTC, datetime
from hashlib import sha256
from math import isfinite
from pathlib import Path

from engine.espn.observations import weekly_matchups


def timestamp(value):
    if not value:
        return None
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return result if result.tzinfo else None


def record_forecast(directory: Path, snapshot, forecast):
    record = _record_forecast(directory, snapshot, forecast)
    from engine.tables.application import active
    from engine.tables.league import query_documents

    if active():
        root = directory / str(snapshot.league_id) / str(snapshot.season) / ".query-records"
        return query_documents(root, "league_forecast", [record])[0]
    return record


def _record_forecast(directory: Path, snapshot, forecast):
    root = directory / str(snapshot.league_id) / str(snapshot.season)
    root.mkdir(parents=True, exist_ok=True)
    # Do not backdate a forecast to the source release or the observed lineup capture.
    path = root / (sha256(json.dumps(forecast, sort_keys=True).encode()).hexdigest() + ".json")
    if path.exists():
        return json.loads(path.read_text())
    now = datetime.now(UTC).isoformat()
    record = dict(forecast, generated_at=now, league_id=snapshot.league_id)
    with tempfile.NamedTemporaryFile(mode="w", dir=root, suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(record, stream, allow_nan=False)
    try:
        os.link(temporary, path)  # Exclusive publication of a completely written capture.
    except FileExistsError:
        return json.loads(path.read_text())
    finally:
        temporary.unlink()
    return record


def accuracy(snapshot, directory: Path):
    from engine.tables.application import active
    from engine.tables.league import query_documents

    captured = []
    root = directory / str(snapshot.league_id) / str(snapshot.season)
    for path in root.glob("*.json"):
        try:
            captured.append(json.loads(path.read_text()))
        except (ValueError, TypeError):
            continue
    if active():
        captured = query_documents(root / ".query-history", "league_forecasts", captured)
    records = []
    for item in captured:
        try:
            deadline = timestamp(item.get("pregame_deadline"))
            generated = timestamp(item.get("generated_at"))
            captured = timestamp(item.get("captured_at"))
            published = timestamp(item.get("published_at"))
            if (
                item.get("league_id") == snapshot.league_id
                and item.get("season") == snapshot.season
                and item.get("source") in ("nextgen_weekly_reference", "nextgen_weekly_model")
                and deadline
                and generated
                and captured
                and published
                and max(generated, captured, published) < deadline
                and item["through_week"] == item["requested_week"] - 1
            ):
                records.append(item)
        except (ValueError, KeyError, TypeError):
            continue
    rows = []
    for week in range(1, min(snapshot.week, snapshot.regular_season_weeks + 1)):
        candidates = sorted(
            (r for r in records if r["requested_week"] == week),
            key=lambda r: r["generated_at"],
            reverse=True,
        )
        for game in weekly_matchups(snapshot, week)["matchups"]:
            ids = {game["home"]["team_id"], game["away"]["team_id"]}
            saved = next(
                (
                    (r, g)
                    for r in candidates
                    for g in r["matchups"]
                    if {g["home"]["team_id"], g["away"]["team_id"]} == ids
                ),
                None,
            )
            prediction, status, correct = None, "No verified pregame forecast", None
            projected_home, projected_away, point_error = None, None, None
            if saved:
                _, prediction = saved
                h, a = game["home"]["score"], game["away"]["score"]
                settled = all(
                    any(
                        entry.week == week
                        and entry.opponent_team_id in ids
                        and entry.outcome in ("W", "L", "T")
                        for entry in team.schedule
                    )
                    for team in snapshot.teams
                    if team.team_id in ids
                )
                projected = {
                    side["team_id"]: side["model_projection"]
                    for side in (prediction["home"], prediction["away"])
                }
                projected_home = projected.get(game["home"]["team_id"])
                projected_away = projected.get(game["away"]["team_id"])
                if settled and all(
                    isinstance(v, (float, int)) and isfinite(v)
                    for v in (h, a, projected_home, projected_away)
                ):
                    point_error = (abs(h - projected_home) + abs(a - projected_away)) / 2
                if not settled:
                    status = "Result unavailable"
                elif prediction["favorite_team_id"] is None:
                    status = "No pick: tied or incomplete forecast"
                elif h is None or a is None:
                    status = "Result unavailable"
                elif h == a:
                    status = "Actual tie · excluded"
                else:
                    correct = prediction["favorite_team_id"] == (
                        game["home"]["team_id"] if h > a else game["away"]["team_id"]
                    )
                    status = "Correct" if correct else "Incorrect"
            rows.append(
                dict(
                    week=week,
                    home=game["home"]["team_name"],
                    away=game["away"]["team_name"],
                    home_id=game["home"]["team_id"],
                    away_id=game["away"]["team_id"],
                    home_score=game["home"]["score"],
                    away_score=game["away"]["score"],
                    predicted_winner=next(
                        (
                            s["team_name"]
                            for s in (prediction["home"], prediction["away"])
                            if s["team_id"] == prediction["favorite_team_id"]
                        ),
                        None,
                    )
                    if prediction
                    else None,
                    projected_home=projected_home,
                    projected_away=projected_away,
                    point_error=point_error,
                    status=status,
                    correct=correct,
                )
            )
    scored = [r for r in rows if r["correct"] is not None]
    point_errors = [r["point_error"] for r in rows if r["point_error"] is not None]
    return dict(
        point_mae=sum(point_errors) / len(point_errors) if point_errors else None,
        point_teams=2 * len(point_errors),
        season=snapshot.season,
        through_week=min(snapshot.week - 1, snapshot.regular_season_weeks),
        correct=sum(r["correct"] for r in scored),
        evaluated=len(scored),
        total=len(rows),
        accuracy=sum(r["correct"] for r in scored) / len(scored) if scored else None,
        games=rows,
        method="Latest saved forecast before 00:00 UTC on the first NFL game date of each week. "
        "No retrospective reconstructions count as pregame predictions.",
    )
