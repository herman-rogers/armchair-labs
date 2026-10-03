"""Conditional-on-recorded-lineups replay, kept separate from prospective accuracy."""

from dataclasses import replace

from engine.espn.observations import weekly_matchups
from engine.metrics.weekly_lineup import weekly_forecast


def season_replay(snapshot, product, identities, schedule):
    manifest = product["manifest"]
    through = min(manifest["through_week"], snapshot.week - 1, snapshot.regular_season_weeks)
    rows = []
    if manifest["season"] != snapshot.season:
        return dict(games=[], evaluated=0, correct=0, total=0, accuracy=None, point_mae=None)
    for week in range(1, through + 1):
        # Model inputs carry the actual per-row origin. Publication time is NEVER
        # backdated: these are reconstructed forecasts generated today.
        predictions = [
            r
            for r in product["predictions"]
            if r["target_week"] == week
            and r["through_week"] == week - 1
            and r["season"] == snapshot.season
        ]
        scoped = dict(
            product, manifest=dict(manifest, through_week=week - 1), predictions=predictions
        )
        ranking_stub = dict(
            version=manifest["analysis"]["version"],
            horizon="next4",
            rankings=[],
            report=dict(
                season=snapshot.season, through_week=week - 1, generated_at=manifest["generated_at"]
            ),
        )
        replay = weekly_forecast(
            replace(snapshot, week=week), ranking_stub, identities, schedule, weekly=scoped
        )
        actual = weekly_matchups(snapshot, week)
        for game in replay["matchups"]:
            ids = {game["home"]["team_id"], game["away"]["team_id"]}
            observed = next(
                (
                    g
                    for g in actual["matchups"]
                    if {g["home"]["team_id"], g["away"]["team_id"]} == ids
                ),
                None,
            )
            scores = (
                {s["team_id"]: s["score"] for s in (observed["home"], observed["away"])}
                if observed
                else {}
            )
            home, away = game["home"], game["away"]
            h, a = scores.get(home["team_id"]), scores.get(away["team_id"])
            hp, ap = home["model_projection"], away["model_projection"]
            complete = all(v is not None for v in (h, a, hp, ap))
            correct = (
                (game["favorite_team_id"] == (home["team_id"] if h > a else away["team_id"]))
                if complete and h != a and game["favorite_team_id"] is not None
                else None
            )
            rows.append(
                dict(
                    week=week,
                    home=home["team_name"],
                    away=away["team_name"],
                    home_id=home["team_id"],
                    away_id=away["team_id"],
                    home_score=h,
                    away_score=a,
                    projected_home=hp,
                    projected_away=ap,
                    correct=correct,
                    predicted_winner=next(
                        (
                            s["team_name"]
                            for s in (home, away)
                            if s["team_id"] == game["favorite_team_id"]
                        ),
                        None,
                    ),
                    point_error=(abs(h - hp) + abs(a - ap)) / 2 if complete else None,
                    status=("Correct" if correct else "Incorrect")
                    if correct is not None
                    else "Incomplete coverage or tied game",
                )
            )
    scored = [r for r in rows if r["correct"] is not None]
    errors = [r["point_error"] for r in rows if r["point_error"] is not None]
    return dict(
        games=rows,
        total=len(rows),
        evaluated=len(scored),
        correct=sum(r["correct"] for r in scored),
        accuracy=sum(r["correct"] for r in scored) / len(scored) if scored else None,
        point_mae=sum(errors) / len(errors) if errors else None,
        method="Retrospective replay using prior-week features and models fitted/selected "
        "before this season. Conditional on recorded actual starting lineups. "
        "K/DST use prior-score references. Not a record of pregame published picks.",
    )
