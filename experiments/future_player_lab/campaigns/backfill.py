"""Automatic backfill: point-in-time forecasts for earlier weeks of the current season.

A campaign published mid-season only has forecasts from its first live origin, so
league pages show no predictions for earlier weeks. Any campaign whose config
declares a ``tracker.backfill`` arm gets them filled in automatically: for each
missing origin, a derived snapshot manifest is written with
``current_observations.through_week`` set to that origin (same sealed asset files;
every loader already truncates current-season box scores at ``through_week``) and the
campaign's own forecast function publishes it as a retrospective forecast.

    python -m experiments.future_player_lab.campaigns.backfill [--campaign ID] [--limit N]

These forecasts are reconstructed: box-score inputs are as of the cutoff, but the
models were fitted before the season and injury reports or market lines are the
recorded ones, so they can be slightly better informed than a live forecast was.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .weekly_v3.data import ROOT

DERIVED = ROOT / "data/research/backfill_snapshots"


def as_of(snapshot: Path, week: int, root: Path = DERIVED) -> Path:
    """A manifest identical to ``snapshot`` except that observations end at ``week``."""
    snapshot = Path(snapshot)
    document = json.loads(snapshot.read_text())
    current = document["current_observations"]
    if week > current["through_week"]:
        raise ValueError("A backfill origin cannot be later than its source snapshot")
    derived = dict(
        document,
        current_observations=dict(current, through_week=week),
        derived_from=str(snapshot.relative_to(ROOT) if snapshot.is_relative_to(ROOT) else snapshot),
        retrospective=True,
    )
    path = Path(root) / f"{snapshot.parent.name}_asof_w{week:02d}" / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(derived, indent=2, sort_keys=True) + "\n"
    if not path.exists() or path.read_text() != text:
        path.write_text(text)
    return path


def missing_origins(registry: Path, campaign_id: str, season: int, through: int) -> list[int]:
    """Origins 0..through-1 lacking a weekly forecast in any of the campaign's weekly models."""
    path = Path(registry) / campaign_id / "manifest.json"
    if not path.exists():
        return []
    weekly = [m for m in json.loads(path.read_text())["models"] if m["horizon"] == "weekly"]
    if not weekly:
        return []
    return [
        origin
        for origin in range(through)
        if not all(
            any((f["season"], f["origin_week"]) == (season, origin) for f in m["forecasts"])
            for m in weekly
        )
    ]


def run(campaign_id, forecast, *, snapshot, registry, limit=None):
    """Publish missing origins newest first via ``forecast(derived_snapshot)``.

    ``limit`` bounds the work per call so a periodic monitor fills history gradually.
    """
    current = json.loads(Path(snapshot).read_text())["current_observations"]
    season, through = current["season"], current["through_week"]
    missing = missing_origins(registry, campaign_id, season, through)
    todo = sorted(missing, reverse=True)[: limit if limit is not None else None]
    done = []
    for origin in todo:
        result = forecast(as_of(snapshot, origin))
        done.append(dict(origin_week=origin, result=result))
    return dict(
        status="backfilled" if done else "complete",
        season=season,
        published=done,
        remaining=sorted(set(missing) - set(todo)),
    )


def main(argv=None):
    from .rescore import REGISTRY, latest_raw_snapshot, run_trackers

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", action="append", help="Limit to these campaign ids")
    parser.add_argument("--snapshot", type=Path, help="Accepted snapshot; default latest")
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    parser.add_argument("--limit", type=int, help="Origins per campaign; default all")
    args = parser.parse_args(argv)
    context = dict(
        snapshot=args.snapshot or latest_raw_snapshot(),
        registry=args.registry,
        allow_network=False,
        limit=args.limit,
    )
    result = run_trackers("backfill", context, filters=args.campaign)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
