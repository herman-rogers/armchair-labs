"""Prebuild application responses without modifying analytical releases.

Run after publishing data/analysis. Current candidates get latest profiles and
comparisons plus completed-season profiles; other cutoffs use bounded lazy caching.
"""

import argparse
import fcntl
import inspect
import json
import os
import tempfile
from pathlib import Path

import polars as pl
from fastapi.params import Param

from engine.api import nextgen_routes, profile_routes, ranking_routes
from engine.config.settings import get_settings
from engine.data.frames import read_frame
from engine.data.releases import atomic_json, digest
from engine.data.serving import SCHEMA_VERSION, encode, generation, model_key
from engine.data.verification import verification_batch


def build(*, history=True, player_limit=None):
    """Precompute the same DuckDB-backed responses served by application requests."""
    from engine.tables.application import query_scope

    with query_scope(get_settings().data_dir):
        return _build(history=history, player_limit=player_limit)


def _build(*, history=True, player_limit=None):
    data = get_settings().data_dir
    current = generation(data)
    if current is None:
        raise ValueError("Publish a verified analysis catalog before building serving models")
    build_id, binding = current
    parent = data / "serving/releases"
    parent.mkdir(parents=True, exist_ok=True)
    models = {}
    with tempfile.TemporaryDirectory(prefix="serving-build-") as temporary:
        staging = Path(temporary)

        def save(handler, **params):
            bound = inspect.signature(handler).bind(**params)
            bound.apply_defaults()
            arguments = {
                k: v.default if isinstance(v, Param) else v for k, v in bound.arguments.items()
            }
            key = model_key(handler.read_model_name, arguments)
            if key in models:
                return
            value = handler.__wrapped__(**arguments)
            path = staging / (key + ".json")
            path.write_bytes(encode(value))
            models[key] = digest(path)

        with verification_batch():
            directory = profile_routes.directory.__wrapped__()
            root, _ = profile_routes.load_profiles()
            weeks = read_frame(root / "nfl_weeks.parquet")
            save(profile_routes.directory)
            for horizon in ("rest_of_season", "next4"):
                # One response covers the small current population, shared by all views.
                save(ranking_routes.rankings, horizon=horizon, limit=1000, offset=0)
            for period in ("prior", "recent3", "career", "current"):
                save(nextgen_routes.players, period=period, limit=1000, offset=0)
            players = [p for p in directory["players"] if p["current_candidate"]]
            if player_limit is not None:
                players = players[:player_limit]
            for i, player in enumerate(players):
                pid = player["player_id"]
                save(profile_routes.profile, player_id=pid)
                save(profile_routes.similar_players, player_id=pid)
                if history:
                    seasons = weeks.filter(pl.col("player_id") == pid)["season"].unique()
                    for year in seasons:
                        if year < directory["report"]["season"]:
                            save(profile_routes.profile, player_id=pid, season=year, week=18)
                if (i + 1) % 50 == 0 or i + 1 == len(players):
                    print(
                        f"Built {i + 1}/{len(players)} players; {len(models)} responses", flush=True
                    )
        if generation(data) != current:
            raise ValueError("Catalog or implementation changed during build; nothing published")
        manifest = dict(
            schema_version=SCHEMA_VERSION, generation=build_id, binding=binding, models=models
        )
        (staging / "manifest.json").write_bytes(encode(manifest))
        bundle = digest(staging / "manifest.json")
        destination = parent / bundle
        # Complete artifacts are copied on the destination filesystem before atomic rename.
        import shutil

        with tempfile.TemporaryDirectory(prefix=".staging-", dir=parent) as local:
            ready = Path(local) / build_id
            shutil.copytree(staging, ready)
            if destination.exists():
                # Rebuilding a generation is idempotent, never silently replaces a different bundle.
                if digest(destination / "manifest.json") != digest(ready / "manifest.json"):
                    raise ValueError("Generation already exists with different build options")
                for key, expected in models.items():
                    if digest(destination / (key + ".json")) != expected:
                        raise ValueError("Existing serving artifact changed")
            else:
                os.rename(ready, destination)
        (data / ".runtime").mkdir(exist_ok=True)
        with (data / ".runtime/catalog_publish.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if generation(data) != current:
                raise ValueError("Catalog changed before serving publication")
            atomic_json(
                data / "serving/current.json",
                dict(generation=build_id, version=bundle, manifest_sha256=bundle),
            )
    return dict(generation=build_id, responses=len(models), path=str(destination))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-history", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build(history=not args.no_history)))
