"""Unchanged ESPN polls must not publish new league releases or native caches."""

import dataclasses
import json

from engine.tables.league import query_snapshot
from tests.test_league_observations import snapshot


def releases(data):
    root = data / "tables/releases"
    return {table.name: sorted(p.name for p in table.iterdir()) for table in root.iterdir()}


def state(data_dir):
    data = data_dir / "league/1/2026/data"
    return (
        releases(data),
        sorted(p.name for p in (data / "cache/tables").glob("*.duckdb")),
        json.loads((data / "tables/current.json").read_text()),
    )


def test_unchanged_poll_reuses_releases_and_cache(tmp_path):
    first = snapshot()
    assert query_snapshot(first, tmp_path) == first
    before = state(tmp_path)

    later = dataclasses.replace(first, captured_at="2026-09-23T00:05:00+00:00")
    queried = query_snapshot(later, tmp_path)

    assert queried == later
    assert queried.captured_at == later.captured_at
    assert state(tmp_path) == before


def test_changed_table_alone_gets_a_new_release(tmp_path):
    first = snapshot()
    query_snapshot(first, tmp_path)
    before, caches, catalog = state(tmp_path)

    team = dataclasses.replace(first.teams[0], wins=first.teams[0].wins + 1)
    changed = dataclasses.replace(
        first,
        captured_at="2026-09-23T00:05:00+00:00",
        teams=[team, *first.teams[1:]],
    )
    assert query_snapshot(changed, tmp_path) == changed
    after, new_caches, new_catalog = state(tmp_path)

    rebuilt = {name for name in after if after[name] != before[name]}
    assert rebuilt == {"league_teams"}
    assert len(after["league_teams"]) == 2
    assert new_catalog != catalog
    assert len(new_caches) == len(caches) + 1
