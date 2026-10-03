"""A normal data refresh must not quietly remove the published ranking product."""

import json
from types import SimpleNamespace

import pytest

from engine.data import catalog, nextgen
from engine.data.releases import reference, write_json


@pytest.mark.parametrize("ranking_schema", [None, 1])
@pytest.mark.parametrize(
    "schema_key,message",
    [
        ("ranking_schema_version", "cannot silently drop its rankings"),
        ("qb_passing_schema_version", "cannot silently drop its QB passing forecasts"),
        ("qb_variations_schema_version", "cannot silently drop its QB variation policies"),
    ],
)
def test_ranking_publication_continuity(tmp_path, monkeypatch, ranking_schema, schema_key, message):
    from engine.api import college_sources, outlook_sources, profile_sources

    gold = SimpleNamespace(
        ref={"version": "gold", "manifest_sha256": "gold-hash"},
        manifest={"history": {}, "current_observations": {"season": 2026, "through_week": 2}},
    )
    products = {"college": "c", "profiles": "p", "outlook_2026": "o", "analysis": "a"}
    for version in [*products.values(), "prior"]:
        root = tmp_path / "research" / version
        root.mkdir(parents=True)
        write_json(root / "manifest.json", dict(gold=gold.ref, history={}, **{schema_key: 1}))
    prior_ref = reference(tmp_path / "research/prior")
    write_json(
        tmp_path / "current.json",
        {"schema_version": 1, "gold": gold.ref, "products": {"analysis": prior_ref}},
    )
    before = (tmp_path / "current.json").read_bytes()
    monkeypatch.setattr(catalog, "load_gold", lambda *_: gold)
    monkeypatch.setattr(college_sources, "load_college", lambda **_: (None, {"gold": gold.ref}))
    monkeypatch.setattr(
        outlook_sources,
        "load_outlook_release",
        lambda *_, **__: (None, {"gold": gold.ref, "through_week": 2}),
    )
    monkeypatch.setattr(
        profile_sources,
        "load_profiles",
        lambda **_: (
            None,
            dict(gold=gold.ref, through_week=2, college_version="c", outlook_version="o"),
        ),
    )
    monkeypatch.setattr(
        nextgen,
        "load_analysis",
        lambda *_: (
            None,
            dict(
                gold=gold.ref,
                profiles=reference(tmp_path / "research/p"),
                **{schema_key: ranking_schema},
            ),
        ),
    )
    if ranking_schema is None:
        with pytest.raises(ValueError, match=message):
            catalog.publish_catalog(tmp_path, "gold", products)
        assert (tmp_path / "current.json").read_bytes() == before
    else:
        catalog.publish_catalog(tmp_path, "gold", products)
        published = json.loads((tmp_path / "current.json").read_text())
        assert published["products"]["analysis"]["version"] == "a"
        assert list((tmp_path / "catalog_history").glob("*.json"))[0].read_bytes() == before
