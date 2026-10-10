"""The library inventories evidence without promoting it or hiding corrupt releases."""

import json

import pytest

from engine.models.catalog import build_catalog, digest, load_campaigns, load_families
from engine.models.render import write_catalog


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def workspace(tmp_path):
    source = tmp_path / "method.py"
    source.write_text("# method\n")
    specs = tmp_path / "families.json"
    put(
        specs,
        {
            "schema_version": 1,
            "families": [
                {
                    "id": "test",
                    "name": "Test method",
                    "layer": "production",
                    "method": "ridge",
                    "targets": ["yards"],
                    "horizons": ["season"],
                    "sources": ["method.py"],
                    "assessment": "Research only",
                    "limits": "Not an ability model",
                    "registry_prefixes": ["forecast:"],
                }
            ],
        },
    )
    return tmp_path, specs


def published(root):
    """A sealed campaign bundle under data/campaigns, exactly as the API would read it."""
    from engine.chairstats import publish_campaign

    work = root / "work"
    put(work / "metrics.json", {"rmse": 1.0})
    put(
        work / "forecast.json",
        {
            "rows": [
                {
                    "player_id": "p1",
                    "position": "RB",
                    "season": 2026,
                    "origin_week": 0,
                    "predictions": {"yards": 900.0},
                }
            ]
        },
    )
    manifest = publish_campaign(
        root / "data/campaigns",
        {"campaign_id": "release", "name": "Release", "kind": "model_campaign"},
        [
            {
                "model_id": "rb_season",
                "position": "RB",
                "horizon": "season",
                "origin_week": 0,
                "targets": ["yards"],
                "training_through_season": 2025,
                "metrics": work / "metrics.json",
                "forecasts": [{"season": 2026, "origin_week": 0, "path": work / "forecast.json"}],
            }
        ],
    )
    return root / "data", {"version": "release", "manifest_sha256": digest(manifest)}


def test_missing_local_data_still_lists_methods_without_approval(workspace):
    root, specs = workspace
    result = build_catalog(root, definitions=specs)
    assert result["families"][0]["name"] == "Test method"
    assert result["current"]["status"] == "unavailable"
    assert result["current"]["campaigns"] == []
    assert not any(x["severity"] == "error" for x in result["issues"])


def test_published_bundles_are_listed_and_research_runs_are_not_promoted(workspace):
    root, specs = workspace
    published(root)
    put(root / "data/research/newer/manifest.json", {"status": "complete"})
    result = build_catalog(root, definitions=specs)
    assert result["current"]["status"] == "bundles_verified"
    assert [c["campaign_id"] for c in result["current"]["campaigns"]] == ["release"]
    newer = next(x for x in result["runs"] if x["id"] == "newer")
    assert newer["serving"] == "not_inferred"
    assert not newer["published_campaign"]
    bundle = next(x for x in result["runs"] if x["path"] == "data/campaigns/release")
    assert bundle["published_campaign"]


def test_corrupt_bundle_is_reported_not_listed(workspace):
    root, specs = workspace
    published(root)
    put(root / "data/campaigns/release/rb_season/metrics.json", {"rmse": 2.0})
    result = build_catalog(root, definitions=specs)
    bundles = build_catalog(root, definitions=specs)["current"]
    assert (
        bundles["status"] == "bundles_verified"
    )  # manifests verify; artifacts are checked on read
    (root / "data/campaigns/release/manifest.sha256").write_text("0" * 64 + "\n")
    result = build_catalog(root, definitions=specs)
    assert result["current"]["status"] == "unavailable"
    assert any("checksum mismatch" in x["message"] for x in result["issues"])


def test_unfinished_and_unindexed_runs_remain_visible(workspace):
    root, specs = workspace
    put(root / "experiments/example_lab/runs/failed/manifest.json", {"status": "failed"})
    (root / "data/research/orphan").mkdir(parents=True)
    result = build_catalog(root, definitions=specs)
    assert {x["id"]: x["status"] for x in result["runs"]} == {
        "failed": "failed",
        "orphan": "unindexed",
    }


def test_unsealed_campaign_directories_are_inventory_not_bundles(workspace):
    root, specs = workspace
    put(
        root / "data/campaigns/draft/manifest.json",
        {
            "status": "complete",
            "kind": "model_campaign",
            "models": [{"model_id": "qb_remaining"}],
        },
    )
    put(root / "data/campaigns/.campaign-staging/manifest.json", {"status": "building"})
    catalog = build_catalog(root, definitions=specs)
    row = next(row for row in catalog["runs"] if row["path"] == "data/campaigns/draft")
    assert row["kind"] == "model_campaign"
    assert row["status"] == "complete"
    assert row["published_campaign"] is False
    assert row["serving"] == "not_inferred"
    assert catalog["current"]["status"] == "unavailable"
    assert not any(".campaign-staging" in row["path"] for row in catalog["runs"])


def test_curated_approval_and_duplicate_family_are_rejected(workspace):
    _, specs = workspace
    document = json.loads(specs.read_text())
    document["families"][0]["approved"] = True
    put(specs, document)
    with pytest.raises(ValueError, match="Serving status"):
        load_families(specs)
    del document["families"][0]["approved"]
    document["families"].append(document["families"][0])
    put(specs, document)
    with pytest.raises(ValueError, match="Duplicate"):
        load_families(specs)


def test_html_escapes_embedded_script_content_and_json_roundtrips(workspace):
    root, specs = workspace
    result = build_catalog(root, definitions=specs)
    result["families"][0]["name"] = '</script><script>alert("x")</script>'
    path = write_catalog(result, root / "output")
    text = path.read_text()
    assert "</script><script>alert" not in text
    assert "\\u003c/script>" in text
    assert json.loads((path.parent / "catalog.json").read_text()) == result


def test_catalog_build_does_not_modify_existing_artifacts(workspace):
    root, specs = workspace
    published(root)
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    build_catalog(root, definitions=specs)
    assert {p: p.read_bytes() for p in root.rglob("*") if p.is_file()} == before


def test_saved_input_audit_distinguishes_retired_scores_from_provider_fields(workspace):
    root, specs = workspace
    directory = root / "experiments/example_lab/runs/saved"
    put(
        directory / "manifest.json",
        {
            "input_provenance": {
                "remaining": {
                    "names": ["prior__box__wopr", "current__age_factor", "prior__targets"]
                }
            }
        },
    )
    put(
        root / "experiments/example_lab/registry/model.json",
        {
            "id": "example",
            "model_directory": "experiments/example_lab/runs/saved",
            "model_manifest_sha256": digest(directory / "manifest.json"),
        },
    )
    result = build_catalog(root, definitions=specs)
    audit = result["configurations"][0]["contract"]["input_audit"]["remaining"]
    assert audit["input_count"] == 3
    assert audit["retired_input_matches"] == ["current__age_factor"]


def test_campaign_provenance_labels_do_not_create_queryable_bundles(workspace):
    from engine.chairstats import CampaignStore

    root, families = workspace
    directory = root / "experiments/example_lab/runs/original"
    put(directory / "manifest.json", {"status": "complete"})
    contract = "experiments/example_lab/registry/model.json"
    put(
        root / contract,
        {
            "id": "example",
            "campaign_id": "old_study",
            "campaign_name": "Study",
            "model_directory": "experiments/example_lab/runs/original",
            "model_manifest_sha256": digest(directory / "manifest.json"),
        },
    )
    definitions = root / "campaigns.json"
    put(
        definitions,
        {
            "schema_version": 1,
            "campaigns": [
                {
                    "campaign_id": "old_study",
                    "campaign_name": "Study",
                    "family_ids": ["test"],
                    "run_patterns": ["experiments/example_lab/runs/original"],
                    "config_paths": [contract],
                    "saved_model_manifest": "experiments/example_lab/runs/original/manifest.json",
                    "saved_model_manifest_sha256": digest(directory / "manifest.json"),
                    "api_bundle": None,
                }
            ],
        },
    )
    before = (directory / "manifest.json").read_bytes()
    catalog = build_catalog(root, definitions=families, campaign_definitions=definitions)
    for row in (catalog["families"][0], catalog["runs"][0], catalog["configurations"][0]):
        assert row["campaigns"][0]["campaign_id"] == "old_study"
    assert catalog["configurations"][0]["contract"]["campaign_name"] == "Study"
    assert catalog["campaigns"][0]["manifest_state"] == "manifest_matches"
    assert CampaignStore(root / "data/campaigns").campaigns() == []
    assert (directory / "manifest.json").read_bytes() == before
    document = json.loads(definitions.read_text())
    document["campaigns"][0]["query_ready"] = True
    put(definitions, document)
    with pytest.raises(ValueError, match="cannot declare"):
        load_campaigns(definitions)


def test_source_study_inherits_parent_without_reclassifying_unrelated_release(workspace):
    root, families = workspace
    put(root / "experiments/example_lab/runs/study/manifest.json", {"status": "complete"})
    put(root / "data/research/mixed_release/manifest.json", {"status": "complete"})
    definitions = root / "campaigns.json"
    document = {
        "schema_version": 1,
        "campaigns": [
            {
                "campaign_id": "parent",
                "campaign_name": "Parent",
                "family_ids": ["test"],
                "saved_model_manifest": "data/campaigns/parent/manifest.json",
                "source_artifact_paths": ["data/research/mixed_release/manifest.json"],
            },
            {
                "campaign_id": "study",
                "campaign_name": "Study",
                "entity_kind": "source_study",
                "parent_campaign_id": "parent",
                "family_ids": ["test"],
                "run_patterns": ["experiments/example_lab/runs/study"],
                "saved_model_manifest": "experiments/example_lab/runs/study/manifest.json",
            },
        ],
    }
    put(definitions, document)
    catalog = build_catalog(root, definitions=families, campaign_definitions=definitions)
    study = next(row for row in catalog["runs"] if row["id"] == "study")
    assert {tag["campaign_id"] for tag in study["campaigns"]} == {"study", "parent"}
    assert study["campaigns"][0]["parent_campaign_name"] == "Parent"
    assert len(catalog["families"][0]["campaigns"]) == 2
    assert "campaigns" not in next(row for row in catalog["runs"] if row["id"] == "mixed_release")
    document["campaigns"][1]["parent_campaign_id"] = "missing"
    put(definitions, document)
    with pytest.raises(ValueError, match="existing parent"):
        load_campaigns(definitions)
