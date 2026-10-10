"""Manifest-bound, read-only access to saved campaign outputs.

The API reads JSON predictions produced offline. Saved weights are inventoried and
verified by ``check`` but are never deserialized in a request. Every published
bundle is queryable; nothing here selects one over another.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from engine.data.verification import observe, observe_listing, verification_batch, verified

ID = re.compile(r"[a-z0-9][a-z0-9_-]{0,79}\Z")
SHA = re.compile(r"[0-9a-f]{64}\Z")


class CampaignNotFound(LookupError):
    """The requested campaign, model, or forecast does not exist."""


def safe_id(value: str) -> str:
    if not isinstance(value, str) or not ID.fullmatch(value):
        raise ValueError("Invalid campaign or model identifier")
    return value


def _digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _json(content: bytes):
    def invalid(value):
        raise ValueError(f"Non-finite JSON value: {value}")

    return json.loads(content, parse_constant=invalid)


def _inside(root: Path, name: str) -> Path:
    if not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts:
        raise ValueError("Artifact paths must be relative to their campaign")
    path = root / name
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Artifact path escapes its campaign")
    return path


def _reference(root: Path, ref: dict) -> Path:
    if not isinstance(ref, dict) or not SHA.fullmatch(str(ref.get("sha256", ""))):
        raise ValueError("Artifact reference requires a SHA-256")
    return _inside(root, ref["path"])


def _validate(root: Path, manifest: dict, campaign_id: str) -> None:
    if manifest.get("schema_version") != 1 or manifest.get("campaign_id") != campaign_id:
        raise ValueError("Campaign manifest identity or schema mismatch")
    if manifest.get("disposition") != "research_preview":
        raise ValueError("Campaign access currently supports research previews only")
    if not isinstance(manifest.get("name"), str) or not isinstance(manifest.get("models"), list):
        raise ValueError("Campaign requires a name and model list")
    ids = set()
    for model in manifest["models"]:
        model_id = safe_id(model["model_id"])
        if model_id in ids:
            raise ValueError("Duplicate model identifier")
        ids.add(model_id)
        if model.get("position") is not None and (
            not isinstance(model["position"], str) or not model["position"]
        ):
            raise ValueError("Model position must be a nonempty string when present")
        targets = model.get("targets")
        if (
            not isinstance(model.get("horizon"), str)
            or not model["horizon"]
            or not isinstance(targets, list)
            or not targets
            or any(not isinstance(target, str) or not target for target in targets)
            or len(set(targets)) != len(targets)
        ):
            raise ValueError("Models require explicit horizons and targets")
        if type(model.get("origin_week")) is not int or not 0 <= model["origin_week"] <= 18:
            raise ValueError("Models require an explicit observation cutoff")
        if not isinstance(model.get("training_through_season"), int):
            raise ValueError("Models require a training cutoff")
        _reference(root, model["metrics"])
        for ref in model.get("artifacts", {}).values():
            _reference(root, ref)
        snapshots = set()
        for forecast in model.get("forecasts", []):
            key = forecast["season"], forecast["origin_week"]
            if key in snapshots:
                raise ValueError("Duplicate forecast season/origin")
            snapshots.add(key)
            if not all(isinstance(value, int) for value in key) or not 0 <= key[1] <= 18:
                raise ValueError("Invalid forecast season/origin")
            if key[0] <= model["training_through_season"]:
                raise ValueError("Forecast must follow the final model training cutoff")
            if key[1] != model.get("origin_week"):
                raise ValueError("Forecast origin differs from model contract")
            _reference(root, forecast)


def _forecast_rows(payload, model: dict, season: int, origin_week: int) -> list[dict]:
    if isinstance(payload, dict):
        for key, expected in (
            ("model_id", model["model_id"]),
            ("season", season),
            ("origin_week", origin_week),
            ("horizon", model["horizon"]),
            ("training_through_season", model["training_through_season"]),
        ):
            if key in payload and payload[key] != expected:
                raise ValueError("Forecast receipt differs from its model contract")
    rows = payload if isinstance(payload, list) else payload["rows"]
    if not isinstance(rows, list):
        raise ValueError("Forecast rows must be a list")
    players = set()
    for row in rows:
        player_id = row["player_id"]
        if not player_id or player_id in players:
            raise ValueError("Forecast player identifiers must be unique and nonempty")
        players.add(player_id)
        for key, expected in (
            ("position", model.get("position")),
            ("season", season),
            ("origin_week", origin_week),
        ):
            if expected is not None and row.get(key) != expected:
                raise ValueError("Forecast row identity differs from its model contract")
        if not isinstance(row.get("predictions"), dict):
            raise ValueError("Forecast rows require a predictions mapping")
        if set(row["predictions"]) != set(model["targets"]):
            raise ValueError("Forecast targets differ from their model contract")
        if any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            for value in row["predictions"].values()
        ):
            raise ValueError("Predictions must be finite numeric values")
    return rows


class CampaignStore:
    def __init__(self, root: Path):
        self.root = Path(root)

    def campaign(self, campaign_id: str) -> dict[str, Any]:
        root = _inside(self.root, safe_id(campaign_id))
        path = root / "manifest.json"
        if not path.exists():
            raise CampaignNotFound(f"Unknown campaign: {campaign_id}")

        def read():
            observe(path)
            observe(root / "manifest.sha256")
            content = path.read_bytes()
            expected = (root / "manifest.sha256").read_text().strip()
            if _digest(content) != expected:
                raise ValueError("Campaign manifest checksum mismatch")
            manifest = _json(content)
            _validate(root, manifest, campaign_id)
            return {**manifest, "manifest_sha256": expected}

        return verified(("campaign-manifest-v1", str(path.absolute())), read)

    def campaigns(self) -> list[dict]:
        def read():
            observe_listing(self.root)
            if not self.root.exists():
                return []
            return [
                self.campaign(path.name)
                for path in sorted(self.root.iterdir())
                if path.is_dir() and not path.name.startswith(".")
            ]

        return verified(("campaign-directory-v1", str(self.root.absolute())), read)

    def model(self, campaign_id: str, model_id: str) -> dict:
        safe_id(model_id)
        model = next(
            (m for m in self.campaign(campaign_id)["models"] if m["model_id"] == model_id), None
        )
        if model is None:
            raise CampaignNotFound(f"Unknown model in campaign {campaign_id}: {model_id}")
        return model

    def artifact(self, campaign_id: str, ref: dict):
        root = _inside(self.root, safe_id(campaign_id))
        path = _reference(root, ref)

        def read():
            observe(path)
            content = path.read_bytes()
            if _digest(content) != ref["sha256"]:
                raise ValueError("Campaign artifact checksum mismatch")
            return _json(content)

        return verified(("campaign-json-v1", str(path.absolute()), ref["sha256"]), read)

    def metrics(self, campaign_id: str, model_id: str):
        return self.artifact(campaign_id, self.model(campaign_id, model_id)["metrics"])

    def forecast(self, campaign_id: str, model_id: str, season: int, origin_week: int):
        model = self.model(campaign_id, model_id)
        ref = next(
            (
                f
                for f in model.get("forecasts", [])
                if (f["season"], f["origin_week"]) == (season, origin_week)
            ),
            None,
        )
        if ref is None:
            raise CampaignNotFound("No saved forecast for that season and origin")
        payload = self.artifact(campaign_id, ref)
        if isinstance(payload, dict) and payload.get("campaign_id", campaign_id) != campaign_id:
            raise ValueError("Forecast receipt belongs to another campaign")
        rows = _forecast_rows(payload, model, season, origin_week)
        metadata = (
            {key: value for key, value in payload.items() if key != "rows"}
            if isinstance(payload, dict)
            else {}
        )
        return {"rows": rows, "metadata": metadata, "artifact_sha256": ref["sha256"]}

    def check(self, campaign_id: str) -> dict:
        """Verify every JSON output and every saved binary without loading any model."""
        with verification_batch():
            manifest = self.campaign(campaign_id)
            count = 0
            root = _inside(self.root, campaign_id)
            for model in manifest["models"]:
                self.metrics(campaign_id, model["model_id"])
                count += 1
                for ref in model.get("forecasts", []):
                    self.forecast(campaign_id, model["model_id"], ref["season"], ref["origin_week"])
                    count += 1
                for ref in model.get("artifacts", {}).values():
                    path = _reference(root, ref)

                    def audit(path=path, ref=ref):
                        observe(path)
                        if _digest(path.read_bytes()) != ref["sha256"]:
                            raise ValueError("Saved model artifact checksum mismatch")
                        return True

                    verified(("campaign-binary-v1", str(path.absolute()), ref["sha256"]), audit)
                    count += 1
        return {
            "campaign_id": campaign_id,
            "verified_artifacts": count,
            "manifest_sha256": manifest["manifest_sha256"],
            "status": "verified",
        }


def publish_campaign(root: Path, metadata: dict, models: list[dict]) -> Path:
    """Create a sealed, create-only bundle from offline training outputs.

    ``metadata`` includes campaign_id/name and descriptive JSON. Each model has
    model_id/position/horizon/origin_week/targets/training_through_season,
    ``metrics: Path``, ``forecasts: [{season, origin_week, path: Path}]``, and
    optionally ``artifacts: {logical_name: Path}``. Source files are copied so
    independent cleanup of research runs cannot break published campaign access.
    """
    root = Path(root)
    campaign_id = safe_id(metadata["campaign_id"])
    root.mkdir(parents=True, exist_ok=True)
    destination = root / campaign_id
    if destination.exists():
        raise FileExistsError(f"Campaign already exists: {campaign_id}")
    staging = Path(tempfile.mkdtemp(prefix=".campaign-", dir=root))
    try:
        entries = []

        def copy(source: Path, name: str):
            target = _inside(staging, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            return {"path": name, "sha256": _digest(target.read_bytes())}

        for model in models:
            entry = dict(model)
            model_id = safe_id(entry["model_id"])
            entry["metrics"] = copy(Path(model["metrics"]), f"{model_id}/metrics.json")
            entry["forecasts"] = [
                {
                    "season": f["season"],
                    "origin_week": f["origin_week"],
                    **copy(
                        Path(f["path"]),
                        f"{model_id}/forecasts/{f['season']}_{f['origin_week']}.json",
                    ),
                }
                for f in model.get("forecasts", [])
            ]
            entry["artifacts"] = {
                safe_id(name): copy(
                    Path(source), f"{model_id}/artifacts/{name}{Path(source).suffix}"
                )
                for name, source in model.get("artifacts", {}).items()
            }
            entries.append(entry)
        manifest = {
            **metadata,
            "schema_version": 1,
            "disposition": "research_preview",
            "models": entries,
        }
        _validate(staging, manifest, campaign_id)
        content = (json.dumps(manifest, indent=2, allow_nan=False) + "\n").encode()
        (staging / "manifest.json").write_bytes(content)
        (staging / "manifest.sha256").write_text(_digest(content) + "\n")
        # Validate rows before exposing the directory, including target/cutoff contracts.
        for entry in entries:
            _json((staging / entry["metrics"]["path"]).read_bytes())
            for ref in entry["forecasts"]:
                payload = _json((staging / ref["path"]).read_bytes())
                if (
                    isinstance(payload, dict)
                    and payload.get("campaign_id", campaign_id) != campaign_id
                ):
                    raise ValueError("Forecast receipt belongs to another campaign")
                _forecast_rows(payload, entry, ref["season"], ref["origin_week"])
        if destination.exists():
            raise FileExistsError(f"Campaign already exists: {campaign_id}")
        staging.rename(destination)
        return destination / "manifest.json"
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _verify_bundle(directory: Path, campaign_id: str, expected_sha: str | None = None) -> dict:
    """Verify a bundle at an arbitrary archive/staging path without deserializing it."""
    content = (directory / "manifest.json").read_bytes()
    sha = _digest(content)
    if sha != (directory / "manifest.sha256").read_text().strip():
        raise ValueError("Campaign manifest checksum mismatch")
    if expected_sha is not None and sha != expected_sha:
        raise ValueError("Campaign replacement manifest pin is stale")
    manifest = _json(content)
    _validate(directory, manifest, campaign_id)
    for model in manifest["models"]:
        json_paths = {model["metrics"]["path"], *(f["path"] for f in model.get("forecasts", []))}
        json_content = {}
        references = [
            model["metrics"],
            *model.get("artifacts", {}).values(),
            *model.get("forecasts", []),
        ]
        for ref in references:
            content = _reference(directory, ref).read_bytes()
            if _digest(content) != ref["sha256"]:
                raise ValueError("Campaign replacement artifact checksum mismatch")
            if ref["path"] in json_paths:
                json_content[ref["path"]] = content
        _json(json_content[model["metrics"]["path"]])
        for forecast in model.get("forecasts", []):
            payload = _json(json_content[forecast["path"]])
            if isinstance(payload, dict) and payload.get("campaign_id", campaign_id) != campaign_id:
                raise ValueError("Forecast receipt belongs to another campaign")
            _forecast_rows(payload, model, forecast["season"], forecast["origin_week"])
    return {"manifest_sha256": sha, "models": len(manifest["models"])}


def _exchange_directories(left: Path, right: Path) -> None:
    """Linux atomic exchange: readers see either complete directory, never a gap."""
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    try:
        rename = libc.renameat2
    except AttributeError as exc:
        raise OSError("Atomic campaign replacement requires Linux renameat2") from exc
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(left), -100, os.fsencode(right), 2) != 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), str(right))


def _revision_receipt(path: Path, receipt: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def replace_campaign(
    root: Path,
    candidate_root: Path,
    campaign_id: str,
    expected_manifest_sha256: str,
    archive_root: Path,
) -> dict:
    """Explicitly replace a campaign revision while retaining its stable identifier.

    The existing create-only publication API remains unchanged. The caller must
    supply a fully built candidate under ``candidate_root / campaign_id`` and pin
    the active manifest. A verified old bundle is preserved at
    ``archive_root / campaign_id / old_sha`` outside the API registry. The candidate
    is copied to the registry filesystem, then exchanged atomically with the active
    directory. Verification or receipt failures after exchange restore the old
    directory. Neither input candidate nor either sealed manifest is rewritten.
    """
    import fcntl

    campaign_id = safe_id(campaign_id)
    if not SHA.fullmatch(expected_manifest_sha256):
        raise ValueError("Replacement requires an explicit current manifest SHA-256")
    root, candidate_root, archive_root = map(Path, (root, candidate_root, archive_root))
    root, candidate_root, archive_root = (
        root.resolve(),
        candidate_root.resolve(),
        archive_root.resolve(),
    )
    if archive_root.is_relative_to(root):
        raise ValueError("Campaign revisions must be archived outside the API registry")
    active = _inside(root, campaign_id)
    candidate = _inside(candidate_root, campaign_id)
    if (
        candidate.is_relative_to(active)
        or active.is_relative_to(candidate)
        or active.is_symlink()
        or candidate.is_symlink()
    ):
        raise ValueError("Replacement requires distinct real campaign directories")
    if not active.is_dir() or not candidate.is_dir():
        raise CampaignNotFound("Active campaign and replacement candidate must both exist")
    # Explicit publication produces real files. Reject links rather than copying
    # extra linked content outside a bundle or exchanging a mounted alias.
    if any(path.is_symlink() for base in (active, candidate) for path in base.rglob("*")):
        raise ValueError("Campaign replacement bundles cannot contain symlinks")
    with (root / ".campaign-replacement.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _verify_bundle(active, campaign_id, expected_manifest_sha256)
        candidate_check = _verify_bundle(candidate, campaign_id)
        new_sha = candidate_check["manifest_sha256"]
        if new_sha == expected_manifest_sha256:
            raise ValueError("Replacement candidate is identical to the active revision")
        archive = archive_root / campaign_id / expected_manifest_sha256
        if (
            root.is_relative_to(archive)
            or candidate.is_relative_to(archive)
            or archive.is_relative_to(candidate)
            or archive.parent.is_symlink()
            or archive.is_symlink()
        ):
            raise ValueError("Archive cannot contain active or candidate registry paths")
        archive.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".campaign-replace-", dir=root))
        staged = staging / campaign_id
        activated, preserve_staging = False, False
        receipt_path = archive.parent / f"revision_{uuid4().hex}.json"
        receipt = {
            "schema_version": 1,
            "campaign_id": campaign_id,
            "old_manifest_sha256": expected_manifest_sha256,
            "new_manifest_sha256": new_sha,
            "archive_directory": str(archive),
            "candidate_directory": str(candidate),
            "active_directory": str(active),
            "created_at": datetime.now(UTC).isoformat(),
            "status": "prepared",
        }
        try:
            shutil.copytree(candidate, staged)
            _verify_bundle(staged, campaign_id, new_sha)
            if archive.exists():
                _verify_bundle(archive, campaign_id, expected_manifest_sha256)
            else:
                archive_stage = Path(tempfile.mkdtemp(prefix=".revision-", dir=archive.parent))
                try:
                    copy = archive_stage / "bundle"
                    shutil.copytree(active, copy)
                    _verify_bundle(copy, campaign_id, expected_manifest_sha256)
                    copy.rename(archive)
                finally:
                    shutil.rmtree(archive_stage)
            _revision_receipt(receipt_path, receipt)
            # Recheck immediately before activation: expensive copies must not
            # make an earlier manifest pin silently authorize a different revision.
            _verify_bundle(active, campaign_id, expected_manifest_sha256)
            _verify_bundle(staged, campaign_id, new_sha)
            _exchange_directories(staged, active)
            activated = True
            _verify_bundle(active, campaign_id, new_sha)
            _verify_bundle(staged, campaign_id, expected_manifest_sha256)
            receipt.update(status="activated", activated_at=datetime.now(UTC).isoformat())
            _revision_receipt(receipt_path, receipt)
            return {**receipt, "receipt_path": str(receipt_path)}
        except BaseException as exc:
            was_activated = activated
            if activated:
                try:
                    _exchange_directories(staged, active)
                    activated = False
                    _verify_bundle(active, campaign_id, expected_manifest_sha256)
                except BaseException as rollback_error:
                    preserve_staging = True
                    raise RuntimeError(
                        f"Campaign rollback failed; preserved recovery files at {staging} "
                        f"and {archive}: {rollback_error}"
                    ) from exc
            receipt.update(
                status="rolled_back" if was_activated else "aborted",
                failed_at=datetime.now(UTC).isoformat(),
                failure=str(exc),
            )
            try:
                _revision_receipt(receipt_path, receipt)
            except OSError as receipt_error:
                exc.add_note(f"Unable to record failed replacement receipt: {receipt_error}")
            raise
        finally:
            if not preserve_staging:
                shutil.rmtree(staging)
