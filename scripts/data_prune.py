"""Plan (and optionally apply) a reversible slimming of ``data/`` and ``experiments/*/runs``.

    uv run python scripts/data_prune.py                      # dry run (default)
    uv run python scripts/data_prune.py --list               # also list every archived path
    uv run python scripts/data_prune.py --apply --archive /mnt/big/armchair-archive

Nothing is ever deleted. ``--apply`` MOVES prunable paths to ``<archive>/<path relative to
the repo>``; a JSON receipt of every move is written to the archive, and ``--restore
<receipt>`` moves them back. Stop ``engine serve``, the hourly refresh and the 15-minute
monitor before ``--apply`` (the script also refuses while their locks are held).

Only areas whose history the runtime never reads are pruned; everything else is kept:

league      data/league/<id>/<season>: the current catalog (tables/current.json) with its
            table releases and source captures, plus the newest receipts. Superseded
            sync releases/captures/catalogs are never read again (a reused capture key with
            a missing receipt is simply rebuilt). News and forecast-log query stores keep
            every receipt's closure; only their stale DuckDB caches move.
caches      cache/tables/*.duckdb that are not the current catalog's snapshot key
            (disposable; engine.tables.query.materialize rebuilds them on demand).
revisions   data/campaign_revisions/<campaign>/<old manifest sha>: archived bundles that
            replace_campaign writes but nothing reads. Newest N per campaign and every
            revision_*.json receipt are kept.
captures    research/opening_drive_v32_injury_captures (only the newest summary.json is
            read, by engine.data.freshness), opening_drive_v32_official_captures (only
            captures within 36 h for the current week are admitted) and superseded
            opening_drive_v2_sources_rolling_* directories (read only by the rescore run
            that wrote them).
evaluations data/outputs/opening_drive_v4_shadow/review_evaluations: write-only per-pass
            snapshots; the newest and any referenced by the log's JSON are kept.
runs        experiments/*/runs/* not reachable from code, configs, campaign provenance
            (src/engine/models/campaigns.json run_patterns etc.), published bundles,
            outputs, data pointers or another kept run/research manifest, and not the
            newest run of its family.

Pass 2 (deploy closure; see build_deploy_plan). The deployed host runs only ``engine
serve``, the hourly ``engine.data.pipeline refresh`` (+ ``campaigns.rescore --due``) and
the 15-minute ``campaigns.monitor``. An item is KEPT when the import closure of those
entry points, checked-in config, the offline tests, the data pointers they read
(table catalog, source_catalog.json, campaigns, league) or a "latest" selector they
evaluate names it, or when a kept item's manifests name it (transitively).

layers      raw snapshots, enriched/gold releases, table batches: incomplete releases
            (no manifest.json) and releases nothing names move; raw/objects not listed by
            any kept snapshot/manifest move.
research    data/research/* named by nothing move.
tables      data/tables releases/catalogs not pinned by the current (or published) catalog.
downloads   expired nflreadpy cache entries in cache/nflverse (older than
            nflverse_cache_duration; nflreadpy deletes them on next read).
outputs     data/outputs/* and data/benchmarks/*: never moved by default.

Opt-in ``--research-inputs``: items named only by research/training code, research history
pointers (data/current.json, catalog_history, releases/*.json, static) or other research
items, plus kept runs outside the deploy closure and cache/historical_evidence. The dry
run reports this group separately ("[opt-in"); ``--list`` prints it with ``??``.

Campaigns, static, releases pointers, league data and campaign revision receipts are
never moved.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shutil
import sys
from collections import defaultdict
from contextlib import ExitStack
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from fnmatch import fnmatchcase
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESEARCH = DATA / "research"
SHA_DIR = re.compile(r"^[0-9a-f]{64}$")


# ----------------------------------------------------------------- bookkeeping


@dataclass
class Item:
    path: Path
    archive: bool
    reason: str


@dataclass
class Area:
    name: str
    roots: list[Path]
    items: list[Item] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def keep(self, path: Path, reason: str):
        self.items.append(Item(path, False, reason))

    def move(self, path: Path, reason: str):
        self.items.append(Item(path, True, reason))


_SIZES: dict[Path, tuple[int, int]] = {}


def measure(path: Path) -> tuple[int, int]:
    """(files, bytes) for a file or directory tree; symlinks are not followed."""
    if path in _SIZES:
        return _SIZES[path]
    files = size = 0
    try:
        if path.is_symlink() or not path.is_dir():
            result = (1, path.lstat().st_size) if path.exists() or path.is_symlink() else (0, 0)
            _SIZES[path] = result
            return result
    except OSError:
        return 0, 0
    for directory, _, names in os.walk(path):
        for name in names:
            try:
                size += os.lstat(os.path.join(directory, name)).st_size
                files += 1
            except OSError:
                pass
    _SIZES[path] = (files, size)
    return files, size


def human(n: int) -> str:
    value = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:,.1f} {unit}" if unit != "B" else f"{n} B"
        value /= 1024
    return f"{n} B"


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def stamp(name: str) -> datetime | None:
    """Parse capture directory names such as 20261030T193004Z / 20261030T193004963110Z."""
    for fmt in ("%Y%m%dT%H%M%S%fZ", "%Y%m%dT%H%M%SZ", "%Y%m%d_%H%M%S"):
        try:
            return datetime.strptime(name, fmt).replace(tzinfo=UTC)
        except ValueError:
            continue
    return None


# ----------------------------------------------------------------- league / table stores


def catalog_closure(data: Path, ref_path: str) -> set[Path]:
    """Catalog file, its table release directories and the captures those releases read.

    Mirrors engine.tables.storage.load_catalog/_load_table (storage.py:44-83): a catalog
    pins release manifests; a release's ``inputs``/``recipe.source`` name its capture files.
    """
    keep: set[Path] = set()
    catalog_path = data / ref_path
    keep.add(catalog_path)
    catalog = read_json(catalog_path, {}) or {}
    pending = [ref["path"] for ref in catalog.get("tables", {}).values() if "path" in ref]
    seen = set()
    while pending:
        manifest_path = data / pending.pop()
        if manifest_path in seen:
            continue
        seen.add(manifest_path)
        keep.add(manifest_path.parent)
        manifest = read_json(manifest_path, {}) or {}
        sources = list(manifest.get("inputs", {}))
        if manifest.get("recipe", {}).get("source"):
            sources.append(manifest["recipe"]["source"])
        for name in sources:
            parts = Path(name).parts
            if len(parts) >= 2 and parts[0] == "captures":
                keep.add(data / parts[0] / parts[1])
            elif parts:
                keep.add(data / name)
        for dep in manifest.get("dependencies", {}).values():
            if isinstance(dep, dict) and "path" in dep:
                pending.append(dep["path"])
    return keep


def cache_key(data: Path) -> str | None:
    """engine.tables.query.materialize key for the current catalog (query.py:~80)."""
    try:
        import duckdb

        from engine.tables.storage import content_hash
    except ImportError:
        return None
    ref = read_json(data / "tables/current.json")
    if not ref:
        return None
    catalog = read_json(data / ref["path"])
    if catalog is None:
        return None
    return content_hash({"catalog": catalog, "duckdb": duckdb.__version__})


def plan_cache(area: Area, data: Path, label: str):
    cache = data / "cache/tables"
    if not cache.is_dir():
        return
    key = cache_key(data)
    entries = [p for p in cache.iterdir() if p.suffix in {".duckdb", ".json"}]
    if key is None and entries:
        # Fall back to the newest snapshot pair when the engine cannot be imported.
        newest = max(entries, key=lambda p: p.stat().st_mtime)
        key = newest.stem
        area.notes.append(f"{label}: engine import failed; kept newest cache by mtime")
    for path in sorted(entries):
        if path.stem == key:
            area.keep(path, "current catalog's DuckDB snapshot")
        else:
            area.move(path, "stale DuckDB snapshot of a superseded catalog (disposable cache)")


def plan_league_store(area: Area, directory: Path, newest_receipts: int | None):
    """``directory`` holds captures/<key>/catalog.json receipts and data/ (league.py:55-87)."""
    data = directory / "data"
    current = read_json(data / "tables/current.json")
    if not current:
        area.notes.append(f"{directory.relative_to(ROOT)}: no tables/current.json; kept all")
        return
    receipts = sorted(
        (p for p in (directory / "captures").glob("*/catalog.json")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    keep_closure = catalog_closure(data, current["path"])
    kept_receipts = set()
    for index, receipt in enumerate(receipts):
        ref = read_json(receipt, {}) or {}
        if newest_receipts is None or ref.get("path") == current["path"] or index < newest_receipts:
            kept_receipts.add(receipt.parent)
            if ref.get("path"):
                keep_closure |= catalog_closure(data, ref["path"])
    label = str(directory.relative_to(ROOT))
    for receipt_dir in sorted(p for p in (directory / "captures").glob("*") if p.is_dir()):
        if receipt_dir in kept_receipts:
            area.keep(receipt_dir, "capture receipt of a kept catalog")
        else:
            area.move(receipt_dir, "superseded sync receipt (its catalog is archived with it)")
    for capture in sorted(p for p in (data / "captures").glob("*") if p.is_dir()):
        if capture in keep_closure:
            area.keep(capture, "source capture of a kept table release")
        else:
            area.move(capture, "source capture used only by superseded releases")
    for catalog in sorted((data / "tables/catalogs").glob("*.json")):
        if catalog in keep_closure:
            area.keep(catalog, "current/kept table catalog")
        else:
            area.move(catalog, "superseded table catalog")
    for release in sorted((data / "tables/releases").glob("*/*")):
        if release in keep_closure:
            area.keep(release, "table release pinned by a kept catalog")
        else:
            area.move(release, "superseded immutable table release")
    plan_cache(area, data, label)


def plan_league(newest_receipts: int) -> Area:
    area = Area("league", [DATA / "league", DATA / "outputs/weekly_forecasts"])
    for directory in sorted(DATA.glob("league/*/*")):
        if not (directory / "data/tables/current.json").exists():
            continue
        # ESPN snapshot keys include captured_at, so old receipts never recur; news keys do.
        limit = newest_receipts if directory.parent.name.isdigit() else None
        plan_league_store(area, directory, limit)
    # Forecast-log query stores re-capture identical records (same key) on every request:
    # every receipt must stay loadable, so only stale caches move.
    for directory in sorted(DATA.glob("outputs/weekly_forecasts/*/*/.query-*")):
        if (directory / "data/tables/current.json").exists():
            plan_league_store(area, directory, None)
    return area


def plan_main_cache() -> Area:
    area = Area("cache", [DATA / "cache/tables"])
    plan_cache(area, DATA, "data")
    return area


# ----------------------------------------------------------------- campaign revisions


def plan_revisions(keep_newest: int) -> Area:
    root = DATA / "campaign_revisions"
    area = Area("campaign_revisions", [root])
    for campaign in sorted(p for p in root.iterdir() if p.is_dir()):
        bundles = sorted(
            (p for p in campaign.iterdir() if p.is_dir() and SHA_DIR.match(p.name)),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for index, bundle in enumerate(bundles):
            if index < keep_newest:
                area.keep(bundle, f"newest {keep_newest} archived revision(s) per campaign")
            else:
                area.move(bundle, "superseded archived bundle (written, never read)")
        for receipt in campaign.glob("revision_*.json"):
            area.keep(receipt, "revision receipt (tiny audit trail)")
    return area


# ----------------------------------------------------------------- research captures


def plan_captures(injury_hours: float, official_hours: float) -> Area:
    area = Area(
        "research_captures",
        [
            RESEARCH / "opening_drive_v32_injury_captures",
            RESEARCH / "opening_drive_v32_official_captures",
            *sorted(RESEARCH.glob("opening_drive_v2_sources_rolling_*")),
            RESEARCH / "backfill_snapshots",
        ],
    )
    # Provider (ESPN) injury captures: engine.data.freshness reads only the newest summary.
    injury = RESEARCH / "opening_drive_v32_injury_captures"
    dated = [(stamp(p.name), p) for p in injury.glob("*") if p.is_dir()]
    dated = [(t, p) for t, p in dated if t is not None]
    if dated:
        newest = max(t for t, _ in dated)
        cutoff = newest - timedelta(hours=injury_hours)
        # "Unchanged" captures are summaries pointing at the full capture they repeat.
        referenced = {
            (read_json(p / "summary.json", {}) or {}).get("observations_from")
            for t, p in dated
            if t >= cutoff
        }
        for t, p in sorted(dated):
            if p.name in referenced:
                area.keep(p, "full provider injury capture repeated by a newer unchanged one")
            elif t >= cutoff:
                area.keep(p, f"provider injury capture within {injury_hours:g} h of newest")
            else:
                area.move(p, "older provider injury capture (only the newest is read)")
    # Official report captures: usable_reports admits <=36 h old, current (season, week).
    official = RESEARCH / "opening_drive_v32_official_captures"
    rows = []
    for p in official.glob("*"):
        t = stamp(p.name)
        if p.is_dir() and t is not None:
            summary = read_json(p / "summary.json", {}) or {}
            rows.append((t, p, (summary.get("season"), summary.get("week"))))
    if rows:
        newest = max(t for t, _, _ in rows)
        weeks = [w for _, _, w in rows if None not in w]
        current_week = max(weeks) if weeks else None
        cutoff = newest - timedelta(hours=official_hours)
        for t, p, week in sorted(rows):
            if t >= cutoff:
                area.keep(p, f"official report capture within {official_hours:g} h of newest")
            elif week == current_week or None in week:
                area.keep(p, "official report capture for the current (season, week)")
            else:
                area.move(p, "official report capture for a past week (never admitted again)")
    # Rolling availability sources: written and read by the same rescore pass.
    rolling = sorted(RESEARCH.glob("opening_drive_v2_sources_rolling_*"))
    for index, p in enumerate(rolling):
        if index == len(rolling) - 1:
            area.keep(p, "newest rolling availability source rebuild")
        else:
            area.move(p, "superseded rolling availability sources (read only by its own run)")
    backfill = RESEARCH / "backfill_snapshots"
    if backfill.exists():
        area.keep(backfill, "derived backfill manifests (tiny; rewritten idempotently)")
    return area


# ----------------------------------------------------------------- per-pass evaluations


def plan_evaluations() -> Area:
    log = DATA / "outputs/opening_drive_v4_shadow"
    root = log / "review_evaluations"
    area = Area("v4_review_evaluations", [root])
    if not root.is_dir():
        return area
    referenced = set()
    for path in log.glob("*.json"):
        referenced.update(re.findall(r"review_evaluations/([0-9T]+Z)", path.read_text()))
    entries = sorted(p for p in root.iterdir() if p.is_dir())
    for index, p in enumerate(entries):
        if p.name in referenced:
            area.keep(p, "referenced by opening_drive_v4_shadow/*.json")
        elif index == len(entries) - 1:
            area.keep(p, "newest review evaluation")
        else:
            area.move(p, "superseded per-pass review evaluation (write-only)")
    return area


# ----------------------------------------------------------------- experiment runs

PATH_REF = re.compile(rb"(?:runs|research|campaigns)[/\\]+([A-Za-z0-9][A-Za-z0-9_.\-]*)")
QUOTED = re.compile(rb"[\"'`]([A-Za-z0-9][A-Za-z0-9_.\-]{3,})[\"'`/]")
TEXT_SUFFIXES = {".json", ".py", ".yaml", ".yml", ".toml", ".sql", ".cfg", ".txt"}
MAX_SCAN = 2_000_000


def tokens(path: Path) -> set[str]:
    try:
        if path.stat().st_size > MAX_SCAN:
            return set()
        content = path.read_bytes()
    except OSError:
        return set()
    found = {m.decode(errors="ignore") for m in PATH_REF.findall(content)}
    found.update(m.decode(errors="ignore") for m in QUOTED.findall(content))
    return found


def scan_tree(root: Path, suffixes=TEXT_SUFFIXES, skip=("runs", "__pycache__", ".git")) -> set:
    found: set[str] = set()
    if root.is_file():
        return tokens(root)
    for directory, dirnames, names in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip and not d.startswith(".venv")]
        for name in names:
            if Path(name).suffix in suffixes:
                found |= tokens(Path(directory) / name)
    return found


def family(name: str) -> str:
    """notebook_tree_<hash> -> notebook_tree; opening_drive_v3_20261011_002 -> opening_drive_v3."""
    parts = name.split("_")
    while len(parts) > 1 and (
        parts[-1].isdigit() or re.fullmatch(r"[0-9a-f]{12,}", parts[-1]) or parts[-1] in {"r"}
    ):
        parts.pop()
    return "_".join(parts)


def plan_runs(deploy_runs: frozenset[Path] = frozenset(), research_inputs: bool = False) -> Area:
    """Pass 1 rules; with ``deploy_runs`` (pass 2), kept runs outside it are research-only."""
    runs = sorted(p for p in ROOT.glob("experiments/*/runs/*") if p.is_dir())
    area = Area("experiment_runs", sorted({p.parent for p in runs}))
    by_name: dict[str, list[Path]] = defaultdict(list)
    for run in runs:
        by_name[run.name].append(run)
    reasons: dict[Path, str] = {}

    def mark(names, reason):
        added = []
        for name in names:
            for run in by_name.get(name, []):
                if run not in reasons:
                    reasons[run] = reason
                    added.append(run)
        return added

    # 1. Code and checked-in configs (src, experiments outside runs, research/*.py, deploy).
    code = set()
    for root in (ROOT / "src", ROOT / "experiments", ROOT / "research", ROOT / "deploy"):
        if root.exists():
            code |= scan_tree(root)
    for name in ("Justfile", "pyproject.toml"):
        code |= tokens(ROOT / name)
    frontier = mark(code, "named in code/config")
    # 2. Campaign provenance globs (engine.models.catalog associates these runs).
    provenance = read_json(ROOT / "src/engine/models/campaigns.json", {}) or {}
    patterns = []
    for row in provenance.get("campaigns", []) + provenance.get("archived_campaigns", []):
        patterns += row.get("run_patterns", [])
        patterns += [
            row[k]
            for k in ("comparison_directory", "saved_model_manifest", "api_bundle")
            if row.get(k)
        ]
    for run in runs:
        relative = run.relative_to(ROOT).as_posix()
        if run not in reasons and any(
            fnmatchcase(relative, p) or p.startswith(relative + "/") for p in patterns
        ):
            reasons[run] = "matches campaigns.json run_patterns/comparison_directory"
            frontier.append(run)
    # 3. Runtime glob consumers in code.
    for run in runs:
        if run in reasons:
            continue
        if run.name.startswith("forecast_sources_"):
            # rescore.latest_supplements globs every forecast_sources_* run (rescore.py:68).
            reasons[run] = "globbed by rescore.latest_supplements"
            frontier.append(run)
        elif run.name.startswith("opening_drive_v31_points_"):
            reasons[run] = "globbed by weekly_v32/run.py"
            frontier.append(run)
    # 4. Runtime data pointers and kept research/output manifests.
    data_roots = [
        DATA / "current.json",
        DATA / "source_catalog.json",
        DATA / "releases",
        DATA / "catalog_history",
        DATA / "weekly",
        DATA / "campaigns",
        DATA / "outputs",
        DATA / "benchmarks",
        DATA / "tables/batches",
        DATA / "raw/snapshots",
        DATA / "enriched",
        DATA / "gold",
    ]
    data_tokens = set()
    for root in data_roots:
        if root.exists():
            data_tokens |= scan_tree(root, suffixes={".json", ".yaml", ".yml"}, skip=("cache",))
    skip_research = {
        "opening_drive_v32_injury_captures",
        "opening_drive_v32_official_captures",
    }
    for directory in sorted(RESEARCH.iterdir()):
        if directory.name not in skip_research:
            data_tokens |= scan_tree(directory, suffixes={".json", ".yaml", ".yml"})
    frontier += mark(data_tokens, "referenced by data manifests/bundles/outputs/research")
    # 5. Newest run of each family stays (latest line of research; warm notebook caches).
    families: dict[tuple[Path, str], list[Path]] = defaultdict(list)
    for run in runs:
        families[(run.parent, family(run.name))].append(run)
    for members in families.values():
        newest = max(members, key=lambda p: p.stat().st_mtime)
        if newest not in reasons:
            reasons[newest] = "newest run of its family"
            frontier.append(newest)
    # 6. Transitive closure: manifests inside every kept run may name other runs.
    scanned = set()
    while frontier:
        run = frontier.pop()
        if run in scanned:
            continue
        scanned.add(run)
        frontier += mark(
            # .py too: runs keep source snapshots whose defaults name sibling runs.
            scan_tree(run, suffixes={".json", ".yaml", ".yml", ".py"}, skip=("__pycache__",)),
            "referenced by another kept run's manifests/source snapshot",
        )
    for run in runs:
        if run in deploy_runs:
            area.keep(run, "used by the deployed processes (pass-2 deploy closure)")
        elif run in reasons and deploy_runs:
            research_only(area, run, f"research run ({reasons[run]})", research_inputs)
        elif run in reasons:
            area.keep(run, reasons[run])
        else:
            area.move(run, "unreferenced research run (not in code, configs, bundles or manifests)")
    return area


# ----------------------------------------------------------------- pass 2: deploy closure
#
# The deployed host runs only ``engine serve`` (engine.api.app), the hourly
# ``python -m engine.data.pipeline refresh`` (which shells out to ``campaigns.rescore
# --due``) and the 15-minute ``campaigns.monitor``. Everything those processes can name
# is kept: identifiers in their import closure, checked-in configs, the offline test
# suite, the data pointers they read, the "latest" selectors they evaluate, and
# transitively anything a kept item's manifests/source snapshots name. Items named only
# by other code (research scripts, notebooks, labs), by research history pointers
# (data/current.json, catalog_history, releases/*.json, static) or by other research
# items are the opt-in ``--research-inputs`` group. Items nothing names at all are
# archived by default.

DEPLOY_ENTRIES = (
    "engine.api.app",  # engine serve -> uvicorn engine.api.app (cli.py:60)
    "engine.data.pipeline",  # deploy/armchair-refresh.service
    "engine.data.refresh",
    "experiments.future_player_lab.campaigns.rescore",  # refresh.py:30 RESCORE subprocess
    "experiments.future_player_lab.campaigns.monitor",  # deploy/opening-drive-monitor.service
)
CAMPAIGNS_PACKAGE = "experiments.future_player_lab.campaigns"
CAMPAIGN_CONFIGS = ROOT / "experiments/future_player_lab/campaigns/configs"
NAME_TOKEN = re.compile(rb"[A-Za-z0-9][A-Za-z0-9_.\-]{3,}")
HEX64 = re.compile(r"[0-9a-f]{64}")
SCAN_SUFFIXES = {".json", ".yaml", ".yml", ".py", ".toml", ".txt", ".md", ".sql", ".cfg"}
MAX_TOKEN_FILE = 64_000_000
LAYER_PARENTS = ("raw/snapshots", "enriched/releases", "gold/releases", "tables/batches")
# Handled (and kept/pruned by time) by the pass-1 ``captures`` area; never scanned here.
PASS1_RESEARCH = ("opening_drive_v32_injury_captures", "opening_drive_v32_official_captures")


def research_only(area: Area, path: Path, why: str, archive: bool):
    reason = f"research/training-only: {why}"
    if archive:
        area.move(path, reason + " [--research-inputs]")
    else:
        area.keep(path, reason + " [opt-in: --research-inputs]")


def module_file(name: str) -> Path | None:
    parts = name.split(".")
    if parts[0] not in {"engine", "experiments"}:
        return None
    base = (ROOT / "src" if parts[0] == "engine" else ROOT).joinpath(*parts)
    if (base / "__init__.py").is_file():
        return base / "__init__.py"
    if base.with_suffix(".py").is_file():
        return base.with_suffix(".py")
    return None


def import_closure(entries) -> set[Path]:
    """Repo modules reachable from ``entries`` through any import, including lazy ones.

    An over-approximation on purpose (imports inside functions count): every module a
    deployed process could execute contributes the identifiers it names.
    """
    import ast

    seen: dict[str, Path] = {}
    stack = list(entries)
    while stack:
        name = stack.pop()
        if not name or name in seen:
            continue
        path = module_file(name)
        if path is None:
            continue
        seen[name] = path
        parts = name.split(".")
        stack += [".".join(parts[:i]) for i in range(1, len(parts))]
        package = name if path.name == "__init__.py" else ".".join(parts[:-1])
        try:
            tree = ast.parse(path.read_text())
        except (SyntaxError, OSError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                stack += [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base = package.split(".")[: len(package.split(".")) - (node.level - 1)]
                    module = ".".join(base + ([node.module] if node.module else []))
                else:
                    module = node.module or ""
                stack.append(module)
                stack += [f"{module}.{alias.name}" for alias in node.names]
            elif (
                isinstance(node, ast.Call)
                and getattr(node.func, "attr", getattr(node.func, "id", None)) == "import_module"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                stack.append(node.args[0].value)
    return set(seen.values())


def deploy_modules() -> set[Path]:
    entries = list(DEPLOY_ENTRIES)
    # Tracker steps are "module:ATTR" relative to the campaigns package (rescore.py:~405).
    for config in CAMPAIGN_CONFIGS.glob("*.json"):
        for module in re.findall(r'"([a-z_][a-z0-9_.]*):[A-Za-z_]\w*"', config.read_text()):
            entries.append(f"{CAMPAIGNS_PACKAGE}.{module}")
    return import_closure(entries)


class Inventory:
    """Named candidate directories and the identifiers each file/tree mentions."""

    def __init__(self):
        self.nodes: dict[str, list[Path]] = defaultdict(list)
        self._files: dict[Path, set[str]] = {}
        self._trees: dict[tuple, set[str]] = {}
        for parent in LAYER_PARENTS:
            for path in sorted((DATA / parent).glob("*")):
                if path.is_dir():
                    self.nodes[path.name].append(path)
        for path in sorted(RESEARCH.glob("*")):
            if path.is_dir() and path.name not in PASS1_RESEARCH:
                self.nodes[path.name].append(path)
        for parent in ("outputs", "benchmarks"):
            for path in sorted((DATA / parent).glob("*")):
                self.nodes[path.name].append(path)
        for path in sorted(ROOT.glob("experiments/*/runs/*")):
            if path.is_dir():
                self.nodes[path.name].append(path)
        # Main-catalog table releases (<table>/<timestamp>_<uuid>): reports' frozen
        # inputs.json pin them by path.
        for path in sorted((DATA / "tables/releases").glob("*/*")):
            if path.is_dir():
                self.nodes[path.name].append(path)

    def relevant(self, token: str) -> bool:
        return token in self.nodes or bool(HEX64.fullmatch(token))

    def file(self, path: Path) -> set[str]:
        if path not in self._files:
            found: set[str] = set()
            try:
                if path.stat().st_size <= MAX_TOKEN_FILE:
                    for raw in NAME_TOKEN.findall(path.read_bytes()):
                        token = raw.decode(errors="ignore")
                        if self.relevant(token):
                            found.add(token)
            except OSError:
                pass
            self._files[path] = found
        return self._files[path]

    def tree(self, root: Path, skip=("__pycache__", ".git", "node_modules", "runs")) -> set[str]:
        key = (root, skip)
        if key in self._trees:
            return self._trees[key]
        found: set[str] = set()
        if root.is_file():
            found = self.file(root) if root.suffix in SCAN_SUFFIXES else set()
        elif root.is_dir():
            for directory, dirnames, names in os.walk(root):
                dirnames[:] = [
                    d for d in dirnames if d not in skip and not d.startswith((".venv", ".query-"))
                ]
                for name in names:
                    if Path(name).suffix in SCAN_SUFFIXES:
                        found |= self.file(Path(directory) / name)
        self._trees[key] = found
        return found

    def item_tokens(self, item: Path) -> set[str]:
        # Inside an item every subdirectory counts (a run's own "runs" folder included).
        return self.tree(item, skip=("__pycache__", ".git", "node_modules"))


def raw_selectors() -> dict[Path, str]:
    """Raw snapshots the deployed code selects as "latest" by scanning the directory."""
    documents = {}
    for manifest in (DATA / "raw/snapshots").glob("*/manifest.json"):
        document = read_json(manifest, {}) or {}
        if document.get("status") == "accepted":
            documents[manifest.parent] = document
    picks: dict[Path, str] = {}

    def pick(rows, reason):
        if rows:
            path = max(rows, key=lambda row: row[0])[1]
            picks.setdefault(path, reason)

    pick(
        [(d["registered_at"], p) for p, d in documents.items() if d.get("registered_at")],
        "capture.latest_snapshot (capture.py:57): refresh carries history forward from it",
    )
    pick(
        [
            (d.get("registered_at", ""), p)
            for p, d in documents.items()
            if "current_observations" in d
        ],
        "rescore.latest_raw_snapshot (rescore.py:56): monitor/rescore input",
    )
    pick(
        [
            (
                (
                    d["current_observations"]["season"],
                    d["current_observations"]["through_week"],
                    str(d["current_observations"].get("saved_at", "")),
                    str(d.get("registered_at", "")),
                ),
                p,
            )
            for p, d in documents.items()
            if "through_week" in (d.get("current_observations") or {})
        ],
        "weekly_v41.track.latest_snapshot (track.py:35)",
    )
    seasons = defaultdict(list)
    for p, d in documents.items():
        if d.get("kind") == "current_injury_capture":
            season = (d.get("current_observations") or {}).get("season")
            seasons[season].append((d.get("registered_at", ""), p))
    for rows in seasons.values():
        pick(rows, "newest reviewed injury capture per season (espn/attention.py:30)")
    return picks


def latest_batch_path() -> Path | None:
    """engine.data.releases.latest_batch (releases.py:188): refresh/tables builder input."""
    rows = []
    for parent in (DATA / "tables/batches", DATA / "gold/releases"):
        for root in parent.glob("*"):
            document = read_json(root / "manifest.json", {}) or {}
            if document.get("status") == "accepted" and document.get("generated_at"):
                rows.append((document["generated_at"], root))
    return max(rows)[1] if rows else None


@dataclass
class Closure:
    reasons: dict[Path, str]
    hashes: set[str]
    via: dict[Path, Path] = field(default_factory=dict)  # transitive item -> naming item


def layer_dependencies(path: Path) -> set[str]:
    """What the loaders dereference from a raw/enriched/gold/batch manifest, nothing more.

    load_manifest (releases.py:127-160) verifies ``files`` and recurses into ``input``;
    capture_raw (capture.py:150-200) reuses the previous snapshot's ``assets`` objects and
    reads its ``college_source`` research capture. Provenance-only fields (``history``,
    ``protected_artifacts``, limitations) are never opened, so they are not followed.
    """
    manifest = read_json(path / "manifest.json", {}) or {}
    names: set[str] = set()
    if isinstance(manifest.get("input"), dict) and manifest["input"].get("version"):
        names.add(manifest["input"]["version"])
    if manifest.get("college_source"):
        names.add(manifest["college_source"])
    for name, sha in (manifest.get("files") or {}).items():
        names.add(Path(name).name)
        if isinstance(sha, str):
            names.add(sha)
    for asset in (manifest.get("assets") or {}).values():
        if isinstance(asset, dict) and asset.get("object"):
            names.add(Path(asset["object"]).name)
    return names


def is_layer(path: Path) -> bool:
    return any(path.parent == DATA / parent for parent in LAYER_PARENTS)


def close(
    inv: Inventory,
    roots: list[tuple[set[str], str]],
    seeds: dict[Path, str],
    structural_layers: bool = True,
    follow_records: bool = True,
) -> Closure:
    """Transitive closure: kept items' manifests/source snapshots may name more items.

    Layer releases contribute only what their loaders dereference (layer_dependencies);
    research dirs, runs and outputs contribute every identifier they mention (their
    readers may follow any pointer, so that is the conservative choice). With
    ``follow_records=False`` data/outputs and data/benchmarks entries are kept but not
    followed: they are record stores/reports the deploy writes or reads as records
    (forecast logs carry provenance labels such as "version": "<research product>" that
    no deployed reader resolves to a path).
    """
    reasons = dict(seeds)
    via: dict[Path, Path] = {}
    hashes: set[str] = set()
    frontier = list(seeds)
    for tokens, reason in roots:
        hashes.update(t for t in tokens if HEX64.fullmatch(t))
        for token in tokens:
            for path in inv.nodes.get(token, []):
                if path not in reasons:
                    reasons[path] = reason
                    frontier.append(path)
    scanned = set()
    while frontier:
        path = frontier.pop()
        if path in scanned:
            continue
        scanned.add(path)
        if structural_layers and is_layer(path):
            tokens = layer_dependencies(path)
        elif not follow_records and path.parent in (DATA / "outputs", DATA / "benchmarks"):
            continue
        else:
            tokens = inv.item_tokens(path)
        hashes.update(t for t in tokens if HEX64.fullmatch(t))
        for token in tokens:
            for other in inv.nodes.get(token, []):
                if other not in reasons:
                    reasons[other] = "named by a kept item's manifests/sources (transitive)"
                    via[other] = path
                    frontier.append(other)
    return Closure(reasons, hashes, via)


@dataclass
class DeployPlan:
    inventory: Inventory
    deploy: Closure
    research: Closure
    main_tables: set[Path]
    published_tables: set[Path]


def build_deploy_plan() -> DeployPlan:
    inv = Inventory()
    modules = deploy_modules()
    code_tokens: set[str] = set()
    for path in modules:
        code_tokens |= inv.file(path)
    # Checked-in configuration the deployed code reads (registry.yaml, campaigns.json,
    # league.yaml, campaign configs, ...): every non-Python text file under src/ and the
    # campaign configs, plus deploy units.
    # Only config files a deployed module names (registry.yaml, league.yaml, ...); e.g.
    # engine/models/campaigns.json is read only by the `engine models` CLI.
    module_text = "\n".join(path.read_text(errors="ignore") for path in modules)
    config_tokens: set[str] = set()
    for directory, dirnames, names in os.walk(ROOT / "src"):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for name in names:
            if Path(name).suffix in SCAN_SUFFIXES - {".py", ".md"} and name in module_text:
                config_tokens |= inv.file(Path(directory) / name)
    config_tokens |= inv.tree(CAMPAIGN_CONFIGS) | inv.tree(ROOT / "deploy")
    config_tokens |= inv.file(ROOT / "pyproject.toml") | inv.file(ROOT / "Dockerfile")
    test_tokens = inv.tree(ROOT / "tests") | set().union(
        *(inv.tree(p) for p in ROOT.glob("experiments/*/tests"))
    )
    # Data pointers the deployed processes read.
    # The API, refresh and rescore open only tables/current.json (storage.py:28). The
    # published handoff (releases/tables/current.json) is read only by engine setup when
    # the local catalog is empty (startup.py:53) -> research/setup group.
    main_tables = catalog_closure(DATA, (read_json(DATA / "tables/current.json") or {})["path"])
    published = read_json(DATA / "releases/tables/current.json", {}) or {}
    published_tables: set[Path] = set()
    if published.get("table_catalog", {}).get("path"):
        published_tables = catalog_closure(DATA, published["table_catalog"]["path"]) - main_tables
    pointer_tokens = set(inv.tree(DATA / "source_catalog.json"))
    for path in main_tables:
        pointer_tokens |= inv.tree(path)
    pointer_tokens |= inv.tree(DATA / "campaigns") | inv.tree(DATA / "league")
    seeds: dict[Path, str] = dict(raw_selectors())
    batch = latest_batch_path()
    if batch is not None:
        seeds.setdefault(batch, "latest_batch (releases.py:188): refresh/tables builder input")
    for path in RESEARCH.glob("opening_drive_v2_sources_rolling_*"):
        seeds[path] = "rolling availability sources (pass-1 captures area decides)"
    for run in ROOT.glob("experiments/future_player_lab/runs/forecast_sources_*"):
        seeds[run] = "globbed by rescore.latest_supplements (rescore.py:68)"
    for run in ROOT.glob("experiments/future_player_lab/runs/opening_drive_v31_points_*"):
        seeds[run] = "globbed by weekly_v32/run.py:364"
    deploy = close(
        inv,
        [
            (code_tokens, "named in code the deployed processes import"),
            (config_tokens, "named in checked-in config (src/*.yaml|json, campaign configs)"),
            (
                pointer_tokens,
                "named by a deployed data pointer (catalog, source_catalog, campaigns)",
            ),
            (test_tokens, "named by the offline test suite"),
        ],
        seeds,
        follow_records=False,
    )
    # Research roots: all other code and research-history pointers, plus every
    # surviving run (pass 1 keeps only runs worth keeping).
    other_code: set[str] = set()
    for root in (ROOT / "src", ROOT / "experiments", ROOT / "research", ROOT / "scripts"):
        for directory, dirnames, names in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in {"__pycache__", "runs"}]
            for name in names:
                path = Path(directory) / name
                if path.suffix in SCAN_SUFFIXES and path != Path(__file__).resolve():
                    other_code |= inv.file(path)
    history = set()
    for root in ("current.json", "catalog_history", "weekly", "releases", "static"):
        history |= inv.tree(DATA / root)
    for path in published_tables:
        history |= inv.tree(path)
    research_seeds = {
        run: "kept research run"
        for run in ROOT.glob("experiments/*/runs/*")
        if run.is_dir() and run not in deploy.reasons
    }
    research = close(
        inv,
        [
            (other_code, "named by research/training code (not imported by the deploy)"),
            (
                history,
                "named by research/setup pointers (data/current.json, catalog_history, "
                "releases/*.json, published table catalog)",
            ),
        ],
        research_seeds,
    )
    return DeployPlan(inv, deploy, research, main_tables, published_tables)


def classify(area: Area, plan: DeployPlan, path: Path, research_inputs: bool, unnamed: str):
    if path in plan.deploy.reasons:
        area.keep(path, plan.deploy.reasons[path])
    elif path in plan.research.reasons:
        research_only(area, path, plan.research.reasons[path], research_inputs)
    elif unnamed:
        area.move(path, unnamed)
    else:
        research_only(area, path, "not named anywhere (human-facing output)", research_inputs)


def plan_layers(plan: DeployPlan, research_inputs: bool) -> Area:
    """Raw snapshots, enriched/gold releases, table batches and the raw object store."""
    area = Area("layers", [DATA / p for p in LAYER_PARENTS] + [DATA / "raw/objects"])
    for parent in LAYER_PARENTS:
        for path in sorted(p for p in (DATA / parent).glob("*") if p.is_dir()):
            if not (path / "manifest.json").is_file():
                # load_manifest/latest_batch/latest_snapshot all require manifest.json
                # (releases.py:127-140,195-201; capture.py:61): never readable.
                area.move(path, "incomplete release (no manifest.json; no loader can open it)")
            else:
                classify(
                    area,
                    plan,
                    path,
                    research_inputs,
                    "release named by nothing (no code, config, pointer or kept manifest)",
                )
    # Raw objects are content-addressed (releases.py:92); snapshot manifests list them.
    deploy_hashes = plan.deploy.hashes
    research_hashes = plan.research.hashes
    for path in sorted((DATA / "raw/objects").glob("*/*")):
        if path.name in deploy_hashes:
            area.keep(path, "raw object of a kept snapshot (or named by deploy data)")
        elif path.name in research_hashes:
            research_only(area, path, "raw object of a research-only snapshot", research_inputs)
        else:
            area.move(path, "orphan raw object (no snapshot or manifest lists it)")
    return area


def plan_research_dirs(plan: DeployPlan, research_inputs: bool) -> Area:
    roots = [p for p in sorted(RESEARCH.glob("*")) if p.name not in PASS1_RESEARCH]
    area = Area("research", [RESEARCH])
    for path in roots:
        if not path.is_dir():
            area.keep(path, "loose research file (tiny)")
            continue
        classify(
            area,
            plan,
            path,
            research_inputs,
            "research directory named by nothing (no code, config, pointer or manifest)",
        )
    for name in PASS1_RESEARCH:
        if (RESEARCH / name).exists():
            area.keep(RESEARCH / name, "pass-1 captures area prunes these by age")
    return area


def plan_outputs(plan: DeployPlan, research_inputs: bool) -> Area:
    area = Area("outputs_benchmarks", [DATA / "outputs", DATA / "benchmarks"])
    for parent in ("outputs", "benchmarks"):
        for path in sorted((DATA / parent).glob("*")):
            if path.name.startswith("."):
                area.keep(path, "hidden runtime file")
            else:
                classify(area, plan, path, research_inputs, "")
    return area


def plan_main_tables(plan: DeployPlan, research_inputs: bool) -> Area:
    """data/tables: the current catalog's releases stay; superseded ones are never read.

    storage.current/load_catalog (storage.py:28-55) open only tables/current.json; build
    refresh compares fingerprints with that catalog's manifests (build.py:~75); transport
    fetch re-downloads any older published catalog from GCS.
    """
    area = Area("tables", [DATA / "tables/releases", DATA / "tables/catalogs"])
    setup = "pinned only by releases/tables/current.json (engine setup offline restore)"
    for catalog in sorted((DATA / "tables/catalogs").glob("*.json")):
        if catalog in plan.main_tables:
            area.keep(catalog, "current table catalog")
        elif catalog.stem in plan.deploy.hashes:
            area.keep(catalog, "table catalog named by deploy data")
        elif catalog in plan.published_tables:
            research_only(area, catalog, setup, research_inputs)
        elif catalog.stem in plan.research.hashes:
            research_only(area, catalog, "table catalog named by research data", research_inputs)
        else:
            area.move(catalog, "superseded table catalog")
    for release in sorted((DATA / "tables/releases").glob("*/*")):
        if release in plan.main_tables:
            area.keep(release, "table release pinned by the current catalog")
        elif release in plan.deploy.reasons:
            area.keep(release, plan.deploy.reasons[release])
        elif release in plan.published_tables:
            research_only(area, release, setup, research_inputs)
        elif release in plan.research.reasons:
            research_only(area, release, plan.research.reasons[release], research_inputs)
        else:
            area.move(release, "superseded immutable table release (named by nothing)")
    return area


def plan_download_caches(plan: DeployPlan, research_inputs: bool, max_age: float) -> Area:
    nflverse = DATA / "cache/nflverse"
    evidence = DATA / "cache/historical_evidence"
    area = Area("download_caches", [nflverse, evidence])
    # nflreadpy filesystem cache (forecasting/capture.py:22): entries older than
    # settings.nflverse_cache_duration are expired and deleted on next read anyway.
    cutoff = datetime.now(UTC).timestamp() - max_age
    for path in sorted(nflverse.glob("*")):
        if path.stat().st_mtime < cutoff:
            area.move(path, f"expired nflreadpy cache entry (> {max_age / 3600:g} h old)")
        else:
            area.keep(path, "fresh nflreadpy cache entry")
    # Read only by load_transaction_backfill via engine.data.availability_history.build
    # (historical_evidence.py:63) and research/*.py; not in the deploy import closure.
    deploy_reads = any(
        "cache/historical_evidence" in p.read_text() or "load_transaction_backfill" in p.read_text()
        for p in deploy_modules()
        if p.name != "historical_evidence.py"
    )
    if evidence.exists():
        if deploy_reads:
            area.keep(evidence, "read by a deployed module")
        else:
            research_only(
                area, evidence, "evidence captures for availability_history.build", research_inputs
            )
    return area


# ----------------------------------------------------------------- report / apply


def summarize(areas: list[Area], show: bool) -> tuple[int, int]:
    grand_keep = grand_move = 0
    grand_keep_files = grand_move_files = 0
    for area in areas:
        total_files = total_bytes = 0
        for root in area.roots:
            if root.exists():
                f, b = measure(root)
                total_files += f
                total_bytes += b
        by_reason: dict[tuple[bool, str], list[int]] = defaultdict(lambda: [0, 0, 0])
        move_files = move_bytes = 0
        enumerated_files = enumerated_bytes = 0
        for item in area.items:
            f, b = measure(item.path)
            row = by_reason[(item.archive, item.reason)]
            row[0] += 1
            row[1] += f
            row[2] += b
            enumerated_files += f
            enumerated_bytes += b
            if item.archive:
                move_files += f
                move_bytes += b
        keep_files, keep_bytes = total_files - move_files, total_bytes - move_bytes
        other_files = total_files - enumerated_files
        other_bytes = total_bytes - enumerated_bytes
        print(
            f"\n== {area.name}  ({', '.join(str(r.relative_to(ROOT)) for r in area.roots[:3])}"
            f"{', ...' if len(area.roots) > 3 else ''})"
        )
        print(f"   KEEP    {keep_files:>8,} files {human(keep_bytes):>12}")
        print(f"   ARCHIVE {move_files:>8,} files {human(move_bytes):>12}")
        for (archive, reason), (n, f, b) in sorted(
            by_reason.items(), key=lambda kv: (not kv[0][0], -kv[1][2])
        ):
            verb = "archive" if archive else "keep   "
            print(f"     {verb} {n:>5} item(s) {f:>8,} files {human(b):>12}  {reason}")
        if other_files > 0:
            print(
                f"     keep    {'':>5}         {other_files:>8,} files {human(other_bytes):>12}"
                "  other files in area (locks, pointers, untouched)"
            )
        for note in area.notes:
            print(f"     note: {note}")
        if show:
            for item in area.items:
                if item.archive:
                    print(f"       -> {item.path.relative_to(ROOT)}")
                elif "[opt-in" in item.reason:
                    print(
                        f"       ?? {item.path.relative_to(ROOT)}  ({human(measure(item.path)[1])})"
                    )
        grand_keep += keep_bytes
        grand_move += move_bytes
        grand_keep_files += keep_files
        grand_move_files += move_files
    print(
        f"\n== TOTAL (pruned areas only)  keep {grand_keep_files:,} files {human(grand_keep)}"
        f"  |  archive {grand_move_files:,} files {human(grand_move)}"
    )
    optin = [i for a in areas for i in a.items if not i.archive and "[opt-in" in i.reason]
    if optin:
        files = sum(measure(i.path)[0] for i in optin)
        size = sum(measure(i.path)[1] for i in optin)
        print(
            f"== OPT-IN research/training-only group (kept; --research-inputs archives it): "
            f"{len(optin)} item(s), {files:,} files {human(size)}"
        )
    return grand_move_files, grand_move


LOCKS = [
    DATA / ".runtime/data_pipeline.lock",
    DATA / "campaigns/.campaign-replacement.lock",
    DATA / "outputs/opening_drive_v41/.tracking.lock",
    DATA / "outputs/opening_drive_v32_shadow/.tracking.lock",
    DATA / "outputs/opening_drive_v4_shadow/.tracking.lock",
    DATA / "cache/tables/.lock",
]


def hold_locks(stack: ExitStack):
    paths = list(LOCKS)
    for directory in DATA.glob("league/*/*"):
        paths += [
            directory / ".capture-lock/.lock",
            directory / "data/cache/tables/.lock",
            directory / "data/.runtime/table-refresh/.lock",
        ]
    for path in paths:
        if not path.is_file():
            continue  # never create lock files
        stream = stack.enter_context(path.open("rb"))
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            held = path.relative_to(ROOT)
            raise SystemExit(f"Refusing: {held} is held; stop the services") from None


def check_archive(archive: Path) -> Path:
    archive = archive.expanduser().resolve()
    for forbidden in [DATA.resolve(), *[p.resolve() for p in ROOT.glob("experiments/*/runs")]]:
        if archive == forbidden or archive.is_relative_to(forbidden):
            raise SystemExit(f"Refusing: archive {archive} is inside {forbidden}")
    if archive == ROOT.resolve():
        raise SystemExit("Refusing: archive cannot be the repository root")
    return archive


def apply(areas: list[Area], archive: Path):
    archive = check_archive(archive)
    moves = [item for area in areas for item in area.items if item.archive]
    when = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    receipt_path = archive / f"data_prune_{when}.json"
    archive.mkdir(parents=True, exist_ok=True)
    done = []
    with ExitStack() as stack:
        hold_locks(stack)
        for item in moves:
            relative = item.path.relative_to(ROOT)
            destination = archive / relative
            if not item.path.exists():
                continue
            if destination.exists():
                raise SystemExit(f"Refusing to overwrite {destination}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(item.path), str(destination))
            done.append({"path": relative.as_posix(), "reason": item.reason})
            if len(done) % 500 == 0:
                receipt_path.write_text(json.dumps({"root": str(ROOT), "moved": done}, indent=1))
    receipt_path.write_text(
        json.dumps({"root": str(ROOT), "archive": str(archive), "moved": done}, indent=1)
    )
    print(f"\nMoved {len(done)} paths into {archive}; receipt {receipt_path}")
    print(f"Undo: uv run python scripts/data_prune.py --restore {receipt_path}")


def restore(receipt_path: Path):
    receipt = json.loads(receipt_path.read_text())
    archive = Path(receipt.get("archive", receipt_path.parent))
    restored = 0
    for row in receipt["moved"]:
        source, destination = archive / row["path"], ROOT / row["path"]
        if not source.exists():
            continue
        if destination.exists():
            print(f"skip (exists): {row['path']}")
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))
        restored += 1
    print(f"Restored {restored} paths")


AREAS = [
    "league",
    "cache",
    "revisions",
    "captures",
    "evaluations",
    "runs",
    "layers",
    "research",
    "outputs",
    "tables",
    "downloads",
]


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--apply", action="store_true", help="Move prunable paths (+ --archive)")
    parser.add_argument("--archive", type=Path, help="Destination outside data/")
    parser.add_argument("--restore", type=Path, help="Move paths in a prune receipt back")
    parser.add_argument("--list", action="store_true", help="List every path to archive")
    parser.add_argument(
        "--keep-revisions",
        type=int,
        default=2,
        help="Archived bundles kept per campaign (default 2)",
    )
    parser.add_argument(
        "--league-receipts",
        type=int,
        default=3,
        help="Newest ESPN sync receipts kept besides the current one (default 3)",
    )
    parser.add_argument(
        "--injury-hours",
        type=float,
        default=24,
        help="Provider injury captures kept, hours before the newest (default 24)",
    )
    parser.add_argument(
        "--official-hours",
        type=float,
        default=72,
        help="Official report captures kept besides the current week (default 72)",
    )
    parser.add_argument(
        "--research-inputs",
        action="store_true",
        help="Also archive the research/training-only group (not used by the deploy)",
    )
    parser.add_argument(
        "--nflverse-max-age",
        type=float,
        default=None,
        help="nflverse cache entries older than this many seconds are expired "
        "(default settings.nflverse_cache_duration)",
    )
    parser.add_argument(
        "--area",
        action="append",
        choices=AREAS,
        help="Limit to these areas (default all)",
    )
    args = parser.parse_args(argv)
    if args.restore:
        return restore(args.restore)
    if args.apply and not args.archive:
        parser.error("--apply requires --archive <dir>")
    if args.archive:
        check_archive(args.archive)
    sys.path.insert(0, str(ROOT / "src"))
    selected = set(args.area or AREAS)
    pass2 = selected & {"layers", "research", "outputs", "tables", "downloads", "runs"}
    plan = build_deploy_plan() if pass2 else None
    max_age = args.nflverse_max_age
    if max_age is None:
        from engine.config.settings import Settings

        max_age = float(Settings.model_fields["nflverse_cache_duration"].default)
    research_inputs = args.research_inputs
    deploy_runs = frozenset(
        p for p in (plan.deploy.reasons if plan else {}) if p.parent.name == "runs"
    )
    builders = {
        "league": lambda: plan_league(args.league_receipts),
        "cache": plan_main_cache,
        "revisions": lambda: plan_revisions(args.keep_revisions),
        "captures": lambda: plan_captures(args.injury_hours, args.official_hours),
        "evaluations": plan_evaluations,
        "runs": lambda: plan_runs(deploy_runs, research_inputs),
        "layers": lambda: plan_layers(plan, research_inputs),
        "research": lambda: plan_research_dirs(plan, research_inputs),
        "outputs": lambda: plan_outputs(plan, research_inputs),
        "tables": lambda: plan_main_tables(plan, research_inputs),
        "downloads": lambda: plan_download_caches(plan, research_inputs, max_age),
    }
    areas = [builder() for name, builder in builders.items() if name in selected]
    print(f"{'APPLY' if args.apply else 'DRY RUN'}: {ROOT}")
    summarize(areas, args.list)
    if args.apply:
        apply(areas, args.archive)
    else:
        print("\nDry run only; nothing moved. Use --apply --archive <dir outside data/>.")


if __name__ == "__main__":
    main()
