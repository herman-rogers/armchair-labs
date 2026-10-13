"""Compare sealed campaign predictions with external forecasts on matched outcomes.

No fitting, model selection, source fetching, or campaign writes happen here.
Source adapters and the evaluator exchange the versioned JSON contract below.
"""

from __future__ import annotations

import hashlib
import html
import io
import json
import math
import shutil
import tempfile
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

from engine.chairstats import CampaignStore
from engine.chairstats.store import _inside, safe_id

SKILL_SCORING = "opening_drive_skill_ppr_v1"
QB_SCORING = "opening_drive_qb_v1"
SCORING = {
    SKILL_SCORING: {
        "receptions": 1,
        "rushing_yards": 0.1,
        "receiving_yards": 0.1,
        "rushing_tds": 6,
        "receiving_tds": 6,
    },
    QB_SCORING: {
        "passing_yards": 0.04,
        "passing_tds": 4,
        "passing_interceptions": -2,
        "rushing_yards": 0.1,
        "rushing_tds": 6,
    },
}
KEYS = ("season", "origin_week", "end_week", "horizon", "position", "player_id", "target")
TOP_K = {"QB": 12, "RB": 24, "WR": 24, "TE": 12}


def digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def read_json(path: Path) -> dict:
    def invalid(value):
        raise ValueError(f"Non-finite JSON: {value}")

    return json.loads(path.read_bytes(), parse_constant=invalid)


def timestamp(value: str) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("Timestamps must include a timezone")
    return result.astimezone(UTC)


def unit(target: str) -> str:
    if target == "points":
        return "fantasy_points"
    if target.endswith("_yards"):
        return "yards"
    if target in {
        "attempts",
        "completions",
        "passing_tds",
        "passing_interceptions",
        "carries",
        "rushing_tds",
        "targets",
        "receptions",
        "receiving_tds",
    }:
        return "count"
    raise ValueError(f"Undeclared target unit: {target}")


def checked_bytes(root: Path, ref: dict) -> bytes:
    content = _inside(root, ref["path"]).read_bytes()
    if digest(content) != ref["sha256"]:
        raise ValueError(f"Artifact checksum mismatch: {ref['path']}")
    return content


def validate_source(source: dict) -> None:
    if source.get("schema_version") != 1:
        raise ValueError("Unsupported benchmark source schema")
    safe_id(source["source_id"])
    if source.get("kind") not in {"projection", "ranking"}:
        raise ValueError("Source kind must be projection or ranking")
    if not source.get("name") or not source.get("limitations"):
        raise ValueError("Source name and limitations must be explicit")
    seen = set()
    for row in source["rows"]:
        key = tuple(row[k] for k in KEYS)
        if key in seen:
            raise ValueError("Duplicate source player/season/horizon/target; select one snapshot")
        seen.add(key)
        if not row["player_id"] or row["position"] not in TOP_K:
            raise ValueError("Source requires exact player IDs and supported positions")
        if any(type(row[k]) is not int for k in ("season", "origin_week", "end_week")):
            raise ValueError("Season and weeks must be integers")
        if not 0 <= row["origin_week"] < row["end_week"] <= 18:
            raise ValueError("Invalid forecast window")
        if row["horizon"] not in {"season", "remaining"}:
            raise ValueError("Only season and remaining totals are supported")
        if row["horizon"] == "season" and row["origin_week"] != 0:
            raise ValueError("Season forecasts must have origin zero")
        value = row["prediction"]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError("External predictions must be finite numbers")
        if source["kind"] == "ranking" and (value <= 0 or row["target"] != "points"):
            raise ValueError("Rankings require positive ranks of fantasy points")
        if row.get("unit") != ("rank" if source["kind"] == "ranking" else unit(row["target"])):
            raise ValueError("External target unit mismatch")
        if row["target"] == "points" and not row.get("scoring"):
            raise ValueError("Points and rankings require an explicit scoring contract")
        if row.get("vintage") not in {"archived", "reconstructed"}:
            raise ValueError("Declare archived or reconstructed source vintage")
        if not row.get("source_url"):
            raise ValueError("Each external row requires source provenance")
        for k in ("available_at", "issued_at"):
            if row.get(k):
                timestamp(row[k])
        if row["vintage"] == "archived" and not row.get("available_at"):
            raise ValueError("Archived forecasts require a witnessed availability timestamp")
        if (
            row.get("issued_at")
            and row.get("available_at")
            and timestamp(row["issued_at"]) > timestamp(row["available_at"])
        ):
            raise ValueError("Issue time follows archive availability")


def publish_source(output: Path, source: dict, raw: dict[str, bytes] | None = None) -> Path:
    """Seal normalized rows and original receipts together; never overwrite."""
    validate_source(source)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as directory:
        stage = Path(directory) / "source"
        stage.mkdir()
        refs = []
        for name, content in (raw or {}).items():
            path = _inside(stage, name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            refs.append({"path": name, "sha256": digest(content)})
        payload = {**source, "raw_files": refs}
        content = json.dumps(payload, indent=2, allow_nan=False).encode()
        (stage / "source.json").write_bytes(content)
        (stage / "source.sha256").write_text(digest(content) + "\n")
        stage.rename(output)
    return output / "source.json"


def load_source(path: Path) -> dict:
    content = path.read_bytes()
    if digest(content) != path.with_suffix(".sha256").read_text().strip():
        raise ValueError("Benchmark source checksum mismatch")
    source = read_json(path)
    validate_source(source)
    for ref in source.get("raw_files", []):
        checked_bytes(path.parent, ref)
    return {**source, "sha256": digest(content)}


def load_calendar(path: Path | None) -> dict:
    if path is None:
        return {}
    result = {}
    for row in read_json(path)["rows"]:
        key = (row["season"], row["origin_week"])
        if key in result:
            raise ValueError("Duplicate calendar cutoff")
        timestamp(row["as_of"])
        result[key] = row
    return result


def campaign_rows(store: CampaignStore, campaign: dict, model: dict, calendar: dict):
    """Known legacy adapters, or an explicit sealed benchmark_evaluation contract.

    The adapter fixes the already selected recipe, never chooses the best result.
    Unrecognized bundles are reported as unsupported instead of guessed.
    """
    root = store.root / campaign["campaign_id"]
    contract = model.get("benchmark_evaluation")
    legacy = campaign["campaign_id"] in {
        "opening_drive_v1",
        "opening_drive_v2",
        "opening_drive_v2_te",
        "scout_team_v1",
    }
    if contract is None and not legacy:
        return [], "No benchmark_evaluation adapter declared", []
    artifact_key = (contract or {}).get("artifact", "evaluation_predictions")
    ref = model.get("artifacts", {}).get(artifact_key)
    if ref is None:
        return [], "No supported saved evaluation artifact", []
    frame = pl.read_parquet(io.BytesIO(checked_bytes(root, ref)))
    recipe = (contract or {}).get("recipe", model.get("primary_policy", "ensemble"))
    required = {"year", "origin", "player_id", "target", "prediction", "actual", "recipe"}
    if not required.issubset(frame.columns):
        return [], "Evaluation schema requires a dedicated adapter", [ref]
    frame = frame.filter(pl.col("recipe") == recipe)
    if frame.is_empty():
        return [], f"Saved recipe {recipe} is absent", [ref]
    cutoffs = dict(calendar)
    refs = [ref]
    audit = model.get("artifacts", {}).get("historical_candidate_audit")
    if audit:
        af = pl.read_parquet(io.BytesIO(checked_bytes(root, audit)))
        refs.append(audit)
        if "availability_as_of" in af.columns:
            for row in af.select("year", "origin", "availability_as_of").unique().to_dicts():
                key = (row["year"], row["origin"])
                new = row["availability_as_of"]
                if key in cutoffs and timestamp(cutoffs[key]["as_of"]) != timestamp(new):
                    raise ValueError("Calendar disagrees with sealed campaign cutoff")
                cutoffs[key] = {"as_of": new, "end_week": 18 if row["year"] >= 2021 else 17}
    for row in (contract or {}).get("cutoffs", []):
        cutoffs[(row["season"], row["origin_week"])] = row
    seen, rows = set(), []
    for row in frame.to_dicts():
        if row["target"] not in model["targets"] or row["origin"] != model["origin_week"]:
            raise ValueError("Saved evaluation differs from model target/origin contract")
        evaluation_key = (row["year"], row["origin"], row["player_id"], row["target"])
        if evaluation_key in seen:
            raise ValueError("Duplicate campaign evaluation keys")
        seen.add(evaluation_key)
        cutoff = cutoffs.get((row["year"], row["origin"]))
        scoring = (contract or {}).get("scoring")
        if scoring is None and legacy:
            scoring = QB_SCORING if model["position"] == "QB" else SKILL_SCORING
        rows.append(
            {
                "season": row["year"],
                "origin_week": row["origin"],
                "end_week": cutoff["end_week"] if cutoff else None,
                "horizon": model["horizon"],
                "position": model["position"],
                "player_id": row["player_id"],
                "target": row["target"],
                "prediction": row["prediction"],
                "actual": row["actual"],
                "as_of": cutoff["as_of"] if cutoff else None,
                "unit": unit(row["target"]),
                "scoring": scoring,
            }
        )
    return rows, None, refs


def compatibility(model: dict, external: dict, kind: str, max_age_days: int):
    if not model["as_of"]:
        return None, "missing_campaign_cutoff"
    for k in ("season", "origin_week", "end_week", "horizon", "position", "target"):
        if model[k] != external[k]:
            return None, f"incompatible_{k}"
    cutoff = timestamp(model["as_of"])
    for key in ("issued_at", "available_at"):
        if external.get(key) and timestamp(external[key]) > cutoff:
            return None, "external_after_cutoff"
    stamp = external.get("issued_at") or external.get("available_at")
    if stamp and (cutoff - timestamp(stamp)).total_seconds() > max_age_days * 86400:
        return None, "external_too_old"
    if kind == "projection" and (
        model["unit"] != external["unit"]
        or (model["target"] == "points" and model["scoring"] != external["scoring"])
    ):
        return None, "incompatible_scoring_or_unit"
    labels = []
    if external["vintage"] != "archived":
        labels.append("reconstructed_external_vintage")
    if kind == "ranking" and model["scoring"] != external["scoring"]:
        labels.append("ranking_scoring_difference")
    return "+".join(labels) or "archived_by_cutoff", None


def spearman(actual, prediction):
    a = pl.Series(actual).rank(method="average").to_numpy()
    p = pl.Series(prediction).rank(method="average").to_numpy()
    if len(a) < 2 or not np.std(a) or not np.std(p):
        return None
    return float(np.corrcoef(a, p)[0, 1])


def metrics(rows: list[dict], kind: str) -> dict:
    actual = np.array([r["actual"] for r in rows])
    result: dict = {"n": len(rows)}
    for name in ("model", "external"):
        prediction = np.array([r[name] for r in rows])
        if kind == "projection":
            errors = prediction - actual
            result[name] = {
                "mae": float(np.abs(errors).mean()),
                "rmse": float(np.sqrt(np.mean(errors**2))),
                "mse": float(np.mean(errors**2)),
                "bias": float(errors.mean()),
            }
    if kind == "projection":
        result["mae_gain"] = result["external"]["mae"] - result["model"]["mae"]
        denom = result["external"]["mse"]
        result["mse_reduction_pct"] = 100 * (1 - result["model"]["mse"] / denom) if denom else None
    # Ranking and selection are evaluated within a season, never across seasons.
    if len({r["season"] for r in rows}) == 1:
        k = min(TOP_K[rows[0]["position"]], len(rows))
        result["top_k"] = k
        for name in ("model", "external"):
            ascending = kind == "ranking" and name == "external"
            pred = [(-r[name] if ascending else r[name]) for r in rows]
            result[f"{name}_spearman"] = spearman(actual, pred)
            selected = sorted(
                rows, key=lambda r: (r[name] if ascending else -r[name], r["player_id"])
            )[:k]
            result[f"{name}_selected_actual"] = sum(r["actual"] for r in selected)
        result["selected_actual_gain"] = (
            result["model_selected_actual"] - result["external_selected_actual"]
        )
    return result


def summarize(rows: list[dict], kind: str) -> dict:
    result = metrics(rows, kind)
    years = sorted({r["season"] for r in rows})
    annual = [metrics([r for r in rows if r["season"] == year], kind) for year in years]
    result["seasons"] = years
    result["season_count"] = len(years)
    for key in ("model_spearman", "external_spearman", "selected_actual_gain"):
        values = [a[key] for a in annual if a.get(key) is not None]
        result[f"mean_annual_{key}"] = float(np.mean(values)) if values else None
    if kind == "projection":
        gains = np.array([a["mae_gain"] for a in annual])
        result["equal_season_mae_gain"] = float(gains.mean())
        # Season clusters, not independent repeated-player rows. Exploratory only.
        result["equal_season_mae_gain_ci95"] = None
        if len(years) >= 3:
            boot = np.random.default_rng(0).choice(gains, size=(2000, len(years))).mean(axis=1)
            result["equal_season_mae_gain_ci95"] = np.quantile(boot, [0.025, 0.975]).tolist()
    return result


def evaluate(
    root: Path,
    campaign_id: str,
    sources: list[Path],
    output: Path,
    calendar_path: Path | None = None,
    max_age_days: int = 14,
) -> dict:
    if not sources or max_age_days < 0:
        raise ValueError("Provide sources and a nonnegative maximum source age")
    if output.exists():
        raise FileExistsError(output)
    if output.resolve().is_relative_to(root.resolve()):
        raise ValueError("Evaluation output must be outside the campaign registry")
    store = CampaignStore(root)
    campaign = store.campaign(campaign_id)
    calendar = load_calendar(calendar_path)
    external = [load_source(p) for p in sources]
    if len({s["source_id"] for s in external}) != len(external):
        raise ValueError("Duplicate external source IDs")
    report = {
        "schema_version": 1,
        "purpose": "external_evaluation_only",
        "campaign_id": campaign_id,
        "campaign_manifest_sha256": campaign["manifest_sha256"],
        "created_at": datetime.now(UTC).isoformat(),
        "max_age_days": max_age_days,
        "limitations": [
            "Historical development evidence; no state-of-the-art or prospective claim.",
            "Each source uses its own matched population; this is not a common leaderboard.",
            "Reconstructed external vintages cannot establish original forecast availability.",
            "Rankings have no point-error metrics; scoring differences are contextual comparisons.",
            "Selection is restricted to matched players; missing players are never assigned zero.",
            "Season-bootstrap intervals are exploratory with few seasons and repeated players.",
        ],
        "sources": [{k: v for k, v in s.items() if k != "rows"} for s in external],
        "models": [],
        "comparisons": [],
        "coverage": [],
    }
    pairs, excluded = [], []
    for model in campaign["models"]:
        rows, reason, refs = campaign_rows(store, campaign, model, calendar)
        report["models"].append(
            {
                "model_id": model["model_id"],
                "reason": reason,
                "evaluation_rows": len(rows),
                "artifacts": refs,
            }
        )
        if reason:
            continue
        blocks = defaultdict(list)
        for row in rows:
            blocks[(row["season"], row["target"])].append(row)
        for source in external:
            # Keep incompatible windows in the index so exclusions explain them.
            lookup = defaultdict(list)
            for row in source["rows"]:
                lookup[(row["season"], row["position"], row["player_id"], row["target"])].append(
                    row
                )
            for (season, target), block in sorted(blocks.items()):
                counts: Counter[str] = Counter()
                matched_ids = set()
                for row in block:
                    candidates = lookup.get((season, row["position"], row["player_id"], target), [])
                    valid = []
                    reasons = []
                    for candidate in candidates:
                        label, rejection = compatibility(
                            row, candidate, source["kind"], max_age_days
                        )
                        if rejection:
                            reasons.append(rejection)
                        else:
                            valid.append((candidate, label))
                    if len(valid) > 1:
                        raise ValueError("Ambiguous compatible external forecasts")
                    rejection = None
                    if not all(
                        isinstance(row[k], (int, float)) and math.isfinite(row[k])
                        for k in ("prediction", "actual")
                    ):
                        rejection = "unknown_outcome_or_prediction"
                    elif not valid:
                        rejection = ",".join(sorted(set(reasons))) or "missing_external"
                    common = {
                        "model_id": model["model_id"],
                        "source_id": source["source_id"],
                        **{k: row[k] for k in KEYS},
                    }
                    if rejection:
                        counts[rejection] += 1
                        excluded.append({**common, "reason": rejection})
                        continue
                    candidate, label = valid[0]
                    matched_ids.add(row["player_id"])
                    pairs.append(
                        {
                            **common,
                            "comparison_class": label,
                            "actual": row["actual"],
                            "model": row["prediction"],
                            "external": candidate["prediction"],
                            "as_of": row["as_of"],
                            "external_available_at": candidate.get("available_at"),
                            "external_issued_at": candidate.get("issued_at"),
                            "source_url": candidate["source_url"],
                        }
                    )
                eligible_external = [
                    r
                    for r in source["rows"]
                    if r["season"] == season
                    and r["position"] == model["position"]
                    and r["target"] == target
                    and compatibility(block[0], r, source["kind"], max_age_days)[1] is None
                ]
                external_ids = {r["player_id"] for r in eligible_external}
                external_top = sorted(
                    eligible_external,
                    key=lambda r: (
                        r["prediction"] if source["kind"] == "ranking" else -r["prediction"],
                        r["player_id"],
                    ),
                )[: TOP_K[model["position"]]]
                model_ids = {r["player_id"] for r in block}
                top = sorted(
                    (
                        r
                        for r in block
                        if r["prediction"] is not None and math.isfinite(r["prediction"])
                    ),
                    key=lambda r: (-r["prediction"], r["player_id"]),
                )[: TOP_K[model["position"]]]
                report["coverage"].append(
                    {
                        "model_id": model["model_id"],
                        "source_id": source["source_id"],
                        "season": season,
                        "target": target,
                        "model_rows": len(block),
                        "matched_rows": len(matched_ids),
                        "matched_fraction": len(matched_ids) / len(block),
                        "external_only_ids": sorted(external_ids - model_ids),
                        "model_top_k": len(top),
                        "model_top_k_missing": [
                            r["player_id"] for r in top if r["player_id"] not in matched_ids
                        ],
                        "external_top_k": len(external_top),
                        "external_top_k_missing": [
                            r["player_id"]
                            for r in external_top
                            if r["player_id"] not in matched_ids
                        ],
                        "excluded": dict(counts),
                    }
                )
    grouped = defaultdict(list)
    for row in pairs:
        grouped[(row["model_id"], row["source_id"], row["target"], row["comparison_class"])].append(
            row
        )
    kinds = {s["source_id"]: s["kind"] for s in external}
    for (model_id, source_id, target, label), rows in sorted(grouped.items()):
        for season in [None, *sorted({r["season"] for r in rows})]:
            block = rows if season is None else [r for r in rows if r["season"] == season]
            # Union of each forecaster's top 2K, chosen without seeing actual outcomes.
            top_ids: set[tuple[int, str]] = set()
            for year in {r["season"] for r in block}:
                yr = [r for r in block if r["season"] == year]
                for name in ("model", "external"):
                    asc = name == "external" and kinds[source_id] == "ranking"
                    top = sorted(yr, key=lambda r: (r[name] if asc else -r[name], r["player_id"]))
                    top_ids.update(
                        (year, r["player_id"]) for r in top[: 2 * TOP_K[yr[0]["position"]]]
                    )
            for cohort, part in [
                ("matched", block),
                (
                    "forecast_top_2k_union",
                    [r for r in block if (r["season"], r["player_id"]) in top_ids],
                ),
            ]:
                report["comparisons"].append(
                    {
                        "model_id": model_id,
                        "source_id": source_id,
                        "target": target,
                        "comparison_class": label,
                        "kind": kinds[source_id],
                        "season": season,
                        "cohort": cohort,
                        **summarize(part, kinds[source_id]),
                    }
                )
    report["status"] = "evaluated" if pairs else "no_comparable_rows"
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as directory:
        stage = Path(directory) / "report"
        stage.mkdir()
        (stage / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
        for name, records in (
            ("paired_rows", pairs),
            ("exclusions", excluded),
            ("coverage", report["coverage"]),
        ):
            (stage / f"{name}.json").write_text(json.dumps(records, indent=2, allow_nan=False))
        if pairs:
            pl.DataFrame(pairs, infer_schema_length=None).write_csv(stage / "paired_rows.csv")
        (stage / "index.html").write_text(render(report))
        inputs = stage / "inputs"
        inputs.mkdir()
        manifest_bytes = (root / campaign_id / "manifest.json").read_bytes()
        if digest(manifest_bytes) != campaign["manifest_sha256"]:
            raise ValueError("Campaign revision changed during evaluation; rerun explicitly")
        (inputs / "campaign_manifest.json").write_bytes(manifest_bytes)
        if calendar_path:
            shutil.copyfile(calendar_path, inputs / "calendar.json")
        for path, source in zip(sources, external, strict=True):
            if load_source(path)["sha256"] != source["sha256"]:
                raise ValueError("External source changed during evaluation")
            shutil.copytree(path.parent, inputs / source["source_id"])
        for path in Path(__file__).parent.glob("*.py"):
            shutil.copyfile(path, inputs / path.name)
        manifest = {
            str(p.relative_to(stage)): digest(p.read_bytes())
            for p in sorted(stage.rglob("*"))
            if p.is_file()
        }
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2))
        stage.rename(output)
    return report


def render(report: dict) -> str:
    def esc(value):
        return html.escape(str(value))

    def fmt(value):
        return "—" if value is None else f"{value:,.2f}"

    rows, points_summary = [], []
    for c in report["comparisons"]:
        if c["cohort"] != "matched":
            continue
        values = [
            c["model_id"],
            c["source_id"],
            c["target"],
            c["season"] or "Pooled",
            c["comparison_class"],
            c["n"],
            fmt(c.get("model", {}).get("mae")),
            fmt(c.get("external", {}).get("mae")),
            fmt(c.get("model", {}).get("rmse")),
            fmt(c.get("external", {}).get("rmse")),
            fmt(c.get("mean_annual_model_spearman")),
            fmt(c.get("mean_annual_external_spearman")),
            fmt(c.get("mean_annual_selected_actual_gain")),
        ]
        rendered = "<tr>" + "".join(f"<td>{esc(v)}</td>" for v in values) + "</tr>"
        rows.append(rendered)
        if c["target"] == "points" and c["season"] is None:
            points_summary.append(rendered)
    headers = [
        "Model",
        "Source",
        "Target",
        "Season",
        "Comparison",
        "N",
        "Our MAE",
        "External MAE",
        "Our RMSE",
        "External RMSE",
        "Our rank correlation",
        "External rank correlation",
        "Selection gain / year",
    ]
    coverage = []
    for c in report["coverage"]:
        if c["target"] == "points" or c["matched_rows"] == 0:
            coverage.append(
                f"<li>{esc(c['model_id'])} / {esc(c['source_id'])} / "
                f"{c['season']} / {esc(c['target'])}: "
                f"{c['matched_rows']}/{c['model_rows']} matched; "
                f"{len(c['model_top_k_missing'])}/{c['model_top_k']} model top picks "
                f"missing; {len(c['external_top_k_missing'])}/{c['external_top_k']} "
                f"external top picks missing. {esc(c['excluded'])}</li>"
            )
    unsupported = [
        f"<li>{esc(m['model_id'])}: {esc(m['reason'])}</li>"
        for m in report["models"]
        if m["reason"]
    ]
    table_start = (
        "<div class='table'><table><tr>" + "".join(f"<th>{esc(v)}</th>" for v in headers) + "</tr>"
    )
    source_notes = "".join(
        f"<h3>{esc(s['name'])}</h3><ul>"
        + "".join(f"<li>{esc(v)}</li>" for v in s["limitations"])
        + "</ul>"
        for s in report["sources"]
    )
    return (
        "<!doctype html><html lang='en'><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>Campaign external evaluation</title><style>"
        "body{font:15px system-ui;margin:2rem;color:#18212f}table{border-collapse:collapse}"
        "th,td{padding:.5rem;border:1px solid #ccd4df;text-align:right}"
        "th{background:#edf2f8}li{margin:.5rem 0}.table{overflow:auto}"
        "details{margin:1.5rem 0}summary{cursor:pointer;font-weight:600}"
        "</style><h1>" + esc(report["campaign_id"]) + " · external evaluation</h1>"
        "<p>Evaluation only. Lower MAE/RMSE and higher rank correlation are better. "
        "Positive selection gain favors our forecast on the shared player pool.</p>"
        "<p>Pooled rank correlations and selection gains are equal-season averages. "
        "Errors are horizon totals, not weekly errors. Compare sources only on identical rows.</p>"
        "<p>Campaign revision: <code>" + esc(report["campaign_manifest_sha256"]) + "</code></p>"
        "<ul>" + "".join(f"<li>{esc(v)}</li>" for v in report["limitations"]) + "</ul>"
        "<p><a href='report.json'>Full metrics, cohorts and provenance</a> · "
        "<a href='coverage.json'>Coverage</a> · <a href='exclusions.json'>Exclusions</a> · "
        "<a href='paired_rows.json'>Matched predictions</a></p>"
        "<h2>Fantasy points · pooled matched comparisons</h2>"
        + (
            table_start + "".join(points_summary) + "</table></div>"
            if points_summary
            else "<p>No comparable fantasy-point rows.</p>"
        )
        + "<details><summary>All targets and individual seasons</summary>"
        + table_start
        + "".join(rows)
        + "</table></div></details><details><summary>Source limitations</summary>"
        + source_notes
        + "</details><details><summary>Coverage and unavailable comparisons</summary><ul>"
        + "".join(unsupported + coverage)
        + "</ul></details></html>"
    )


def evaluate_all(
    root: Path,
    sources: list[Path],
    output: Path,
    calendar_path: Path | None = None,
    max_age_days: int = 14,
) -> dict:
    """Create one review set for all locally sealed campaigns, including unsupported cells."""
    if output.exists():
        raise FileExistsError(output)
    if output.resolve().is_relative_to(root.resolve()):
        raise ValueError("Evaluation output must be outside the campaign registry")
    campaigns = CampaignStore(root).campaigns()
    if not campaigns:
        raise ValueError("No saved campaigns to evaluate")
    output.parent.mkdir(parents=True, exist_ok=True)
    summary: dict = {"schema_version": 1, "purpose": "external_evaluation_only", "campaigns": []}
    with tempfile.TemporaryDirectory(dir=output.parent) as directory:
        stage = Path(directory) / "evaluations"
        stage.mkdir()
        for campaign in campaigns:
            cid = campaign["campaign_id"]
            report = evaluate(root, cid, sources, stage / cid, calendar_path, max_age_days)
            summary["campaigns"].append(
                {
                    "campaign_id": cid,
                    "manifest_sha256": report["campaign_manifest_sha256"],
                    "status": report["status"],
                    "report": f"{cid}/index.html",
                }
            )
        (stage / "summary.json").write_text(json.dumps(summary, indent=2))
        links = "".join(
            f"<li><a href='{html.escape(c['report'])}'>{html.escape(c['campaign_id'])}</a> "
            f"— {html.escape(c['status'])}</li>"
            for c in summary["campaigns"]
        )
        (stage / "index.html").write_text(
            "<!doctype html><html lang='en'><meta charset='utf-8'>"
            "<title>External campaign evaluations</title>"
            "<style>body{font:17px system-ui;max-width:850px;margin:3rem auto;padding:1rem}"
            "li{margin:1rem 0}</style><h1>External campaign evaluations</h1>"
            "<p>Each report compares saved predictions with external sources. "
            "No fitting or model selection. Source vintage, scoring and player coverage "
            "limit comparability; these reports do not establish a state-of-the-art ranking.</p>"
            f"<ul>{links}</ul><a href='summary.json'>Campaign revision receipts</a></html>"
        )
        stage.rename(output)
    return summary
