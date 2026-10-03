"""Promote selected application outputs through the ordinary table publisher.

Research runs are never discovered by directory scanning. Only outputs bound by
the selected application/weekly manifests are promoted. Original artifacts stay
unchanged; JSON records retain exact payloads, and Parquet retains column types.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from engine.data.releases import current_catalog, digest, identifier, inside
from engine.tables.registry import TableSpec, table_name

ROW = "_record_number"
PAYLOAD = "_record_json"


def selection(data: Path) -> dict:
    catalog = current_catalog(data)
    if catalog is None:
        return {}
    weekly = data / "weekly/current.json"
    return {
        "source": catalog["gold"],
        "products": catalog["products"],
        "weekly": json.loads(weekly.read_text()) if weekly.exists() else None,
        "history": catalog.get("previous_catalog_sha256"),
    }


def recipes(data: Path) -> dict[str, TableSpec]:
    selected = selection(data)
    if not selected:
        return {}
    products = dict(selected["products"])
    if selected["weekly"]:
        products["weekly"] = selected["weekly"]
    pointer = selected["history"]
    seen = set()
    while pointer:
        if pointer in seen:
            raise ValueError("Cyclic publication history")
        seen.add(pointer)
        path = data / "catalog_history" / (identifier(pointer) + ".json")
        if digest(path) != pointer:
            raise ValueError("Historical catalog changed")
        previous = json.loads(path.read_text())
        ref = previous.get("products", {}).get("analysis")
        if ref:
            products["history_" + ref["manifest_sha256"][:16]] = ref
        pointer = previous.get("previous_catalog_sha256")
    result = {}
    for product, ref in products.items():
        if product not in {"analysis", "profiles", "college", "weekly"} and not product.startswith(
            "history_"
        ):
            continue
        root = data / "research" / identifier(ref["version"])
        manifest_path = root / "manifest.json"
        if digest(manifest_path) != ref["manifest_sha256"]:
            raise ValueError(f"Application manifest changed: {product}")
        manifest = json.loads(manifest_path.read_text())
        historical = product.startswith("history_")
        if manifest.get("status") != "complete" or (
            not historical and manifest.get("gold") != selected["source"]
        ):
            raise ValueError(f"Application product is incomplete or uses another source: {product}")
        if product == "weekly" and manifest["analysis"] != products["analysis"]:
            raise ValueError("Weekly product uses another analysis release")
        outputs = manifest.get("files", manifest.get("output_sha256", {}))
        report_name = "ranking_report.json" if "ranking_report.json" in outputs else "report.json"
        cutoff = {}
        if report_name in outputs:
            report_path = root / report_name
            if digest(report_path) != outputs[report_name]:
                raise ValueError(f"Application report changed: {product}")
            report = json.loads(report_path.read_text())
            cutoff = {key: report[key] for key in ("season", "through_week") if key in report}
        for filename, expected in sorted(outputs.items()):
            if product == "analysis" and filename not in {
                "players.parquet",
                "rankings.parquet",
                "forecasts.parquet",
                "predictions.parquet",
                "features.parquet",
                "evaluations.json",
                "registry.json",
                "incidents.json",
                "report.json",
                "ranking_report.json",
                "ranking_evaluations.json",
                "qb_passing/current.parquet",
                "qb_passing/report.json",
                "qb_passing/evaluations.json",
                "qb_passing/predictions.parquet",
                "qb_variations/decisions.json",
                "qb_variations/evaluations.json",
                "qb_variations/report.json",
                "qb_variations/ranking_predictions.parquet",
            }:
                continue
            if product == "weekly" and filename not in {"report.json", "predictions.parquet"}:
                continue
            if historical and filename not in {
                "rankings.parquet",
                "ranking_report.json",
                "registry.json",
                "incidents.json",
            }:
                continue
            path = inside(root, filename)
            # Nested research challengers, code and captured inputs remain research.
            if path.suffix not in {".parquet", ".json"} or (
                "/" in filename and not filename.startswith(("qb_passing/", "qb_variations/"))
            ):
                continue
            if filename.startswith("qb_passing/") and filename.count("/") > 1:
                continue
            if path.name in {"manifest.json", "started.json"}:
                continue
            if digest(path) != expected:
                raise ValueError(f"Application artifact changed: {product}/{filename}")
            name = table_name("app_" + product + "_" + filename.rsplit(".", 1)[0].replace("/", "_"))
            if name in result:
                raise ValueError(f"Application table name collision: {name}")
            relative = path.relative_to(data).as_posix()
            result[name] = TableSpec(
                description=f"Published {product} output: {filename}.",
                grain="One source record, ordered by _record_number within this release.",
                primary_key=[ROW],
                kind="python",
                builder="engine.tables.products:build_product",
                refresh_hours=24,
                min_rows=0,
                observation_cutoff=cutoff,
                inputs=[relative, manifest_path.relative_to(data).as_posix()],
                parameters={
                    "artifact": relative,
                    "sha256": expected,
                    "product": ref,
                    "source_release": manifest["gold"],
                    "json_array": isinstance(json.loads(path.read_text()), list)
                    if path.suffix == ".json"
                    else False,
                },
                limitations=["Published model eligibility and observation cutoffs still apply."],
            )
    return result


def build_product(context) -> pl.DataFrame:
    path = inside(context.data_dir, context.parameters["artifact"])
    if digest(path) != context.parameters["sha256"]:
        raise ValueError(f"Application input changed: {path.name}")
    if path.suffix == ".parquet":
        frame = pl.read_parquet(path)
        if ROW in frame.columns:
            raise ValueError(f"Reserved application column: {ROW}")
        return frame.with_row_index(ROW).with_columns(pl.col(ROW).cast(pl.Int64))
    value = json.loads(path.read_text())
    records = value if isinstance(value, list) else [value]
    # JSON stays lossless (including missing keys vs explicit nulls). Consumers
    # can query fields with DuckDB JSON operators, without Python deserialization.
    return pl.DataFrame(
        {ROW: range(len(records)), PAYLOAD: [json.dumps(r, allow_nan=False) for r in records]},
        schema={ROW: pl.Int64, PAYLOAD: pl.String},
    )
