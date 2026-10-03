"""Shared read-only inputs, fixed learners, and paired evidence for Protocol 2."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
from data import FAMILIES, MARKET, finite, make_panel
from evaluation import paired_summary
from methods import matrix
from safety import sha256, snapshot, verify_source_pins, write_json
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from engine.data.releases import load_gold

LAB = Path(__file__).resolve().parent
REPO = LAB.parents[1]
OLD = LAB / "runs/literature_002"
CONFIG = json.loads((LAB / "config.json").read_text())
PARAMS = CONFIG["booster"]


class Inputs:
    def __init__(self):
        verify_source_pins(REPO, CONFIG["source_sha256"])
        self.release = load_gold(REPO / "data", version=CONFIG["gold_version"])
        if self.release.ref["manifest_sha256"] != CONFIG["gold_manifest_sha256"]:
            raise ValueError("Gold pin mismatch")
        self.hashes = {}
        self.frames = {}
        self.bind(self.release.root / "manifest.json")
        for name in CONFIG["source_sha256"]:
            self.bind(REPO / name)

    def bind(self, path, expected=None):
        path = Path(path).resolve()
        value = sha256(path)
        if expected is not None and expected != value:
            raise ValueError(f"Input digest mismatch: {path}")
        key = str(path.relative_to(REPO))
        if key in self.hashes and self.hashes[key] != value:
            raise ValueError(f"Input changed during reading: {path}")
        self.hashes[key] = value
        return path

    def gold(self, name):
        if name not in self.frames:
            self.frames[name] = pl.read_parquet(self.bind(self.release.path(name)))
        return self.frames[name]

    def previous(self):
        manifest = json.loads(self.bind(OLD / "manifest.json").read_text())
        if manifest["status"] != "complete" or not manifest["protected_inputs_unchanged"]:
            raise ValueError("Cannot reuse an incomplete/invalid trial")
        return pl.read_parquet(
            self.bind(OLD / "predictions.parquet", manifest["outputs"]["predictions.parquet"])
        )

    def panel(self):
        rows = make_panel(
            self.gold("preseason_features"),
            self.gold("season_outcomes"),
            self.gold("nfl_player_weeks"),
        )
        for row in rows:
            for family in FAMILIES[row["position"]]:
                row[f"raw_{family}"] = row[f"rate_{family}"]
                n = row[f"exposure_{family}"]
                row[f"n_{family}"] = float(np.log1p(n)) if n is not None else None
        return [r for r in rows if r["outcome_complete"] and r["forecast_season"] >= 2004]

    def raw_providers(self):
        path = REPO / "data/raw/snapshots/canonical_20260923_r4/manifest.json"
        manifest = json.loads(self.bind(path).read_text())
        found = {}
        for key, asset in manifest["assets"].items():
            if not key.startswith("history/cache/nflverse/") or not key.endswith(".parquet"):
                continue
            p = REPO / "data" / asset["object"]
            cols = set(pl.read_parquet_schema(p))
            kind = (
                "passing"
                if "completion_percentage_above_expectation" in cols
                else "receiving"
                if "avg_separation" in cols
                else "rushing"
                if "rush_yards_over_expected_per_att" in cols
                else "schedule"
                if "home_qb_id" in cols
                else None
            )
            if kind:
                if kind in found:
                    raise ValueError(f"Ambiguous provider snapshot: {kind}")
                found[kind] = pl.read_parquet(self.bind(p, asset["sha256"]))
        return found

    def verify(self):
        after = snapshot([REPO / name for name in self.hashes], REPO, missing_ok=True)
        changed = [name for name, value in self.hashes.items() if value != after.get(name)]
        if changed:
            raise ValueError(f"Experiment dependencies changed: {changed}")


def columns(rows, *, raw=True, market=True):
    result = sorted(c for c in rows[0] if c.startswith("x_"))
    if raw:
        result += [c for f in FAMILIES[rows[0]["position"]] for c in (f"raw_{f}", f"n_{f}")]
    return result + (MARKET if market else [])


def predict(train, test, label, features, *, kind="mean", quantile=None, lower=0):
    train = [r for r in train if finite(r.get(label)) is not None]
    if not train:
        raise ValueError(f"No earlier training labels for {label}")
    x, z = matrix(train, features), matrix(test, features)
    y = np.asarray([r[label] for r in train], dtype=float)
    varying = [i for i in range(x.shape[1]) if len(np.unique(x[np.isfinite(x[:, i]), i])) > 1]
    if not varying or len(train) < 30 or (kind == "probability" and len(np.unique(y)) == 1):
        value = np.quantile(y, quantile) if quantile else y.mean()
        result = np.full(len(test), value)
    else:
        with threadpool_limits(limits=1):
            if kind == "probability":
                model = HistGradientBoostingClassifier(**PARAMS).fit(x[:, varying], y)
                result = model.predict_proba(z[:, varying])[:, list(model.classes_).index(1)]
            else:
                extra = {"loss": "quantile", "quantile": quantile} if quantile else {}
                model = HistGradientBoostingRegressor(**PARAMS, **extra).fit(x[:, varying], y)
                result = model.predict(z[:, varying])
    if lower is not None:
        result = np.maximum(result, lower)
    return np.clip(result, 0, 1) if kind == "probability" else result


def record(row, target, model, control, actual, prediction, baseline, **extra):
    return dict(
        player_id=row.get("player_id"),
        position=row["position"],
        target=target,
        model=model,
        control=control,
        season=row["forecast_season"],
        population=row.get("player_population", "all"),
        actual=float(actual),
        prediction=float(prediction),
        control_prediction=float(baseline),
        **extra,
    )


def sign_test(gains, seed=20260923):
    values = np.asarray(gains, dtype=float)
    if len(values) < 5 or not np.any(values):
        return 1.0
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(10000, len(values)))
    null = (signs * values).mean(axis=1)
    return float((1 + (null >= values.mean() - 1e-12).sum()) / (len(null) + 1))


def holm(summaries, field="p_value"):
    primary = [r for r in summaries if r.get("slice", "all") == "all"]
    order = sorted(primary, key=lambda r: r[field])
    ceiling = 0.0
    for index, row in enumerate(order):
        ceiling = max(ceiling, min(1.0, (len(order) - index) * row[field]))
        row["holm_p"] = ceiling
        row["holm_family_size"] = len(order)
    return summaries


def point_summaries(rows):
    groups = {}
    for r in rows:
        names = {"all", r.get("population", "all")}
        if r["season"] >= 2017:
            names.add("modern")
        for name in sorted(names):
            key = (r["position"], r["target"], r["model"], r["control"], name)
            groups.setdefault(key, []).append(r)
    results = []
    for (position, target, model, control, name), values in sorted(groups.items()):
        result = paired_summary(values, CONFIG)
        result.update(
            position=position,
            target=target,
            model=model,
            control=control,
            slice=name,
            p_value=sign_test([a["control_mse"] - a["mse"] for a in result["annual"]]),
        )
        results.append(result)
    return holm(results)


def save_points(root, name, rows):
    pl.DataFrame(rows, infer_schema_length=None).write_parquet(root / f"{name}_predictions.parquet")
    results = point_summaries(rows)
    write_json(root / f"{name}_summaries.json", results)
    return results
