"""Broad fixed library with warm-start capacity paths and training-only discovery."""

import itertools
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

from ..models import MultiTarget, Recipe

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".packages"))


def paths(config):
    rng = np.random.default_rng(config["seed"])
    result = []
    # Matched feature-view controls, then broader structural/settings search.
    for i in range(24):
        result.append(
            dict(
                name=f"lgb_{i:02d}",
                family="lgb",
                view=["summary", "raw", "inventory", "all", "market", "basic"][i % 6],
                representation="identity"
                if i < 18
                else ["selected", "interactions", "leaf", "selected", "interactions", "leaf"][
                    i - 18
                ],
                learning_rate=0.05 if i < 6 else float(rng.choice([0.02, 0.05, 0.1])),
                leaves=15 if i < 6 else int(rng.choice([7, 15, 31, 63])),
                leaf_samples=25 if i < 6 else int(rng.choice([5, 15, 30, 60])),
                l2=10 if i < 6 else float(rng.choice([0, 1, 10, 100])),
                max_features=1.0 if i < 6 else float(rng.choice([0.5, 0.75, 1.0])),
                depth=None if i < 6 else rng.choice([3, 6, 12, None]),
                half_life=0 if i < 6 else int(rng.choice([0, 3, 7])),
                seed=config["seed"] + i,
            )
        )
    for i in range(8):
        path = dict(result[i + 6])
        path.update(name=f"xgb_{i:02d}", family="xgb", depth=[3, 5, 7, 4][i % 4])
        result.append(path)
    for i in range(4):
        path = dict(result[i + 2])
        path.update(name=f"cat_{i:02d}", family="cat", depth=[4, 6, 8, 5][i])
        result.append(path)
    for i in range(4):
        path = dict(result[i + 2])
        path.update(name=f"hist_{i:02d}", family="hist", representation="selected")
        result.append(path)
    for i in range(6):
        result.append(
            dict(
                name=f"forest_{i:02d}",
                family="extra" if i < 3 else "forest",
                view=["raw", "all", "market"][i % 3],
                representation="identity",
                leaf_samples=[5, 15, 30][i % 3],
                depth=[12, 8, 6][i % 3],
                max_features=["sqrt", 0.05, 0.1][i % 3],
                half_life=0,
                seed=config["seed"] + 100 + i,
            )
        )
    for i, family in enumerate(
        ["ridge", "svr", "kernel", "mlp", "pls", "hurdle", "neural_tree", "multi_extra"]
    ):
        result.append(
            dict(
                name=family,
                family=family,
                view="all",
                representation="selected"
                if family in {"hurdle", "multi_extra", "neural_tree"}
                else "identity",
                half_life=0,
                seed=config["seed"] + 200 + i,
            )
        )
    return result


def names_for(path, config):
    if path["family"] in {"hist", "lgb", "xgb", "cat", "extra", "forest"}:
        return [f"{path['name']}_t{n}" for n in config["tree_checkpoints"]]
    return [path["name"]]


class Discover:
    def __init__(self, method, seed):
        self.method, self.seed = method, seed

    def fit(self, x, y, weight):
        self.active = np.flatnonzero(np.any(np.isfinite(x), axis=0))
        z = x[:, self.active]
        self.med = np.nanmedian(z, axis=0)
        z = np.where(np.isfinite(z), z, self.med)
        self.selector = ExtraTreesRegressor(
            n_estimators=32,
            max_depth=5,
            min_samples_leaf=20,
            max_features="sqrt",
            n_jobs=1,
            random_state=self.seed,
        )
        self.selector.fit(z, y, sample_weight=weight)
        order = np.argsort(-self.selector.feature_importances_, kind="stable")
        self.selected = order[: min(128, len(order))]
        self.pairs = list(itertools.combinations_with_replacement(order[:16], 2))
        self.pairs = self.pairs[:136]
        self.scale = np.maximum(np.nanstd(z, axis=0), 1.0)
        self.evidence = {
            "selected_indices": self.active[self.selected].tolist(),
            "pairs": [[int(self.active[a]), int(self.active[b])] for a, b in self.pairs]
            if self.method == "interactions"
            else [],
        }
        return self

    def transform(self, x):
        z = x[:, self.active]
        z = np.where(np.isfinite(z), z, self.med)
        if self.method == "selected":
            return np.column_stack(
                [z[:, self.selected], ~np.isfinite(x[:, self.active[self.selected]])]
            )
        if self.method == "leaf":
            # Tree-leaf response coordinates summarize learned nonlinear partitions.
            encoded = np.column_stack([t.predict(z) for t in self.selector.estimators_])
            return np.column_stack([x, encoded])
        a = np.clip((z - self.med) / self.scale, -100, 100)
        products = np.column_stack([a[:, i] * a[:, j] for i, j in self.pairs])
        ratios = np.column_stack([a[:, i] / (1 + abs(a[:, j])) for i, j in self.pairs])
        return np.column_stack([x, products, ratios])


def fit_path(path, config, x, y, exposure, years, test_x, test_exposure):
    start = time.monotonic()
    target = y[:, 0] / exposure
    weight = np.ones(len(y))
    if path["half_life"]:
        weight = 2.0 ** ((years - years.max()) / path["half_life"])
    active = np.flatnonzero(np.any(np.isfinite(x), axis=0))
    x, test_x = x[:, active], test_x[:, active]
    evidence = {}
    if path["representation"] != "identity":
        discovery = Discover(path["representation"], path["seed"]).fit(x, target, weight)
        x, test_x = discovery.transform(x), discovery.transform(test_x)
        evidence = discovery.evidence
        evidence["selected_indices"] = active[evidence["selected_indices"]].tolist()
        evidence["pairs"] = [[int(active[a]), int(active[b])] for a, b in evidence["pairs"]]
    family = path["family"]
    result, detail = {}, {"discovery": evidence, "active_columns": len(active)}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if family in {"lgb", "xgb", "cat"}:
            maximum = max(config["tree_checkpoints"])
            if family == "lgb":
                from lightgbm import LGBMRegressor

                model = LGBMRegressor(
                    n_estimators=maximum,
                    learning_rate=path["learning_rate"],
                    num_leaves=path["leaves"],
                    max_depth=path["depth"] or -1,
                    min_child_samples=path["leaf_samples"],
                    reg_lambda=path["l2"],
                    colsample_bytree=path["max_features"],
                    max_bin=63,
                    verbosity=-1,
                    n_jobs=1,
                    random_state=path["seed"],
                )
            elif family == "xgb":
                from xgboost import XGBRegressor

                model = XGBRegressor(
                    n_estimators=maximum,
                    learning_rate=path["learning_rate"],
                    max_depth=path["depth"],
                    min_child_weight=path["leaf_samples"],
                    reg_lambda=path["l2"],
                    colsample_bytree=path["max_features"],
                    subsample=0.8,
                    max_bin=63,
                    tree_method="hist",
                    n_jobs=1,
                    random_state=path["seed"],
                )
            else:
                from catboost import CatBoostRegressor

                model = CatBoostRegressor(
                    iterations=maximum,
                    learning_rate=path["learning_rate"],
                    depth=path["depth"],
                    l2_leaf_reg=path["l2"],
                    rsm=path["max_features"],
                    border_count=63,
                    thread_count=1,
                    verbose=False,
                    allow_writing_files=False,
                    random_seed=path["seed"],
                )
            model.fit(x, target, sample_weight=weight)
            detail["capacity"] = []
            for n, name in zip(config["tree_checkpoints"], names_for(path, config), strict=True):
                kw = (
                    {"num_iteration": n}
                    if family == "lgb"
                    else ({"iteration_range": (0, n)} if family == "xgb" else {"ntree_end": n})
                )
                result[name] = model.predict(test_x, **kw) * test_exposure
                detail["capacity"].append(
                    {
                        "trees": n,
                        "elapsed": time.monotonic() - start,
                        "train_mse_rate": float(np.mean((model.predict(x, **kw) - target) ** 2)),
                    }
                )
        elif family == "hist":
            model = HistGradientBoostingRegressor(
                learning_rate=path["learning_rate"],
                max_leaf_nodes=path["leaves"],
                min_samples_leaf=path["leaf_samples"],
                l2_regularization=path["l2"],
                max_features=path["max_features"],
                max_depth=path["depth"],
                max_bins=63,
                early_stopping=False,
                warm_start=True,
                random_state=path["seed"],
            )
        elif family in {"extra", "forest"}:
            cls = ExtraTreesRegressor if family == "extra" else RandomForestRegressor
            model = cls(
                min_samples_leaf=path["leaf_samples"],
                max_depth=path["depth"],
                max_features=path["max_features"],
                warm_start=True,
                n_jobs=1,
                random_state=path["seed"],
            )
        if family in {"hist", "extra", "forest"}:
            detail["capacity"] = []
            for n, name in zip(config["tree_checkpoints"], names_for(path, config), strict=True):
                model.set_params(**{"max_iter" if family == "hist" else "n_estimators": n})
                model.fit(x, target, sample_weight=weight)
                result[name] = model.predict(test_x) * test_exposure
                detail["capacity"].append(
                    {
                        "trees": n,
                        "elapsed": time.monotonic() - start,
                        "train_mse_rate": float(np.mean((model.predict(x) - target) ** 2)),
                    }
                )
            if family != "hist":
                order = np.argsort(-model.feature_importances_)[:40]
                detail["importance"] = [
                    {"index": int(active[j]), "importance": float(model.feature_importances_[j])}
                    for j in order
                ]
        elif family in {"lgb", "xgb", "cat"}:
            pass
        elif family in {"pls", "hurdle", "neural_tree", "multi_extra"}:
            f = "extra" if family == "multi_extra" else family
            representation = {"pls": "pls", "neural_tree": "neural"}.get(f, "identity")
            recipe = Recipe(
                family,
                f,
                representation,
                "all",
                0,
                0,
                dict(
                    alpha=100.0,
                    components=24,
                    iterations=160,
                    leaves=15,
                    leaf_samples=20,
                    C=10.0,
                    gamma=0.1,
                    hidden=64,
                    epochs=200,
                ),
            )
            labels = y / exposure[:, None]
            model = MultiTarget(recipe, path["seed"]).fit(x, labels, weight)
            result[path["name"]] = model.predict(test_x)[:, 0] * test_exposure
            detail["learners"] = [learner.diagnostics() for _, learner in model.learners]
        else:
            med = np.nanmedian(x, axis=0)
            z = np.where(np.isfinite(x), x, med)
            mean, scale = z.mean(axis=0), np.maximum(z.std(axis=0), 1e-6)
            z = np.clip((z - mean) / scale, -100, 100)
            t = np.clip((np.where(np.isfinite(test_x), test_x, med) - mean) / scale, -100, 100)
            z = np.column_stack([z, ~np.isfinite(x)])
            t = np.column_stack([t, ~np.isfinite(test_x)])
            if family in {"svr", "kernel", "mlp"}:
                mapper = PCA(
                    n_components=min(48, len(z) - 1, z.shape[1]),
                    svd_solver="randomized",
                    random_state=path["seed"],
                )
                z, t = mapper.fit_transform(z), mapper.transform(t)
            if family == "kernel":
                mapper = Nystroem(
                    n_components=min(192, len(z)), gamma=0.01, random_state=path["seed"]
                )
                z, t = mapper.fit_transform(z), mapper.transform(t)
            scale_y = max(target.std(), 1e-6)
            if family in {"ridge", "kernel"}:
                model = Ridge(alpha=100.0 if family == "ridge" else 1.0)
            elif family == "svr":
                model = SVR(C=10, gamma="scale", epsilon=0.1)
            else:
                model = MLPRegressor(
                    hidden_layer_sizes=(64, 32),
                    alpha=1.0,
                    max_iter=250,
                    early_stopping=False,
                    random_state=path["seed"],
                )
            model.fit(z, target / scale_y, sample_weight=weight)
            result[path["name"]] = model.predict(t) * scale_y * test_exposure
        detail["warnings"] = sorted({str(w.message) for w in caught})
    detail["fit_seconds"] = time.monotonic() - start
    if any(not np.isfinite(v).all() for v in result.values()):
        raise ValueError("Nonfinite forecast")
    return result, detail
