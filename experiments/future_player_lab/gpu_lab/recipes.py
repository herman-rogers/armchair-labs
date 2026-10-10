"""A frozen factorial library and seeded hyperparameter search, independent of outcomes."""

from __future__ import annotations

import numpy as np


def recipe_library(seed=20261003, smoke=False):
    recipes = []

    def add(family, representation="raw", params=None, view="all", history=0):
        rep = {"kind": representation, "n_components": 48, "missing_indicators": True}
        recipes.append(
            {
                "id": f"r{len(recipes):03d}_{family}_{representation}_{view}",
                "family": family,
                "device": "cuda" if family in {"xgboost", "catboost", "cuml_rf"} else "cpu",
                "representation": rep,
                "params": params or {},
                "view": view,
                "history": history,
                "seed": seed,
            }
        )

    defaults = {
        "xgboost": dict(
            n_estimators=160,
            max_depth=3,
            learning_rate=0.035,
            min_child_weight=20,
            reg_lambda=15,
            subsample=0.85,
            colsample_bytree=0.8,
            max_bin=128,
        ),
        "catboost": dict(n_estimators=160, depth=4, learning_rate=0.035, l2_leaf_reg=15),
        "cuml_rf": dict(
            n_estimators=64,
            max_depth=10,
            min_samples_leaf=12,
            max_features=0.7,
            n_bins=64,
            n_streams=4,
        ),
        "lightgbm": dict(
            n_estimators=160,
            num_leaves=15,
            learning_rate=0.035,
            min_child_samples=25,
            reg_lambda=15,
            colsample_bytree=0.8,
        ),
    }
    representations = [
        "raw",
        "correlation",
        "pca",
        "group_pca",
        "svd",
        "random_projection",
        "cluster",
        "raw_pca",
    ]
    for family, params in defaults.items():
        for representation in representations if not smoke else ["raw"]:
            add(family, representation, dict(params))
    add(
        "hist",
        params=dict(
            max_iter=120,
            max_leaf_nodes=15,
            learning_rate=0.05,
            min_samples_leaf=30,
            l2_regularization=10,
        ),
    )
    add("extra_trees", params=dict(n_estimators=160, min_samples_leaf=8, max_features=0.65))
    if smoke:
        for recipe in recipes:
            if "n_estimators" in recipe["params"]:
                recipe["params"]["n_estimators"] = 12
            if "max_iter" in recipe["params"]:
                recipe["params"]["max_iter"] = 12
        return recipes
    rng = np.random.default_rng(seed)
    for i in range(8):
        family = ["xgboost", "catboost", "cuml_rf"][i % 3]
        params = dict(defaults[family])
        params["n_estimators"] = int(rng.choice([80, 240, 400]))
        if family == "xgboost":
            params.update(
                max_depth=int(rng.choice([2, 4, 6])),
                learning_rate=float(rng.choice([0.02, 0.06, 0.1])),
                min_child_weight=int(rng.choice([5, 20, 50])),
                reg_lambda=float(rng.choice([1, 10, 50])),
                colsample_bytree=float(rng.choice([0.35, 0.7, 1])),
            )
        elif family == "catboost":
            params.update(
                depth=int(rng.choice([3, 5, 6])),
                learning_rate=float(rng.choice([0.025, 0.07])),
                l2_leaf_reg=float(rng.choice([3, 20, 50])),
            )
        else:
            params["n_estimators"] = int(rng.choice([32, 64, 128]))
            params.update(
                max_depth=int(rng.choice([6, 14, 20])),
                min_samples_leaf=int(rng.choice([3, 10, 25])),
                max_features=float(rng.choice([0.3, 0.6, 1.0])),
            )
        add(
            family,
            str(rng.choice(["raw", "correlation", "raw_pca"])),
            params,
            history=int(rng.choice([0, 6, 10])),
        )
    for view in [
        "no_pbp",
        "no_team",
        "no_tracking",
        "no_college_combine",
        "no_availability",
        "no_extended_history",
        "box_only",
        "no_provider_derived",
    ]:
        add("xgboost", params=dict(defaults["xgboost"]), view=view)
    for representation in ["raw", "correlation"]:
        add("xgboost", representation, params=dict(defaults["xgboost"]))
        recipes[-1]["target_mode"] = "factorized"
        recipes[-1]["id"] += "_factorized"
    return recipes


def view_mask(names, view):
    """Prespecified source ablations, never inherited feature usefulness rankings."""

    def keep(name):
        if view == "all":
            return True
        if view == "no_pbp":
            return "pbp_" not in name
        if view == "no_team":
            return "team_" not in name
        if view == "no_tracking":
            return "ngs__" not in name and "participation_" not in name
        if view == "no_college_combine":
            return "college__" not in name and "combine__" not in name
        if view == "no_availability":
            return not any(v in name for v in ["injuries__", "rosters__", "depth__", "snaps__"])
        if view == "no_extended_history":
            return not name.startswith(
                ("prior2__", "career__", "recent8__", "last8__", "last_observed__")
            )
        if view == "box_only":
            return (
                "box__" in name
                or name.startswith("history_")
                or name in {"age", "weeks", "origin", "scheduled_games"}
            )
        if view == "no_provider_derived":
            return name.split("__")[-1] not in {"cp", "cpoe", "xpass"} and not any(
                v in name.lower() for v in ["epa", "wpa", "expect", "xyac", "_wp", "_ep", "fantasy"]
            )
        raise ValueError(f"Unknown feature view {view}")

    return np.array([keep(n) for n in names])
