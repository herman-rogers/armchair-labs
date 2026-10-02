"""Joint searches over representations, estimators and history policies."""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR


@dataclass(frozen=True)
class Recipe:
    name: str
    family: str
    representation: str
    view: str
    train_years: int
    half_life: float
    params: dict


def recipes(config):
    rng = np.random.default_rng(config["seed"])
    result = []
    for family in config["families"]:
        if family not in {
            "ridge",
            "hist",
            "extra",
            "svr",
            "kernel",
            "mlp",
            "pls",
            "interactions",
            "neural_tree",
            "hurdle",
        }:
            raise ValueError(f"Unknown model family: {family}")
        for trial in range(config["trials_per_family"]):
            # Seeded random search, independent of every outcome and evaluation fold.
            representation = {
                "svr": "pca",
                "kernel": "kernel",
                "pls": "pls",
                "interactions": "interactions",
                "neural_tree": "neural",
            }.get(family, "identity")
            params = {
                "alpha": float(10 ** rng.uniform(0, 3)),
                "leaves": int(rng.choice([7, 15, 31])),
                "leaf_samples": int(rng.choice([10, 25, 50])),
                "components": int(rng.choice([12, 24, 48])),
                "iterations": int(rng.choice([60, 100, 160])),
                "C": float(10 ** rng.uniform(-0.5, 1.5)),
                "gamma": float(10 ** rng.uniform(-2, 0)),
                "hidden": int(rng.choice([32, 64, 96])),
                "epochs": int(rng.choice([40, 70, 100])),
            }
            if trial == 0:
                params.update(iterations=40, leaves=7, epochs=40, hidden=32, components=24)
            result.append(
                Recipe(
                    f"{family}_{trial:03d}",
                    family,
                    representation,
                    "summary" if trial == 0 else str(rng.choice(["all", "sequence", "summary"])),
                    0 if trial == 0 else int(rng.choice([0, 5, 10])),
                    0 if trial == 0 else float(rng.choice([0, 3, 7])),
                    params,
                )
            )
    return result


def view_indices(names, view):
    return np.array(
        [
            i
            for i, name in enumerate(names)
            if view == "all"
            or not name.startswith("sequence:" if view == "summary" else "summary:")
        ]
    )


class Transform:
    def fit(self, x):
        self.medians = np.array(
            [np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else 0 for c in x.T],
            dtype=np.float32,
        )
        filled = np.where(np.isfinite(x), x, self.medians)
        self.mean = filled.mean(axis=0)
        self.scale = filled.std(axis=0)
        self.scale[self.scale < 1e-6] = 1
        # Constants can still carry newly missing values; every column retains its missing flag.
        self.active = np.flatnonzero(filled.std(axis=0) > 1e-6)
        return self

    def transform(self, x):
        missing = ~np.isfinite(x)
        z = (np.where(missing, self.medians, x) - self.mean) / self.scale
        return np.c_[np.clip(z[:, self.active], -30, 30), missing.astype(np.float32)]


class Learner:
    def __init__(self, recipe, seed):
        self.recipe, self.seed = recipe, seed

    def fit(self, x, y, weights):
        self.transformer = Transform().fit(x)
        z = self.transformer.transform(x)
        self.y_mean, self.y_scale = y.mean(axis=0), y.std(axis=0)
        self.y_scale[self.y_scale < 1e-6] = 1
        scaled = (y - self.y_mean) / self.y_scale
        self.representer = None
        p, rep = self.recipe.params, self.recipe.representation
        n = max(1, min(p["components"], len(z) - 1, z.shape[1]))
        if rep == "pca":
            self.representer = PCA(n_components=n, svd_solver="randomized", random_state=self.seed)
            z = self.representer.fit_transform(z)
        elif rep == "pls":
            self.representer = PLSRegression(n_components=min(n, 12), scale=False, max_iter=150)
            self.representer.fit(z, scaled)
            z = self.representer.transform(z)
        elif rep == "kernel":
            self.representer = Nystroem(
                n_components=min(128, len(z)), gamma=p["gamma"] / z.shape[1], random_state=self.seed
            )
            z = self.representer.fit_transform(z)
        elif rep == "interactions":
            selector = ExtraTreesRegressor(
                n_estimators=32, max_depth=5, min_samples_leaf=12, n_jobs=1, random_state=self.seed
            )
            selector.fit(
                z, scaled if scaled.shape[1] > 1 else scaled.ravel(), sample_weight=weights
            )
            self.interaction_indices = np.argsort(-selector.feature_importances_)[
                : min(24, z.shape[1])
            ]
            self.pairs = [
                (a, b) for a in self.interaction_indices for b in self.interaction_indices if a <= b
            ]
            z = self._interactions(z)
        elif rep == "neural":
            self.representer = MLPRegressor(
                hidden_layer_sizes=(p["hidden"], n),
                alpha=0.1,
                max_iter=p["epochs"],
                early_stopping=False,
                random_state=self.seed,
                batch_size=min(256, len(z)),
            )
            with warnings.catch_warnings(record=True) as records:
                warnings.simplefilter("always", ConvergenceWarning)
                self.representer.fit(z, scaled, sample_weight=weights)
            self.encoder_not_converged = any(
                isinstance(w.message, ConvergenceWarning) for w in records
            )
            z = self._neural(z)
        self.models = []
        families = self.recipe.family
        if families in ("hist", "svr"):
            for target in range(y.shape[1]):
                if families == "hist":
                    model = HistGradientBoostingRegressor(
                        max_iter=p["iterations"],
                        max_leaf_nodes=p["leaves"],
                        min_samples_leaf=p["leaf_samples"],
                        l2_regularization=10,
                        max_bins=63,
                        early_stopping=False,
                        random_state=self.seed,
                    )
                else:
                    model = SVR(C=p["C"], gamma="scale", epsilon=0.1, cache_size=256)
                model.fit(z, scaled[:, target], sample_weight=weights)
                self.models.append(model)
        else:
            if families in ("extra", "neural_tree", "hurdle"):
                model = ExtraTreesRegressor(
                    n_estimators=p["iterations"],
                    max_features=0.8,
                    min_samples_leaf=p["leaf_samples"],
                    max_depth=12,
                    n_jobs=1,
                    random_state=self.seed,
                )
            elif families == "mlp":
                model = MLPRegressor(
                    hidden_layer_sizes=(p["hidden"], max(8, p["hidden"] // 2)),
                    alpha=0.1,
                    max_iter=p["epochs"],
                    early_stopping=False,
                    random_state=self.seed,
                    batch_size=min(256, len(z)),
                )
            else:
                model = Ridge(alpha=p["alpha"])
            with warnings.catch_warnings(record=True) as records:
                warnings.simplefilter("always", ConvergenceWarning)
                model.fit(
                    z, scaled if scaled.shape[1] > 1 else scaled.ravel(), sample_weight=weights
                )
            self.not_converged = any(isinstance(w.message, ConvergenceWarning) for w in records)
            self.models = [model]
        self.output_width = y.shape[1]
        return self

    def _interactions(self, z):
        return np.c_[z, np.column_stack([z[:, a] * z[:, b] for a, b in self.pairs])]

    def _neural(self, z):
        for weights, bias in zip(
            self.representer.coefs_[:-1], self.representer.intercepts_[:-1], strict=True
        ):
            z = np.maximum(0, z @ weights + bias)
        return z

    def predict(self, x):
        z = self.transformer.transform(x)
        if self.recipe.representation == "interactions":
            z = self._interactions(z)
        elif self.recipe.representation == "neural":
            z = self._neural(z)
        elif self.representer is not None:
            z = self.representer.transform(z)
        if self.recipe.family in ("hist", "svr"):
            y = np.column_stack([m.predict(z) for m in self.models])
        else:
            y = self.models[0].predict(z).reshape(len(x), self.output_width)
        return y * self.y_scale + self.y_mean

    def diagnostics(self):
        result = {
            "recipe": asdict(self.recipe),
            "variable_columns": len(self.transformer.active),
            "optimizer_budget_reached": getattr(self, "not_converged", False),
            "encoder_budget_reached": getattr(self, "encoder_not_converged", False),
        }
        if self.recipe.representation == "interactions":
            result["interaction_transformed_indices"] = self.pairs
        return result


class MultiTarget:
    """Share representations among observed targets, without imputing unknown labels."""

    def __init__(self, recipe, seed):
        self.recipe, self.seed = recipe, seed

    def fit(self, x, y, weights):
        groups = {}
        for j in range(y.shape[1]):
            valid = np.isfinite(y[:, j])
            groups.setdefault(valid.tobytes(), (valid, []))[1].append(j)
        self.learners = []
        self.width = y.shape[1]
        self.participation = None
        if self.recipe.family == "hurdle":
            active = y[:, 4] > 0
            if active.any() and not active.all():
                self.role_transform = Transform().fit(x)
                self.participation = ExtraTreesClassifier(
                    n_estimators=self.recipe.params["iterations"],
                    min_samples_leaf=20,
                    max_depth=10,
                    n_jobs=1,
                    random_state=self.seed,
                )
                self.participation.fit(
                    self.role_transform.transform(x), active, sample_weight=weights
                )
                groups = {}
                for j in range(y.shape[1]):
                    valid = np.isfinite(y[:, j]) & active
                    groups.setdefault(valid.tobytes(), (valid, []))[1].append(j)
        for valid, cols in groups.values():
            if valid.sum() < 2:
                continue
            learner = Learner(self.recipe, self.seed).fit(
                x[valid], y[valid][:, cols], weights[valid]
            )
            self.learners.append((cols, learner))
        return self

    def predict(self, x):
        result = np.full((len(x), self.width), np.nan)
        for cols, learner in self.learners:
            result[:, cols] = learner.predict(x)
        if self.participation is not None:
            probability = self.participation.predict_proba(self.role_transform.transform(x))[:, 1]
            result *= probability[:, None]
        # Counts/workload are nonnegative; points and yardage may legitimately be negative.
        result[:, [1, 3, 4]] = np.maximum(0, result[:, [1, 3, 4]])
        result[:, 4] = np.minimum(1, result[:, 4])
        return result
