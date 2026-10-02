"""Every fitted transformation belongs to a single chronological training fold."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR


@dataclass(frozen=True)
class Recipe:
    name: str
    inputs: str
    model: str
    representation: str = "identity"
    setting: float = 0


def recipes():
    return [
        Recipe("basic_tree", "basic", "tree", setting=15),
        Recipe("legacy_screen_tree", "inventory", "tree", "legacy", 15),
        Recipe("inventory_tree7", "inventory", "tree", setting=7),
        Recipe("inventory_tree15", "inventory", "tree", setting=15),
        Recipe("inventory_ridge", "inventory", "ridge", setting=100),
        Recipe("inventory_pca_svr", "inventory", "svr", "pca", 10),
        Recipe("inventory_selected_svr", "inventory", "svr", "selected", 10),
        Recipe("raw_tree7", "raw", "tree", setting=7),
        Recipe("raw_tree15", "raw", "tree", setting=15),
        Recipe("raw_recent_tree", "recent", "tree", setting=15),
        Recipe("raw_ridge", "raw", "ridge", setting=100),
        Recipe("raw_transform_ridge", "raw", "ridge", "transform", 100),
        Recipe("raw_pca_svr", "raw", "svr", "pca", 10),
        Recipe("raw_selected_svr", "raw", "svr", "selected", 10),
        Recipe("inventory_market_tree", "market", "tree", setting=15),
    ]


class FoldModel:
    def __init__(self, recipe, seed):
        self.recipe, self.seed = recipe, seed

    def fit(self, x, y):
        # All columns remain eligible, including binaries, sparse and all-missing
        # columns. All-missing values receive zero plus an explicit missing flag.
        self.medians = np.array(
            [np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else 0.0 for c in x.T]
        )
        filled = np.where(np.isfinite(x), x, self.medians)
        self.mean, self.scale = filled.mean(axis=0), filled.std(axis=0)
        self.scale[self.scale < 1e-8] = 1
        self.indices = np.arange(x.shape[1])
        self.selector = None
        self.pca = None
        z = self._base(x)
        if self.recipe.representation == "legacy":
            scores = []
            for j, c in enumerate(x.T):
                present = np.isfinite(c)
                if present.mean() < 0.25 or np.unique(c[present]).size < 3:
                    continue
                if np.ptp(y[present]) == 0:
                    continue
                score = abs(np.corrcoef(rankdata(c[present]), rankdata(y)[present])[0, 1])
                if np.isfinite(score):
                    scores.append((score, j))
            self.indices = np.array([j for _, j in sorted(scores, reverse=True)[:80]], dtype=int)
            if not len(self.indices):
                raise ValueError("Legacy selector has no candidates")
        elif self.recipe.representation == "selected":
            self.selector = ExtraTreesRegressor(
                n_estimators=48,
                max_depth=7,
                min_samples_leaf=8,
                max_features=0.7,
                random_state=self.seed,
                n_jobs=1,
            ).fit(z, y)
            p = x.shape[1]
            scores = self.selector.feature_importances_[:p] + self.selector.feature_importances_[p:]
            self.indices = np.argsort(-scores, kind="stable")[:64]
        z = self._represent(z, fit=True)
        if self.recipe.model == "tree":
            self.model = HistGradientBoostingRegressor(
                max_iter=100,
                learning_rate=0.05,
                max_leaf_nodes=int(self.recipe.setting),
                min_samples_leaf=20,
                l2_regularization=10,
                max_bins=63,
                early_stopping=False,
                random_state=self.seed,
            )
        elif self.recipe.model == "ridge":
            self.model = Ridge(alpha=self.recipe.setting)
        else:
            self.model = SVR(C=self.recipe.setting, epsilon=0.1, gamma="scale", cache_size=256)
        # The target is points per scheduled game. No test-label scale estimate.
        self.model.fit(z, y)
        return self

    def _base(self, x):
        missing = ~np.isfinite(x)
        filled = np.where(missing, self.medians, x)
        standardized = (filled - self.mean) / self.scale
        # Training and test have identical transformations; training-derived scale.
        standardized = np.clip(standardized, -20, 20)
        return np.column_stack((standardized, missing.astype(float)))

    def _represent(self, z, *, fit=False):
        p = z.shape[1] // 2
        z = z[:, np.r_[self.indices, p + self.indices]]
        if self.recipe.representation == "transform":
            # Generic nonlinear basis; no manually selected football interactions.
            z = np.column_stack((z, np.sign(z) * np.log1p(abs(z)), z * z))
        if self.recipe.representation == "pca":
            if fit:
                self.pca = PCA(
                    n_components=min(32, z.shape[0] - 1, z.shape[1]),
                    svd_solver="randomized",
                    random_state=self.seed,
                ).fit(z)
            z = self.pca.transform(z)
        return z

    def predict(self, x):
        return self.model.predict(self._represent(self._base(x)))

    def diagnostics(self, names):
        result = {"input_count": len(names), "retained_input_count": len(self.indices)}
        if self.recipe.representation in ("selected", "legacy"):
            result["selected_features"] = [names[j] for j in self.indices]
        if self.pca is not None:
            result["pca_explained_variance"] = float(self.pca.explained_variance_ratio_.sum())
        if self.selector is not None:
            p = len(names)
            importance = self.selector.feature_importances_
            result["importance"] = {
                names[j]: float(importance[j] + importance[p + j]) for j in self.indices
            }
            pairs = Counter()
            for estimator in self.selector.estimators_:
                tree = estimator.tree_
                for node, feature in enumerate(tree.feature):
                    if feature < 0:
                        continue
                    for child in (tree.children_left[node], tree.children_right[node]):
                        second = tree.feature[child]
                        if second >= 0 and feature % p != second % p:
                            pairs[tuple(sorted((names[feature % p], names[second % p])))] += 1
            result["split_cooccurrences"] = [
                {"first": a, "second": b, "count": n} for (a, b), n in pairs.most_common(30)
            ]
        return result


def choose(history, eligible, year, inner_seasons=3):
    years = sorted({s for name in eligible for s in history.get(name, {}) if s < year})[
        -inner_seasons:
    ]
    if len(years) != inner_seasons:
        raise ValueError("Insufficient earlier validation seasons")
    losses = {
        name: float(np.mean([history[name][s] for s in years]))
        for name in eligible
        if all(s in history.get(name, {}) for s in years)
    }
    if not losses:
        raise ValueError("No common chronological validation folds")
    return min(losses, key=lambda name: (losses[name], name)), years, losses


def point_predictions(model, x, season_length):
    # Same deterministic nonnegative bound for every method; no actual availability cap.
    return np.maximum(0, model.predict(x)) * season_length
