"""Training-fitted unsupervised representations for broad raw-input research.

No interface accepts targets. Call ``fit_representation`` separately within every
chronological training fold; cache only folds with identical training rows and
specifications. A descriptive full-data fit must not supply a forecasting fold.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.random_projection import SparseRandomProjection

KINDS = (
    "raw",
    "variance",
    "correlation",
    "pca",
    "raw_pca",
    "group_pca",
    "svd",
    "random_projection",
    "cluster",
)


def source_balance_family(name: str) -> str:
    """Source families shared across windows; missingness has separate group weights."""
    if name.startswith("missing:"):
        return "missing:" + source_balance_family(name.removeprefix("missing:"))
    if "participation_" in name:
        return "participation"
    if "pbp_" in name:
        return "plays"
    if "team_prior__" in name or "team_current__" in name:
        return "team_box"
    for family in ("box", "snaps", "ngs", "depth", "injuries", "rosters", "college", "combine"):
        if family + "__" in name:
            return family
    if name.startswith("history_"):
        return "history_totals"
    if name.endswith(("__observations", "__calendar_recency")):
        return "history_coverage"
    return "calendar_demographics"


def feature_family(name: str) -> str:
    """Keep observation provenance visible for native and transformed inputs."""
    if name.startswith("missing:"):
        return "missing:" + feature_family(name.removeprefix("missing:"))
    if "__" in name:
        parts = name.split("__")
        return "__".join(parts[:2]) if len(parts) > 2 else parts[0]
    return name.split(":", 1)[0]


class FittedRepresentation:
    """Median imputation, missing flags and an optional unsupervised transform.

    The raw anchor retains every input whose imputed value or missingness varies
    in training. It never filters using observed outcomes or other folds. Raw
    values are unscaled; only projection geometry uses robust scaling. IQR-zero
    columns fall back to standard deviation, preserving rare/binary predictors.

    Correlation filtering uses absolute Pearson correlation after imputation.
    Feature clustering groups signed standardized feature vectors after a seeded
    random sketch of their training rows and returns actual representative
    columns, not cluster labels. Both may use a recorded training-row subsample
    to bound work; PCA/SVD fit all supplied training rows.
    """

    def __init__(self, spec: Mapping | str, seed: int = 20261003):
        self.spec = {"kind": spec} if isinstance(spec, str) else dict(spec)
        self.kind = self.spec.get("kind", "raw")
        if self.kind not in KINDS:
            raise ValueError(f"Unknown representation {self.kind!r}; choose from {KINDS}")
        allowed = {
            "kind",
            "n_components",
            "missing_indicators",
            "min_variance",
            "correlation_threshold",
            "max_fit_rows",
            "sketch_size",
            "clip",
        }
        unknown = set(self.spec) - allowed
        if unknown:
            raise ValueError(f"Unknown representation options: {sorted(unknown)}")
        self.seed = int(seed)
        self.n_components = int(self.spec.get("n_components", 64))
        self.min_variance = float(self.spec.get("min_variance", 0))
        self.correlation_threshold = float(self.spec.get("correlation_threshold", 0.98))
        self.max_fit_rows = int(self.spec.get("max_fit_rows", 2048))
        self.sketch_size = int(self.spec.get("sketch_size", 128))
        self.clip = self.spec.get("clip", None)
        if self.n_components < 1 or self.max_fit_rows < 2 or self.sketch_size < 1:
            raise ValueError("Representation dimensions must be positive; max_fit_rows >= 2")
        if not np.isfinite(self.min_variance) or self.min_variance < 0:
            raise ValueError("min_variance must be finite and nonnegative")
        if not 0 < self.correlation_threshold <= 1:
            raise ValueError("correlation_threshold must be in (0, 1]")
        if self.clip is not None and (not np.isfinite(self.clip) or self.clip <= 0):
            raise ValueError("clip must be positive and finite, or None")

    @staticmethod
    def _matrix(x):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError("Representation input must be a two-dimensional numeric matrix")
        return np.where(np.isfinite(x), x, np.nan)

    def fit(self, x, feature_names: Sequence[str]):
        x = self._matrix(x)
        if x.shape[0] < 2:
            raise ValueError("At least two training observations are required")
        if x.shape[1] != len(feature_names) or len(set(feature_names)) != len(feature_names):
            raise ValueError("Feature names must be unique and match the matrix columns")
        self.feature_names_in_ = np.asarray(feature_names, dtype=object)
        self.n_features_in_ = x.shape[1]
        self.n_train_ = x.shape[0]
        missing = np.isnan(x)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            self.medians_ = np.nanmedian(x, axis=0)
        self.medians_ = np.nan_to_num(self.medians_, nan=0)
        values = np.where(missing, self.medians_, x)
        self.value_mask_ = values.var(axis=0, dtype=np.float64) > self.min_variance
        self.missing_mask_ = (
            missing.any(axis=0) & ~missing.all(axis=0)
            if self.spec.get("missing_indicators", True)
            else np.zeros(x.shape[1], dtype=bool)
        )
        self.base_names_ = np.concatenate(
            [
                self.feature_names_in_[self.value_mask_],
                np.asarray(
                    [f"missing:{n}" for n in self.feature_names_in_[self.missing_mask_]],
                    dtype=object,
                ),
            ]
        )
        self.degenerate_ = len(self.base_names_) == 0
        if self.degenerate_:
            self.base_names_ = np.asarray(["__constant__"], dtype=object)
        base = self._base(x)
        self.selected_indices_ = np.arange(base.shape[1])
        self.projector_ = None
        self.cluster_records_ = []
        self.fit_rows_used_ = len(x)
        self.center_ = np.median(base, axis=0)
        q25, q75 = np.quantile(base, [0.25, 0.75], axis=0)
        std = base.std(axis=0, dtype=np.float64)
        self.scale_ = np.where(q75 > q25, q75 - q25, np.where(std > 0, std, 1))
        self.scale_ = self.scale_.astype(np.float32)
        self.group_weights_ = np.ones(base.shape[1], dtype=np.float32)
        self.group_counts_ = {}
        if self.kind == "group_pca":
            # Equal aggregate variance per source family prevents a source from
            # dominating geometry merely because it exposes hundreds of fields.
            # Missing flags form separate source-family blocks. All counts and
            # scales come from the nonconstant columns of this training fold.
            self.center_ = base.mean(axis=0)
            self.scale_ = np.where(std > 0, std, 1).astype(np.float32)
            groups = [source_balance_family(str(name)) for name in self.base_names_]
            self.group_counts_ = {group: groups.count(group) for group in sorted(set(groups))}
            self.group_weights_ = np.asarray(
                [1 / np.sqrt(self.group_counts_[group]) for group in groups],
                dtype=np.float32,
            )

        if self.kind in {"correlation", "cluster"} and not self.degenerate_:
            self._select(base, missing)
        elif self.kind in {"pca", "raw_pca", "group_pca", "svd", "random_projection"}:
            scaled = self._scaled(base)
            n = min(self.n_components, scaled.shape[1], max(1, scaled.shape[0] - 1))
            if self.degenerate_:
                self.output_names_ = self.base_names_.copy()
                return self
            if self.kind in {"pca", "raw_pca", "group_pca"}:
                self.projector_ = PCA(
                    n_components=n, svd_solver="randomized", random_state=self.seed
                )
            elif self.kind == "svd":
                # sklearn's SVD requires at least two input columns.
                if scaled.shape[1] > 1:
                    self.projector_ = TruncatedSVD(n_components=n, n_iter=5, random_state=self.seed)
            else:
                self.projector_ = SparseRandomProjection(
                    n_components=n,
                    dense_output=True,
                    random_state=self.seed,
                )
            if self.projector_ is not None:
                self.projector_.fit(scaled)
                actual = self.projector_.components_.shape[0]
                projection_names = np.asarray(
                    [f"{self.kind.replace('raw_', '')}:component:{i:03d}" for i in range(actual)],
                    dtype=object,
                )
                self.output_names_ = (
                    np.concatenate([self.base_names_, projection_names])
                    if self.kind == "raw_pca"
                    else projection_names
                )
                return self
        self.output_names_ = self.base_names_[self.selected_indices_]
        return self

    def _base(self, x):
        if self.degenerate_:
            return np.zeros((len(x), 1), dtype=np.float32)
        missing = np.isnan(x)
        values = np.where(
            missing[:, self.value_mask_], self.medians_[self.value_mask_], x[:, self.value_mask_]
        )
        return np.column_stack([values, missing[:, self.missing_mask_]]).astype(np.float32)

    def _scaled(self, base):
        scaled = (base - self.center_) / self.scale_
        if self.clip is not None:
            scaled = np.clip(scaled, -self.clip, self.clip)
        scaled *= getattr(self, "group_weights_", 1.0)
        return scaled.astype(np.float32)

    def _select(self, base, missing):
        rng = np.random.default_rng(self.seed)
        rows = np.sort(rng.choice(len(base), min(self.max_fit_rows, len(base)), replace=False))
        self.fit_rows_used_ = len(rows)
        z = base[rows].astype(np.float64)
        z -= z.mean(axis=0)
        norms = np.linalg.norm(z, axis=0)
        # A feature that varies outside the sampled rows remains eligible. Its
        # zero-vector sketch is kept individually instead of declaring it constant.
        varying = np.flatnonzero(norms > 0)
        unsampled = np.flatnonzero(norms == 0)
        z = np.divide(z, norms, out=np.zeros_like(z), where=norms > 0).astype(np.float32)
        if self.kind == "correlation":
            coverage = np.r_[
                (~missing[:, self.value_mask_]).mean(axis=0),
                np.ones(self.missing_mask_.sum()),
            ]
            # Prefer the more completely observed member of a correlated pair.
            ordered = np.argsort(-coverage, kind="stable")
            blocked = np.zeros(base.shape[1], dtype=bool)
            selected = []
            # Block multiplication avoids a full p-by-p correlation allocation.
            for start in range(0, len(ordered), 256):
                indices = ordered[start : start + 256]
                correlations = z[:, indices].T @ z
                for i, correlations_i in zip(indices, correlations, strict=True):
                    if not blocked[i]:
                        selected.append(int(i))
                        blocked |= np.abs(correlations_i) >= self.correlation_threshold - 1e-6
            self.selected_indices_ = np.sort(np.asarray(selected))
            return
        count = min(self.n_components, len(varying))
        if count == 0 or count == len(varying):
            return
        projector = SparseRandomProjection(
            n_components=min(self.sketch_size, len(rows)),
            dense_output=True,
            random_state=self.seed,
        )
        sketches = projector.fit_transform(z[:, varying].T).astype(np.float32)
        clusterer = MiniBatchKMeans(
            n_clusters=count,
            random_state=self.seed,
            n_init=3,
            batch_size=min(1024, len(varying)),
            max_iter=100,
            reassignment_ratio=0,
        ).fit(sketches)
        selected = list(unsampled)
        for label in sorted(set(clusterer.labels_)):
            members = np.flatnonzero(clusterer.labels_ == label)
            distances = np.linalg.norm(
                sketches[members] - clusterer.cluster_centers_[label], axis=1
            )
            representative = int(varying[members[np.argmin(distances)]])
            selected.append(representative)
            self.cluster_records_.append(
                {
                    "cluster": int(label),
                    "representative": str(self.base_names_[representative]),
                    "members": self.base_names_[varying[members]].tolist(),
                }
            )
        self.selected_indices_ = np.sort(np.asarray(selected, dtype=int))

    def transform(self, x):
        if not hasattr(self, "output_names_"):
            raise ValueError("Representation has not been fitted")
        x = self._matrix(x)
        if x.shape[1] != self.n_features_in_:
            raise ValueError("Transform requires the fitted input feature order and width")
        base = self._base(x)
        if self.projector_ is None:
            return np.ascontiguousarray(base[:, self.selected_indices_], dtype=np.float32)
        transformed = self.projector_.transform(self._scaled(base))
        if self.kind == "raw_pca":
            transformed = np.column_stack([base, transformed])
        return np.ascontiguousarray(transformed, dtype=np.float32)

    def fit_transform(self, x, feature_names: Sequence[str]):
        return self.fit(x, feature_names).transform(x)

    def get_feature_names_out(self):
        return self.output_names_.copy()

    def summary(self, top_loadings: int = 8):
        """JSON-safe provenance, native representatives and signed component loadings."""
        result = {
            "spec": self.spec,
            "seed": self.seed,
            "training_rows": self.n_train_,
            "geometry_fit_rows": self.fit_rows_used_,
            "input_features": self.n_features_in_,
            "nonconstant_values": int(self.value_mask_.sum()),
            "missing_indicators": int(self.missing_mask_.sum()),
            "output_features": len(self.output_names_),
            "feature_names": self.output_names_.tolist(),
            "feature_groups": [feature_family(n) for n in self.output_names_],
            "constant_in_training": self.feature_names_in_[~self.value_mask_].tolist(),
            "degenerate_training_inputs": self.degenerate_,
            "cluster_representatives": self.cluster_records_,
            "fit_uses_targets": False,
            "scale": "unscaled raw; training median/IQR (std fallback) for projections",
            "source_balance_groups": getattr(self, "group_counts_", {}),
        }
        if self.kind == "group_pca":
            result["scale"] = "training mean/std, then inverse sqrt nonconstant source-group width"
        if self.projector_ is not None:
            ratios = getattr(self.projector_, "explained_variance_ratio_", None)
            if ratios is not None:
                result["explained_variance_ratio"] = np.asarray(ratios).tolist()
            components = self.projector_.components_
            if hasattr(components, "toarray"):
                components = components.toarray()
            result["component_loadings"] = [
                [
                    {"feature": str(self.base_names_[i]), "loading": float(component[i])}
                    for i in np.argsort(-np.abs(component), kind="stable")[:top_loadings]
                ]
                for component in components
            ]
        return result


def fit_representation(x_train, feature_names, spec, seed=20261003):
    """Fit exclusively on supplied training predictors; never read other folds or y."""
    return FittedRepresentation(spec, seed).fit(x_train, feature_names)
