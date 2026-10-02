"""Exact, expanding-history ridge block additions and refitted deletions."""

from __future__ import annotations

import numpy as np


def prepare(train, test, names):
    """Train-only median imputation/scaling; each value and missing flag form a block."""
    left, right, blocks = [], [], {}
    for j, name in enumerate(names):
        a, b = train[:, j].copy(), test[:, j].copy()
        missing, missing_test = ~np.isfinite(a), ~np.isfinite(b)
        median = np.median(a[~missing]) if (~missing).any() else 0.0
        a[missing], b[missing_test] = median, median
        pairs = [(a, b)]
        if missing.any():
            pairs.append((missing.astype(float), missing_test.astype(float)))
        blocks[name] = []
        for a, b in pairs:
            if np.ptp(a) == 0:
                continue
            mean, std = a.mean(), a.std()
            if std > 0:
                blocks[name].append(len(left))
                left.append((a - mean) / std)
                right.append((b - mean) / std)
    x = np.column_stack(left) if left else np.empty((len(train), 0))
    xt = np.column_stack(right) if right else np.empty((len(test), 0))
    return x, xt, blocks


class Context:
    def __init__(self, x, xt, y, blocks, names, alpha=100):
        self.names = list(dict.fromkeys(names))
        self.indices = [i for n in self.names for i in blocks[n]]
        self.blocks = {n: [self.indices.index(i) for i in blocks[n]] for n in self.names}
        self.x, self.xt = x[:, self.indices], xt[:, self.indices]
        self.y_mean, self.y = y.mean(), y - y.mean()
        self.alpha = alpha
        self.inverse = np.linalg.inv(self.x.T @ self.x + alpha * np.eye(len(self.indices)))
        self.coef = self.inverse @ (self.x.T @ self.y)
        self.prediction = self.xt @ self.coef + self.y_mean
        self.test_inverse = self.xt @ self.inverse
        self._addition = None

    def prepare_additions(self, x, xt):
        cross = self.x.T @ x
        projected = self.inverse @ cross
        self._addition = (
            x,
            xt - self.xt @ projected,
            cross,
            projected,
            x.T @ self.y - cross.T @ self.coef,
        )

    def add(self, block):
        if not block:
            return self.prediction.copy()
        x, residual_test, cross, projected, response = self._addition
        z = x[:, block]
        schur = z.T @ z + self.alpha * np.eye(len(block)) - cross[:, block].T @ projected[:, block]
        gamma = np.linalg.solve(schur, response[block])
        return self.prediction + residual_test[:, block] @ gamma

    def remove(self, name):
        block = self.blocks[name]
        if not block:
            return self.prediction.copy()
        adjustment = np.linalg.solve(self.inverse[np.ix_(block, block)], self.coef[block])
        return self.prediction - self.test_inverse[:, block] @ adjustment


def supported_years(raw, years):
    return sum(np.isfinite(raw[years == y]).sum() >= 10 for y in np.unique(years))


def deduplicate(x, names, blocks):
    """Exact equal training blocks only; no access to test values or labels."""
    retained, seen = [], set()
    for name in names:
        block = x[:, blocks[name]]
        key = (block.shape[1], block.tobytes())
        if key not in seen:
            retained.append(name)
            seen.add(key)
    return retained


def point_prediction(predicted, years, caps):
    result = np.maximum(predicted, 0) * np.where(years >= 2021, 17, 16)
    result[caps == 0] = 0
    return result


def bh_adjust(p):
    p = np.asarray(p)
    order = np.argsort(p)
    adjusted = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    result = np.empty(len(p))
    result[order] = np.minimum(adjusted, 1)
    return result


def uncertainty(gains, seed=20260923):
    a = np.asarray(gains, dtype=float)
    rng = np.random.default_rng(seed)
    bootstrap = a[rng.integers(0, len(a), (10000, len(a)))].mean(axis=1)
    signs = rng.choice([-1, 1], (10000, len(a)))
    p = (1 + (np.abs((signs * a).mean(axis=1)) >= abs(a.mean()) - 1e-12).sum()) / 10001
    return dict(
        ci_low=float(np.quantile(bootstrap, 0.025)),
        ci_high=float(np.quantile(bootstrap, 0.975)),
        p_value=float(p),
        loo_min=float(((a.sum() - a) / (len(a) - 1)).min()) if len(a) > 1 else None,
    )
