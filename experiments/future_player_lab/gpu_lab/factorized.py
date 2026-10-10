"""Opportunity × efficiency challenger, trained using earlier observed outcomes only."""

from __future__ import annotations

import numpy as np
from experiments.future_player_lab.gpu_backends import build_estimator

DENOMINATORS = {
    "rushing_yards": "carries",
    "rushing_tds": "carries",
    "receiving_yards": "targets",
    "receiving_tds": "targets",
    "receptions": "targets",
}


class FactorizedTree:
    """Two trees; product of conditional means is a tested approximation, not an identity."""

    def __init__(self, recipe, target, threads):
        self.recipe, self.target, self.threads = recipe, target, threads
        self.metadata = {"target_mode": "opportunity_times_efficiency", "components": []}

    def fit(self, x, y, opportunity):
        known = np.isfinite(y) & np.isfinite(opportunity)
        positive = known & (opportunity > 0)
        if positive.sum() < 20:
            raise ValueError("Insufficient positive-opportunity labels for efficiency fit")
        self.volume, meta1 = build_estimator(
            self.recipe["family"],
            self.recipe["params"],
            seed=self.recipe["seed"],
            device=self.recipe["device"],
            threads=self.threads,
        )
        self.efficiency, meta2 = build_estimator(
            self.recipe["family"],
            self.recipe["params"],
            seed=self.recipe["seed"] + 1,
            device=self.recipe["device"],
            threads=self.threads,
        )
        self.volume.fit(x[known], opportunity[known])
        # Exposure weighting stabilizes ratios with one or two observed opportunities.
        weights = opportunity[positive] / opportunity[positive].mean()
        self.efficiency.fit(x[positive], y[positive] / opportunity[positive], sample_weight=weights)
        self.metadata["components"] = [meta1, meta2]
        return self

    def predict(self, x):
        volume = np.maximum(0, self.volume.predict(x))
        efficiency = self.efficiency.predict(x)
        if self.target in {"receptions", "rushing_tds", "receiving_tds"}:
            efficiency = np.clip(efficiency, 0, 1)
        return volume * efficiency
