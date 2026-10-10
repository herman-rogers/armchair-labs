"""Serializable per-calendar-week predictor for the bounded QB comparison."""

import numpy as np


class PredictionPolicy:
    def __init__(self, members, *, count=False, rate=None, rate_indices=None, prior_index=None):
        self.members = members
        self.count = count
        self.rate = rate
        self.rate_indices = rate_indices
        self.prior_index = prior_index

    def predict(self, x):
        values = []
        for member in self.members:
            value = member["model"].predict(x[:, member["indices"]]).astype(float)
            values.append(np.maximum(value, 0) if self.count else value)
        result = np.mean(values, axis=0)
        if self.rate is not None:
            fitted = np.clip(self.rate.predict(x[:, self.rate_indices]), 0, 20)
            rate = 0.5 * fitted + 0.5 * x[:, self.prior_index]
            result = result * rate
        return result
