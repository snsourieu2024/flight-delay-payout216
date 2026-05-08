"""Calibration plots and Brier score helpers."""
from __future__ import annotations

import numpy as np


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Mean squared error between predicted probabilities and binary outcomes."""
    return float(np.mean((np.asarray(y_prob, dtype=float)
                          - np.asarray(y_true, dtype=float)) ** 2))


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
) -> float:
    """ECE: weighted mean of |confidence − accuracy| across probability bins."""
    bins = np.linspace(0, 1, n_bins + 1)
    binids = np.clip(np.digitize(y_prob, bins) - 1, 0, n_bins - 1)
    n = len(y_true)
    ece = 0.0
    for b in range(n_bins):
        mask = binids == b
        if not mask.any():
            continue
        conf = float(y_prob[mask].mean())
        acc = float(y_true[mask].mean())
        ece += abs(conf - acc) * (mask.sum() / n)
    return float(ece)
