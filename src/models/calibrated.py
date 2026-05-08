"""Probability calibration wrappers.

Expected-value math fails on uncalibrated probabilities.  XGBoost in
particular outputs scores that are NOT probabilities.  This module wraps
fitted estimators with isotonic regression on a held-out calibration fold.
"""
from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline


def calibrate(
    pipeline: Pipeline,
    method: str = "isotonic",
    cv: int = 3,
) -> CalibratedClassifierCV:
    """Wrap a fitted (or unfitted) Pipeline with a probability calibrator.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        Any classifier-ending pipeline.  May or may not be fit.
    method : {"isotonic", "sigmoid"}
        Isotonic is non-parametric and usually wins on tabular data.
        Sigmoid (Platt) is faster and works on tiny calibration sets.
    cv : int
        ``CalibratedClassifierCV`` will fit ``cv`` clones of the base
        estimator and average their calibrated outputs.  Use cv="prefit"
        when the pipeline was fit on a separate calibration fold.

    Returns
    -------
    sklearn.calibration.CalibratedClassifierCV
    """
    return CalibratedClassifierCV(estimator=pipeline, method=method, cv=cv)


def reliability_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (mean_predicted, fraction_positives, bin_count) for a reliability diagram."""
    bins = np.linspace(0, 1, n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    binids = np.clip(binids, 0, n_bins - 1)

    mean_pred = np.zeros(n_bins)
    frac_pos = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        mask = binids == b
        if mask.any():
            mean_pred[b] = y_prob[mask].mean()
            frac_pos[b] = y_true[mask].mean()
            counts[b] = int(mask.sum())
    return mean_pred, frac_pos, counts
