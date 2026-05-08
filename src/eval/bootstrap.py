"""Bootstrap CIs for profit and classification metrics."""
from __future__ import annotations

from typing import Callable

import numpy as np


def bootstrap_metric(
    statistic: Callable[..., float],
    n_resamples: int = 1000,
    seed: int = 0,
    *args,
    **kwargs,
) -> dict[str, float]:
    """Generic non-parametric bootstrap.

    The first positional argument of ``statistic`` is treated as ``y_true``;
    every other argument with the same length is resampled by the same
    indices.  Scalar arguments and arguments with length != n are passed
    through unchanged.
    """
    if not args:
        raise ValueError("Provide y_true as the first positional argument")
    n = len(args[0])
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        rs_args = [a[idx] if hasattr(a, "__len__") and len(a) == n else a for a in args]
        rs_kwargs = {
            k: (v[idx] if hasattr(v, "__len__") and len(v) == n else v)
            for k, v in kwargs.items()
        }
        samples.append(statistic(*rs_args, **rs_kwargs))
    samples = np.asarray(samples, dtype=float)
    return {
        "point": float(statistic(*args, **kwargs)),
        "mean": float(samples.mean()),
        "std": float(samples.std()),
        "ci_low_95": float(np.percentile(samples, 2.5)),
        "ci_high_95": float(np.percentile(samples, 97.5)),
        "n_resamples": int(n_resamples),
    }
