"""Train-fold-only rolling delay statistics.

The single most common source of leakage in flight-delay projects is computing
"average delay per route/airline/airport" over the entire dataset and then
using it as a feature.  This module computes those statistics **only from
training-fold rows** and exposes a Sklearn-compatible transformer that learns
the lookup table at fit-time and applies it (with a fallback prior) at
transform-time.

It also enforces a *temporal* gap: when computing the rolling delay rate for
flight ``f`` on date ``t``, we only count flights with date < t (no
same-day data, since that would peek at near-future state of the system).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from ..data.ec261 import label_eligible_delay


class HistoricalDelayRateEncoder(BaseEstimator, TransformerMixin):
    """Compute per-key historical EC261-eligible delay rates.

    Parameters
    ----------
    key_cols : list[str]
        Columns to group by, e.g. ``["ORIGIN", "DEST"]`` for route delay.
    windows_days : list[int]
        Rolling windows to emit, e.g. ``[30, 90, 365]``.
    smoothing : float
        Beta-smoothing prior weight; the global base rate is added with this
        weight so that rare keys default to the population mean instead of 0.
    output_prefix : str
        Prefix for output column names.
    """

    def __init__(
        self,
        key_cols: list[str],
        windows_days: tuple[int, ...] = (30, 90, 365),
        smoothing: float = 100.0,
        output_prefix: str = "DELAY_RATE",
    ):
        self.key_cols = key_cols
        self.windows_days = tuple(windows_days)
        self.smoothing = smoothing
        self.output_prefix = output_prefix

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None):
        if y is None:
            y = label_eligible_delay(X)
        self.global_rate_ = float(np.asarray(y).mean())

        df = X[["FL_DATE", *self.key_cols]].copy()
        df["_y"] = np.asarray(y).astype(int)
        df["_t"] = (
            pd.to_datetime(df["FL_DATE"]).values.astype("datetime64[D]").astype("int64")
        )
        df = df.sort_values("_t").reset_index(drop=True)

        self.lookup_: dict[tuple, dict] = {}
        for key, sub in df.groupby(self.key_cols, sort=False, observed=True):
            t_arr = sub["_t"].to_numpy()
            cum_y = np.concatenate([[0], sub["_y"].to_numpy().cumsum()])
            self.lookup_[key if isinstance(key, tuple) else (key,)] = {
                "t": t_arr,
                "cum_y": cum_y,
            }
        self.feature_names_out_ = [
            f"{self.output_prefix}_{w}D" for w in self.windows_days
        ]
        return self

    def _rate_for_key(self, key: tuple, day_now: int, window: int) -> float:
        entry = self.lookup_.get(key)
        if entry is None:
            return self.global_rate_
        t_arr = entry["t"]
        cum_y = entry["cum_y"]
        end = np.searchsorted(t_arr, day_now, side="left")
        start = np.searchsorted(t_arr, day_now - window, side="left")
        n = end - start
        if n <= 0:
            return self.global_rate_
        positives = cum_y[end] - cum_y[start]
        return (positives + self.smoothing * self.global_rate_) / (n + self.smoothing)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        df = X[["FL_DATE", *self.key_cols]].copy()
        df["_t"] = (
            pd.to_datetime(df["FL_DATE"]).values.astype("datetime64[D]").astype("int64")
        )
        out = np.empty((len(df), len(self.windows_days)), dtype=float)
        keys = list(zip(*[df[c] for c in self.key_cols]))
        for i, (key, day) in enumerate(zip(keys, df["_t"].to_numpy())):
            for j, w in enumerate(self.windows_days):
                out[i, j] = self._rate_for_key(key, int(day), int(w))
        return out

    def get_feature_names_out(self, input_features=None) -> list[str]:
        joined = "_".join(c.upper() for c in self.key_cols)
        return [f"{joined}_{self.output_prefix}_{w}D" for w in self.windows_days]
