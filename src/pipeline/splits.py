"""Temporal split helpers.

A *random* k-fold leaks future routes/aircraft/operational regimes into
the past — the most common silent failure in flight-delay ML projects.
We therefore use:

- A year-based train/val/test split (default 2018-2022 / 2023 / 2024).
- For hyperparameter tuning, ``ExpandingTimeSeriesSplit`` over the training
  years which respects time order at every fold boundary.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import TEST_YEARS, TRAIN_YEARS, VAL_YEARS


@dataclass
class TemporalSplit:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


def temporal_split(
    df: pd.DataFrame,
    train_years: tuple[int, ...] = TRAIN_YEARS,
    val_years: tuple[int, ...] = VAL_YEARS,
    test_years: tuple[int, ...] = TEST_YEARS,
    train_frac: float = 0.65,
    val_frac: float = 0.15,
) -> TemporalSplit:
    """Split rows temporally on ``FL_DATE``.

    Uses the configured year-based split when ALL three sets are populated.
    Otherwise degrades to:

    1. **Multi-year fallback** (≥3 unique years) — last 20% of years to test,
       previous 15% to validation.
    2. **Sub-year fallback** (≤2 unique years, e.g. when only one year or one
       month of data is on disk) — quantile split on the raw date so we still
       respect time order. ``train_frac`` and ``val_frac`` control the cuts;
       defaults give 65% / 15% / 20% by date.
    """
    dates = pd.to_datetime(df["FL_DATE"])
    years = dates.dt.year.to_numpy()
    available = sorted(np.unique(years).tolist())

    train_set = set(train_years) & set(available)
    val_set = set(val_years) & set(available)
    test_set = set(test_years) & set(available)

    if train_set and val_set and test_set:
        return TemporalSplit(
            train_idx=np.flatnonzero(np.isin(years, list(train_set))),
            val_idx=np.flatnonzero(np.isin(years, list(val_set))),
            test_idx=np.flatnonzero(np.isin(years, list(test_set))),
        )

    if len(available) >= 3:
        n = len(available)
        n_test = max(1, int(round(n * 0.20)))
        n_val = max(1, int(round(n * 0.15)))
        test_set = set(available[-n_test:])
        val_set = set(available[-(n_test + n_val):-n_test])
        train_set = set(available) - val_set - test_set
        return TemporalSplit(
            train_idx=np.flatnonzero(np.isin(years, list(train_set))),
            val_idx=np.flatnonzero(np.isin(years, list(val_set))),
            test_idx=np.flatnonzero(np.isin(years, list(test_set))),
        )

    timestamps = dates.astype("int64").to_numpy()
    train_cut = np.quantile(timestamps, train_frac)
    val_cut = np.quantile(timestamps, train_frac + val_frac)
    return TemporalSplit(
        train_idx=np.flatnonzero(timestamps <= train_cut),
        val_idx=np.flatnonzero((timestamps > train_cut) & (timestamps <= val_cut)),
        test_idx=np.flatnonzero(timestamps > val_cut),
    )


class ExpandingTimeSeriesSplit:
    """Expanding-window CV that respects FL_DATE order.

    Used inside ``GridSearchCV`` / ``BayesSearchCV`` so hyperparameter
    selection itself does not leak future data into past folds.
    """

    def __init__(self, n_splits: int = 4, gap_days: int = 0):
        self.n_splits = n_splits
        self.gap_days = gap_days

    def split(self, X, y=None, groups=None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        if not isinstance(X, pd.DataFrame) or "FL_DATE" not in X.columns:
            raise ValueError(
                "ExpandingTimeSeriesSplit requires a DataFrame with FL_DATE column"
            )
        days = (
            pd.to_datetime(X["FL_DATE"]).values.astype("datetime64[D]").astype("int64")
        )
        order = np.argsort(days, kind="mergesort")
        days_sorted = days[order]
        unique_days = np.unique(days_sorted)
        if len(unique_days) <= self.n_splits + 1:
            return
        boundaries = np.linspace(0, len(unique_days), self.n_splits + 2)[1:-1].astype(int)
        for cut in boundaries:
            t_cut = unique_days[cut]
            train_mask = days < t_cut
            val_mask = (days >= t_cut + self.gap_days) & (days < t_cut + 30 + self.gap_days)
            yield np.flatnonzero(train_mask), np.flatnonzero(val_mask)

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits
