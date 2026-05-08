"""Sin/cos cyclical encoders for hour, day-of-week, month."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CyclicalEncoder(BaseEstimator, TransformerMixin):
    """Encode integer cyclical columns as (sin, cos) pairs.

    Periods default to (24, 7, 12) for HOUR, DAYOFWEEK, MONTH respectively.
    Pass an explicit ``periods`` mapping for other columns.

    Output columns are renamed ``<col>_SIN`` and ``<col>_COS``.
    """

    DEFAULT_PERIODS = {"HOUR": 24, "DAYOFWEEK": 7, "MONTH": 12}

    def __init__(self, periods: dict[str, int] | None = None):
        self.periods = periods or self.DEFAULT_PERIODS

    def fit(self, X, y=None):
        self.feature_names_in_ = list(X.columns) if isinstance(X, pd.DataFrame) else None
        return self

    def transform(self, X) -> np.ndarray:
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names_in_)
        out = []
        for col in X.columns:
            period = self.periods.get(col, 24)
            theta = 2 * np.pi * X[col].astype(float).to_numpy() / period
            out.append(np.sin(theta))
            out.append(np.cos(theta))
        return np.stack(out, axis=1)

    def get_feature_names_out(self, input_features=None) -> list[str]:
        cols = input_features if input_features is not None else self.feature_names_in_
        names: list[str] = []
        for c in cols:
            names.extend([f"{c}_SIN", f"{c}_COS"])
        return names
