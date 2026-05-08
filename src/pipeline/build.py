"""Assemble the end-to-end Sklearn ``Pipeline``.

The pipeline is the canonical artefact of this project — every model is
trained inside one of these so preprocessing, feature engineering, and
classification are inseparable and reproducible.

Stages:

    raw BTS DataFrame
          ↓
    BookingTimeFeatureBuilder  (drops forbidden cols, derives HOUR, etc.)
          ↓
    HistoricalDelayRateEncoder × 4  (route, carrier, origin, tail)
          ↓
    ColumnTransformer
          - StandardScaler on numerics
          - OneHotEncoder(min_frequency=...) on categoricals
          - CyclicalEncoder on HOUR/DAYOFWEEK/MONTH
          ↓
    classifier  (any sklearn-compatible estimator)
"""
from __future__ import annotations

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..features.booking_time import (
    BOOKING_CATEGORICAL,
    BOOKING_CYCLICAL,
    BOOKING_NUMERIC,
    BookingTimeFeatureBuilder,
)
from ..features.cyclical import CyclicalEncoder
from ..features.historical import HistoricalDelayRateEncoder


class HistoricalFeaturesStack(BaseEstimator, TransformerMixin):
    """Run several HistoricalDelayRateEncoders and append their outputs.

    Each encoder is fit on the same training labels but with a different
    grouping key.  The output is the original DataFrame plus one column per
    (key × window) combination.
    """

    def __init__(
        self,
        key_groups: tuple[tuple[str, ...], ...] = (
            ("ORIGIN", "DEST"),
            ("OP_UNIQUE_CARRIER",),
            ("ORIGIN",),
            ("TAIL_NUM",),
        ),
        windows_days: tuple[int, ...] = (30, 90, 365),
        smoothing: float = 100.0,
    ):
        self.key_groups = key_groups
        self.windows_days = windows_days
        self.smoothing = smoothing

    def fit(self, X, y=None):
        self.encoders_ = []
        self.output_names_ = []
        for keys in self.key_groups:
            enc = HistoricalDelayRateEncoder(
                key_cols=list(keys),
                windows_days=self.windows_days,
                smoothing=self.smoothing,
                output_prefix="DR",
            )
            enc.fit(X, y)
            self.encoders_.append((keys, enc))
            self.output_names_.extend(enc.get_feature_names_out())
        return self

    def transform(self, X):
        out = X.copy()
        for keys, enc in self.encoders_:
            arr = enc.transform(X)
            for j, w in enumerate(self.windows_days):
                out[f"{'_'.join(keys)}_DR_{w}D"] = arr[:, j]
        return out

    def get_feature_names_out(self, input_features=None):
        return self.output_names_


def _make_column_transformer() -> ColumnTransformer:
    historical_numerics = []
    for keys in (("ORIGIN", "DEST"), ("OP_UNIQUE_CARRIER",), ("ORIGIN",), ("TAIL_NUM",)):
        for w in (30, 90, 365):
            historical_numerics.append(f"{'_'.join(keys)}_DR_{w}D")

    numeric_cols = list(BOOKING_NUMERIC) + historical_numerics + ["IS_WEEKEND", "IS_HOLIDAY"]
    categorical_cols = list(BOOKING_CATEGORICAL)
    cyclical_cols = list(BOOKING_CYCLICAL)

    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                ]),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline([
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(
                        handle_unknown="infrequent_if_exist",
                        min_frequency=200,
                        sparse_output=True,
                    )),
                ]),
                categorical_cols,
            ),
            (
                "cyc",
                Pipeline([
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("cyc", CyclicalEncoder()),
                ]),
                cyclical_cols,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_pipeline(
    classifier,
    smoothing: float = 100.0,
) -> Pipeline:
    """Build the canonical end-to-end pipeline around any classifier."""
    return Pipeline([
        ("book", BookingTimeFeatureBuilder()),
        ("hist", HistoricalFeaturesStack(smoothing=smoothing)),
        ("prep", _make_column_transformer()),
        ("clf", classifier),
    ])
