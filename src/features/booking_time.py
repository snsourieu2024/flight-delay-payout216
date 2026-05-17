"""Booking-time feature builder.

Every feature emitted by this transformer must be available to a hypothetical
buyer at the moment they purchase a ticket — i.e. ``BOOKING_HORIZON_DAYS``
before scheduled departure.  Anything that materialises later (actual
departure delay, actual weather at gate, cause-code minutes, …) is forbidden.

The leakage audit table in ``notebooks/02_feature_engineering.ipynb`` walks
through every column in BTS and explicitly states whether it is allowed
("booking-time available") or forbidden ("post-hoc").  This module is the
single source of truth for that audit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


BOOKING_NUMERIC = [
    "DISTANCE",
    "CRS_ELAPSED_TIME",
    "AIRCRAFT_AGE_YEARS",
    "WX_PRECIP_FCST_24H",
    "WX_WIND_FCST_24H",
    "WX_VISIBILITY_FCST_24H",
    "WX_CONVECTIVE_INDEX_24H",
    # Leakage-safe operational features (schedule-derived only; precomputed on
    # the full frame before splitting because rotation spans rows). Absent in
    # the synthetic/smoke path -> created NaN below and median-imputed.
    "SCHED_TURNAROUND_MIN",
    "LEG_OF_DAY",
    "ORIG_HOURLY_DEPARTURES",
]

# Operational features that the pipeline tolerates as missing (filled NaN when
# the enrichment step did not run, e.g. synthetic data / smoke test).
OPERATIONAL_OPTIONAL = [
    "SCHED_TURNAROUND_MIN",
    "LEG_OF_DAY",
    "ORIG_HOURLY_DEPARTURES",
]
BOOKING_CATEGORICAL = [
    "OP_UNIQUE_CARRIER",
    "ORIGIN",
    "DEST",
    "AIRCRAFT_TYPE",
    "DISTANCE_TIER",
]
BOOKING_CYCLICAL = ["HOUR", "DAYOFWEEK", "MONTH"]


FORBIDDEN_COLUMNS = {
    "DEP_TIME": "actual departure time, materialises post-booking",
    "DEP_DELAY": "actual departure delay, parent of label",
    "ARR_TIME": "actual arrival time, materialises post-booking",
    "ARR_DELAY": "actual arrival delay, IS the label numerator",
    "CARRIER_DELAY": "cause-code minutes, generates the label",
    "WEATHER_DELAY": "cause-code minutes, generates the label",
    "NAS_DELAY": "cause-code minutes, generates the label",
    "SECURITY_DELAY": "cause-code minutes, generates the label",
    "LATE_AIRCRAFT_DELAY": "cause-code minutes, generates the label",
    "CANCELLED": "post-hoc operational outcome",
    "DIVERTED": "post-hoc operational outcome",
}


class BookingTimeFeatureBuilder(BaseEstimator, TransformerMixin):
    """Derive booking-time features from raw BTS rows.

    Adds:
      - HOUR, DAYOFWEEK, MONTH (from FL_DATE + CRS_DEP_TIME)
      - IS_WEEKEND, IS_HOLIDAY
      - DISTANCE_TIER (short / medium / long, matching EC261 thresholds)

    Drops every column listed in ``FORBIDDEN_COLUMNS`` so it is impossible
    for downstream code to accidentally feed leakage features into the model.
    """

    US_FED_HOLIDAYS_MMDD = {
        (1, 1), (1, 20), (2, 17), (5, 26), (7, 4), (9, 1), (10, 13),
        (11, 11), (11, 27), (12, 25),
    }

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        date = pd.to_datetime(df["FL_DATE"])
        crs = df["CRS_DEP_TIME"].fillna(0).astype(int)
        df["HOUR"] = crs // 100
        df["MINUTE"] = crs % 100
        df["DAYOFWEEK"] = date.dt.dayofweek
        df["MONTH"] = date.dt.month
        df["IS_WEEKEND"] = (df["DAYOFWEEK"] >= 5).astype(int)
        df["IS_HOLIDAY"] = [
            int((d.month, d.day) in self.US_FED_HOLIDAYS_MMDD) for d in date
        ]
        df["DISTANCE_TIER"] = pd.cut(
            df["DISTANCE"].fillna(0),
            bins=[-1, 932, 2175, 100_000],
            labels=["short", "medium", "long"],
        ).astype(str)

        if "AIRCRAFT_AGE_YEARS" not in df.columns:
            df["AIRCRAFT_AGE_YEARS"] = np.nan
        if "AIRCRAFT_TYPE" not in df.columns:
            df["AIRCRAFT_TYPE"] = "UNKNOWN"
        for c in ("WX_PRECIP_FCST_24H", "WX_WIND_FCST_24H",
                  "WX_VISIBILITY_FCST_24H", "WX_CONVECTIVE_INDEX_24H"):
            if c not in df.columns:
                df[c] = np.nan
        for c in OPERATIONAL_OPTIONAL:
            if c not in df.columns:
                df[c] = np.nan

        df = df.drop(columns=[c for c in FORBIDDEN_COLUMNS if c in df.columns])
        return df


def leakage_audit_table() -> pd.DataFrame:
    """Return the audit table used in the EDA notebook and report."""
    rows = []
    for col in BOOKING_NUMERIC + BOOKING_CATEGORICAL:
        rows.append({"column": col, "status": "allowed",
                     "reason": "available at booking horizon"})
    for col in BOOKING_CYCLICAL:
        rows.append({"column": col, "status": "allowed",
                     "reason": "derived from scheduled time"})
    for col, why in FORBIDDEN_COLUMNS.items():
        rows.append({"column": col, "status": "FORBIDDEN", "reason": why})
    return pd.DataFrame(rows)
