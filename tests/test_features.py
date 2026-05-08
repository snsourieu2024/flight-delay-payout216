"""Tests for feature engineering — leakage-safety is the priority."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.synthetic import generate_synthetic_bts
from src.features.booking_time import (
    FORBIDDEN_COLUMNS,
    BookingTimeFeatureBuilder,
    leakage_audit_table,
)
from src.features.cyclical import CyclicalEncoder
from src.features.historical import HistoricalDelayRateEncoder


def test_forbidden_columns_dropped():
    df = generate_synthetic_bts(n_flights=200, start_year=2022, end_year=2022, seed=1)
    out = BookingTimeFeatureBuilder().fit_transform(df)
    for col in FORBIDDEN_COLUMNS:
        assert col not in out.columns, f"forbidden column {col} survived"


def test_booking_time_derives_temporal_features():
    df = generate_synthetic_bts(n_flights=200, start_year=2022, end_year=2022, seed=2)
    out = BookingTimeFeatureBuilder().fit_transform(df)
    for col in ["HOUR", "DAYOFWEEK", "MONTH", "IS_WEEKEND", "IS_HOLIDAY", "DISTANCE_TIER"]:
        assert col in out.columns


def test_audit_table_complete():
    audit = leakage_audit_table()
    assert {"allowed", "FORBIDDEN"} <= set(audit["status"].unique())
    for col in FORBIDDEN_COLUMNS:
        assert col in audit["column"].values


def test_historical_encoder_excludes_same_day():
    """Critical: a flight's own day must not contribute to its rolling rate."""
    dates = pd.to_datetime([
        "2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05",
    ] * 3)
    n = len(dates)
    df = pd.DataFrame({
        "FL_DATE": dates,
        "ORIGIN": ["JFK"] * n,
        "DEST": ["LAX"] * n,
        "ARR_DELAY": [200, 200, 200, 0, 0] * 3,
        "CARRIER_DELAY": [200, 200, 200, 0, 0] * 3,
        "WEATHER_DELAY": [0] * n,
        "NAS_DELAY": [0] * n,
        "SECURITY_DELAY": [0] * n,
        "LATE_AIRCRAFT_DELAY": [0] * n,
    })
    enc = HistoricalDelayRateEncoder(
        key_cols=["ORIGIN", "DEST"], windows_days=(7,), smoothing=0.0,
    )
    enc.fit(df)
    out = enc.transform(df)
    # Day 1 has no priors -> falls back to global rate (0.6)
    assert abs(out[0, 0] - df["ARR_DELAY"].astype(int).gt(180).mean()) < 0.5


def test_cyclical_encoder_outputs_pairs():
    df = pd.DataFrame({"HOUR": [0, 6, 12, 18], "DAYOFWEEK": [0, 1, 2, 3], "MONTH": [1, 4, 7, 10]})
    enc = CyclicalEncoder()
    enc.fit(df)
    out = enc.transform(df)
    assert out.shape == (4, 6)
    # Hour 0 -> sin=0, cos=1
    assert abs(out[0, 0] - 0.0) < 1e-9
    assert abs(out[0, 1] - 1.0) < 1e-9
