"""Tests for EC261 labelling and compensation logic."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.ec261 import KM_PER_MILE, compensation_eur, compute_compensation, label_eligible_delay


def test_compensation_tiers():
    distances = np.array([0, 1499, 1500, 1501, 3499, 3500, 3501, 9999])
    out = compute_compensation(distances)
    assert out[0] == 250
    assert out[1] == 250
    assert out[2] == 250
    assert out[3] == 400
    assert out[4] == 400
    assert out[5] == 400
    assert out[6] == 600
    assert out[7] == 600


def test_label_long_delay_carrier_attributable():
    df = pd.DataFrame([{
        "ARR_DELAY": 200,
        "CARRIER_DELAY": 200,
        "WEATHER_DELAY": 0,
        "NAS_DELAY": 0,
        "SECURITY_DELAY": 0,
        "LATE_AIRCRAFT_DELAY": 0,
    }])
    assert label_eligible_delay(df).iloc[0] == 1


def test_label_long_delay_weather_exempt():
    """Weather is an extraordinary circumstance — not eligible."""
    df = pd.DataFrame([{
        "ARR_DELAY": 240,
        "CARRIER_DELAY": 30,
        "WEATHER_DELAY": 200,
        "NAS_DELAY": 10,
        "SECURITY_DELAY": 0,
        "LATE_AIRCRAFT_DELAY": 0,
    }])
    assert label_eligible_delay(df).iloc[0] == 0


def test_label_short_delay_never_eligible():
    df = pd.DataFrame([{
        "ARR_DELAY": 100,
        "CARRIER_DELAY": 100,
        "WEATHER_DELAY": 0,
        "NAS_DELAY": 0,
        "SECURITY_DELAY": 0,
        "LATE_AIRCRAFT_DELAY": 0,
    }])
    assert label_eligible_delay(df).iloc[0] == 0


def test_label_late_aircraft_eligible():
    """Late aircraft cascade IS carrier-attributable under EC261."""
    df = pd.DataFrame([{
        "ARR_DELAY": 200,
        "CARRIER_DELAY": 30,
        "WEATHER_DELAY": 0,
        "NAS_DELAY": 0,
        "SECURITY_DELAY": 0,
        "LATE_AIRCRAFT_DELAY": 170,
    }])
    assert label_eligible_delay(df).iloc[0] == 1


def test_label_nan_arr_delay_is_zero():
    df = pd.DataFrame([{
        "ARR_DELAY": np.nan,
        "CARRIER_DELAY": 0,
        "WEATHER_DELAY": 0,
        "NAS_DELAY": 0,
        "SECURITY_DELAY": 0,
        "LATE_AIRCRAFT_DELAY": 0,
    }])
    assert label_eligible_delay(df).iloc[0] == 0


def test_compensation_eur_uses_km_conversion():
    df = pd.DataFrame({"DISTANCE": [500, 1500, 3000]})  # miles
    comp = compensation_eur(df).to_numpy()
    expected = compute_compensation(df["DISTANCE"].to_numpy() * KM_PER_MILE)
    np.testing.assert_array_equal(comp, expected)
