"""Tests for the BTS PREZIP → internal schema normaliser."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import BTS_COLUMNS
from src.data.bts_schema import BTS_PREZIP_RENAME, normalize_bts_columns


def _prezip_row() -> dict:
    """A realistic single-row BTS PREZIP dict (camelCase column names)."""
    return {
        "Year": 2024, "Quarter": 1, "Month": 1, "DayofMonth": 8, "DayOfWeek": 1,
        "FlightDate": "2024-01-08",
        "Reporting_Airline": "9E",
        "DOT_ID_Reporting_Airline": 20363, "IATA_CODE_Reporting_Airline": "9E",
        "Tail_Number": "N485PX",
        "Flight_Number_Reporting_Airline": 4801,
        "OriginAirportID": 12953, "OriginAirportSeqID": 1295304,
        "OriginCityMarketID": 31703, "Origin": "LGA", "OriginCityName": "New York, NY",
        "OriginState": "NY", "OriginStateFips": 36, "OriginStateName": "New York", "OriginWac": 22,
        "DestAirportID": 13871, "DestAirportSeqID": 1387102, "DestCityMarketID": 33316,
        "Dest": "OMA", "DestCityName": "Omaha, NE", "DestState": "NE",
        "DestStateFips": 31, "DestStateName": "Nebraska", "DestWac": 65,
        "CRSDepTime": 856, "DepTime": 851.0, "DepDelay": -5.0,
        "CRSArrTime": 1135, "ArrTime": 1124.0, "ArrDelay": -11.0,
        "Cancelled": 0.0, "Diverted": 0.0,
        "CRSElapsedTime": 219.0, "Distance": 1148.0,
        "CarrierDelay": np.nan, "WeatherDelay": np.nan, "NASDelay": np.nan,
        "SecurityDelay": np.nan, "LateAircraftDelay": np.nan,
    }


def test_rename_map_is_complete_with_respect_to_internal_schema():
    """Every internal BTS_COLUMNS name must be a target of the rename map."""
    targets = set(BTS_PREZIP_RENAME.values())
    missing = [c for c in BTS_COLUMNS if c not in targets]
    assert not missing, f"Internal columns missing from BTS_PREZIP_RENAME: {missing}"


def test_normalize_renames_and_keeps_only_required_columns():
    df = pd.DataFrame([_prezip_row()])
    out = normalize_bts_columns(df)
    assert list(out.columns) == BTS_COLUMNS
    assert len(out) == 1


def test_normalize_parses_dates_and_dtypes():
    df = pd.DataFrame([_prezip_row()])
    out = normalize_bts_columns(df)
    assert pd.api.types.is_datetime64_any_dtype(out["FL_DATE"])
    assert out["FL_DATE"].iloc[0] == pd.Timestamp("2024-01-08")
    assert out["DISTANCE"].iloc[0] == 1148
    assert pd.isna(out["CARRIER_DELAY"].iloc[0])


def test_normalize_idempotent_on_already_internal_schema():
    """Re-running on already-normalised data must be a no-op (used by the loader)."""
    df = pd.DataFrame([_prezip_row()])
    once = normalize_bts_columns(df)
    twice = normalize_bts_columns(once)
    pd.testing.assert_frame_equal(once.reset_index(drop=True), twice.reset_index(drop=True))


def test_normalize_raises_on_missing_required_column():
    row = _prezip_row()
    del row["CarrierDelay"]
    df = pd.DataFrame([row])
    with pytest.raises(ValueError, match="missing"):
        normalize_bts_columns(df)
