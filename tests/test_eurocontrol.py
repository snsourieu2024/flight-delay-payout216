"""Tests for the EUROCONTROL R&D Archive loader."""
from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.eurocontrol import (
    EU_OUTPUT_COLS,
    NM_PER_MILE,
    find_flights_file,
    load_eu_processed,
    load_eurocontrol_flights,
    process_eurocontrol_to_parquet,
)


SAMPLE_HEADER = (
    '"ECTRL ID","ADEP","ADEP Latitude","ADEP Longitude","ADES","ADES Latitude",'
    '"ADES Longitude","FILED OFF BLOCK TIME","FILED ARRIVAL TIME",'
    '"ACTUAL OFF BLOCK TIME","ACTUAL ARRIVAL TIME","AC Type","AC Operator",'
    '"AC Registration","ICAO Flight Type","STATFOR Market Segment",'
    '"Requested FL","Actual Distance Flown (nm)"'
)

# Three rows that exercise: typical short-haul, long-haul, and a midnight wrap.
SAMPLE_ROWS = [
    # Short-haul intra-EU (~1100km)
    '"100001","EGLL","51.4775","-0.4614","LFPG","49.00972","2.54778",'
    '"01-03-2023 06:00:00","01-03-2023 07:30:00",'
    '"01-03-2023 06:08:00","01-03-2023 07:42:00",'
    '"A320","BAW","GEUUC","S","Traditional Scheduled","350","185"',
    # Long-haul transatlantic (~3000nm = ~3450 statute miles)
    '"100002","KIAD","38.945","-77.45667","LFPG","49.00972","2.54778",'
    '"01-03-2023 00:00:00","01-03-2023 07:18:01",'
    '"01-03-2023 00:04:38","01-03-2023 07:27:53",'
    '"B77W","AFR","FGSQC","S","Traditional Scheduled","310","3000"',
    # Midnight wrap (filed 23:30, arrives 01:15 next day, no actuals reported)
    '"100003","EDDF","50.03333","8.57056","LIRF","41.7999","12.2462",'
    '"15-03-2023 23:30:00","16-03-2023 01:15:00","",""'
    ',"A321","DLH","DAIDA","S","Not Classified","370","700"',
]


@pytest.fixture()
def eu_drop(tmp_path: Path) -> Path:
    """Build a fake monthly drop in tmp_path / 202303 / Flights_*.csv.gz."""
    drop_dir = tmp_path / "202303"
    drop_dir.mkdir()
    flights_path = drop_dir / "Flights_20230301_20230331.csv.gz"

    body = "\n".join([SAMPLE_HEADER, *SAMPLE_ROWS])
    with gzip.open(flights_path, "wt", encoding="utf-8") as fh:
        fh.write(body)
    return drop_dir


def test_find_flights_file_returns_unique_match(eu_drop: Path):
    path = find_flights_file(eu_drop)
    assert path.name == "Flights_20230301_20230331.csv.gz"


def test_find_flights_file_rejects_empty_directory(tmp_path: Path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="No Flights"):
        find_flights_file(empty)


def test_find_flights_file_rejects_multiple_matches(tmp_path: Path):
    d = tmp_path / "202303"
    d.mkdir()
    (d / "Flights_20230301_20230331.csv.gz").touch()
    (d / "Flights_20230101_20230131.csv.gz").touch()
    with pytest.raises(ValueError, match="Multiple"):
        find_flights_file(d)


def test_load_returns_expected_schema(eu_drop: Path):
    df = load_eurocontrol_flights(eu_drop)
    assert list(df.columns) == EU_OUTPUT_COLS
    assert len(df) == 3


def test_load_computes_arr_delay_in_minutes(eu_drop: Path):
    df = load_eurocontrol_flights(eu_drop)

    # Short-haul row: filed 07:30:00, actual 07:42:00 => 12 min late.
    short_haul = df[df["TAIL_NUM"] == "GEUUC"].iloc[0]
    assert short_haul["ARR_DELAY"] == pytest.approx(12.0, abs=0.01)
    assert short_haul["DEP_DELAY"] == pytest.approx(8.0, abs=0.01)
    assert short_haul["ORIGIN"] == "EGLL"
    assert short_haul["DEST"] == "LFPG"
    assert short_haul["CRS_DEP_TIME"] == 600
    assert short_haul["DEP_TIME"] == 608

    # Long-haul row: filed arr 07:18:01, actual 07:27:53 => 9.87 min.
    long_haul = df[df["TAIL_NUM"] == "FGSQC"].iloc[0]
    assert long_haul["ARR_DELAY"] == pytest.approx(9.87, abs=0.05)


def test_load_marks_cancelled_when_actuals_missing(eu_drop: Path):
    df = load_eurocontrol_flights(eu_drop)
    cancelled = df[df["TAIL_NUM"] == "DAIDA"].iloc[0]
    assert cancelled["CANCELLED"] == 1
    assert pd.isna(cancelled["ARR_DELAY"])
    assert cancelled["DIVERTED"] == 0


def test_load_converts_distance_nm_to_statute_miles(eu_drop: Path):
    df = load_eurocontrol_flights(eu_drop)
    long_haul = df[df["TAIL_NUM"] == "FGSQC"].iloc[0]
    expected_miles = round(3000 * NM_PER_MILE)
    assert long_haul["DISTANCE"] == expected_miles
    assert long_haul["ACTUAL_DISTANCE_NM"] == 3000


def test_load_zeroes_cause_codes_when_unavailable(eu_drop: Path):
    df = load_eurocontrol_flights(eu_drop)
    for col in (
        "CARRIER_DELAY",
        "WEATHER_DELAY",
        "NAS_DELAY",
        "SECURITY_DELAY",
        "LATE_AIRCRAFT_DELAY",
    ):
        assert (df[col] == 0).all(), f"{col} should be zeroed for R&D Archive data"


def test_load_fl_date_is_normalised_to_midnight(eu_drop: Path):
    df = load_eurocontrol_flights(eu_drop)
    assert (df["FL_DATE"].dt.hour == 0).all()
    assert (df["FL_DATE"].dt.minute == 0).all()


def test_process_to_parquet_round_trip(eu_drop: Path, tmp_path: Path):
    dst = tmp_path / "cache"
    written = process_eurocontrol_to_parquet(eu_drop.parent, dst_dir=dst)
    assert len(written) == 1
    assert written[0].name == "eurocontrol_2023_03.parquet"

    loaded = load_eu_processed(raw_dir=dst)
    assert list(loaded.columns) == EU_OUTPUT_COLS
    assert len(loaded) == 3


def test_process_skips_existing_parquet(eu_drop: Path, tmp_path: Path):
    dst = tmp_path / "cache"
    process_eurocontrol_to_parquet(eu_drop.parent, dst_dir=dst)
    mtime_before = (dst / "eurocontrol_2023_03.parquet").stat().st_mtime
    process_eurocontrol_to_parquet(eu_drop.parent, dst_dir=dst, overwrite=False)
    mtime_after = (dst / "eurocontrol_2023_03.parquet").stat().st_mtime
    assert mtime_before == mtime_after


def test_load_eu_processed_filters_months(eu_drop: Path, tmp_path: Path):
    dst = tmp_path / "cache"
    process_eurocontrol_to_parquet(eu_drop.parent, dst_dir=dst)
    out = load_eu_processed(months=["2023-03"], raw_dir=dst)
    assert len(out) == 3
    with pytest.raises(FileNotFoundError):
        load_eu_processed(months=["1999-12"], raw_dir=dst)


def test_load_eu_sample_prefers_eurocontrol_parquet_over_synthetic(
    eu_drop: Path, tmp_path: Path
):
    """``load_eu_sample`` should pick up EUROCONTROL parquets and skip the synthetic generator."""
    from src.data.loaders import load_eu_sample

    dst = tmp_path / "cache"
    process_eurocontrol_to_parquet(eu_drop.parent, dst_dir=dst)

    df = load_eu_sample(raw_dir=dst)
    assert len(df) == 3
    assert "ICAO_FLIGHT_TYPE" in df.columns, "Expected EUROCONTROL columns, not synthetic"
