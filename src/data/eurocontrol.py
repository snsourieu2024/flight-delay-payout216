"""EUROCONTROL R&D Data Archive loader.

The R&D Data Archive (https://www.eurocontrol.int/dashboard/rnd-data-archive)
distributes a free, login-free version of the data Eurocontrol uses internally
for ATM research. Each monthly drop ships a directory of gzipped CSVs, of which
``Flights_YYYYMMDD_YYYYMMDD.csv.gz`` is the per-flight schedule + actuals file
this project needs.

R&D Archive flight columns:

    ECTRL ID, ADEP, ADEP Latitude, ADEP Longitude, ADES, ADES Latitude,
    ADES Longitude, FILED OFF BLOCK TIME, FILED ARRIVAL TIME,
    ACTUAL OFF BLOCK TIME, ACTUAL ARRIVAL TIME, AC Type, AC Operator,
    AC Registration, ICAO Flight Type, STATFOR Market Segment,
    Requested FL, Actual Distance Flown (nm)

The loader maps these to the project's internal BTS-style schema so the rest
of the pipeline (label builder, profit metric, transfer-validation notebook)
can consume them with no special-casing. Cause-code columns are filled with
zero because the R&D Archive does not publish IATA delay reason codes — the
report's section 4.3 / notebook 06 use the softer ``ARR_DELAY >= 180`` proxy
when scoring on EU operations, exactly because of this gap.

Public API
----------
``load_eurocontrol_flights(directory)``
    Read one Flights_*.csv.gz file into a BTS-shaped DataFrame.

``process_eurocontrol_to_parquet(src, dst)``
    Convert a directory of monthly drops into per-month parquet caches under
    ``data/raw/`` so subsequent loads are ~30x faster.

``load_eu_processed(months=None)``
    Read the cached parquets, optionally restricted to a subset of months
    (``["2023-03", "2024-03"]``).
"""
from __future__ import annotations

import gzip
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from ..config import RAW_DIR


NM_PER_MILE = 1.15077945
EUROCONTROL_DATE_FMT = "%d-%m-%Y %H:%M:%S"

EC_RAW_COLS = [
    "ECTRL ID",
    "ADEP",
    "ADEP Latitude",
    "ADEP Longitude",
    "ADES",
    "ADES Latitude",
    "ADES Longitude",
    "FILED OFF BLOCK TIME",
    "FILED ARRIVAL TIME",
    "ACTUAL OFF BLOCK TIME",
    "ACTUAL ARRIVAL TIME",
    "AC Type",
    "AC Operator",
    "AC Registration",
    "ICAO Flight Type",
    "STATFOR Market Segment",
    "Requested FL",
    "Actual Distance Flown (nm)",
]

# Final schema — BTS-compatible plus a few EU-only metadata columns the
# transfer-validation notebook keeps for slicing (operator country code
# is buried in ICAO callsign so we just keep AC_OPERATOR raw).
EU_OUTPUT_COLS = [
    "FL_DATE",
    "OP_UNIQUE_CARRIER",
    "TAIL_NUM",
    "OP_CARRIER_FL_NUM",
    "ORIGIN",
    "DEST",
    "CRS_DEP_TIME",
    "DEP_TIME",
    "DEP_DELAY",
    "CRS_ARR_TIME",
    "ARR_TIME",
    "ARR_DELAY",
    "CANCELLED",
    "DIVERTED",
    "CRS_ELAPSED_TIME",
    "DISTANCE",
    "CARRIER_DELAY",
    "WEATHER_DELAY",
    "NAS_DELAY",
    "SECURITY_DELAY",
    "LATE_AIRCRAFT_DELAY",
    "AIRCRAFT_TYPE",
    "ADEP_LAT",
    "ADEP_LON",
    "ADES_LAT",
    "ADES_LON",
    "ICAO_FLIGHT_TYPE",
    "STATFOR_MARKET_SEGMENT",
    "ACTUAL_DISTANCE_NM",
    "REQUESTED_FL",
]


_FILENAME_RE = re.compile(r"Flights_(\d{8})_(\d{8})\.csv(\.gz)?$")


def find_flights_file(directory: Path) -> Path:
    """Return the path to the single ``Flights_*.csv(.gz)`` file in ``directory``."""
    directory = Path(directory)
    candidates = [p for p in directory.iterdir() if _FILENAME_RE.search(p.name)]
    if not candidates:
        raise FileNotFoundError(
            f"No Flights_*.csv.gz file in {directory}. "
            "Expected a EUROCONTROL R&D Archive monthly drop."
        )
    if len(candidates) > 1:
        raise ValueError(
            f"Multiple Flights_*.csv files in {directory}: {candidates}. "
            "Keep only one per directory."
        )
    return candidates[0]


def _hhmm_int(series: pd.Series) -> pd.Series:
    """Convert a datetime series to BTS-style HHMM integer (e.g. 14:30 -> 1430)."""
    hours = series.dt.hour.astype("Int64")
    minutes = series.dt.minute.astype("Int64")
    return (hours * 100 + minutes).astype("Int64")


def _minutes_diff(later: pd.Series, earlier: pd.Series) -> pd.Series:
    """Difference in whole minutes, NaN-safe.

    Late-night arrivals can wrap past midnight. We do *not* re-bucket the date
    because both sides are full timestamps — the wraparound is already encoded.
    """
    delta = (later - earlier).dt.total_seconds() / 60.0
    return delta


def load_eurocontrol_flights(directory: str | Path) -> pd.DataFrame:
    """Load one monthly EUROCONTROL R&D Archive drop into a BTS-shaped frame.

    Parameters
    ----------
    directory : path-like
        A folder like ``202303/`` that contains a ``Flights_YYYYMMDD_YYYYMMDD.csv.gz``
        and the matching ``FIR``/``Route``/etc. files (which we ignore).

    Returns
    -------
    pd.DataFrame
        Rows in the schema declared in :data:`EU_OUTPUT_COLS`. Cause-code
        columns are filled with 0 (R&D Archive does not publish cause codes).
    """
    directory = Path(directory)
    path = find_flights_file(directory)

    # Only read the columns we need. ``low_memory=False`` to avoid mixed dtype
    # inference on partially-quoted numeric columns.
    raw = pd.read_csv(
        path,
        usecols=EC_RAW_COLS,
        compression="gzip" if path.suffix == ".gz" else None,
        low_memory=False,
    )

    out = pd.DataFrame(index=raw.index)

    # Parse all four timestamps once.
    filed_off = pd.to_datetime(
        raw["FILED OFF BLOCK TIME"], format=EUROCONTROL_DATE_FMT, errors="coerce"
    )
    filed_arr = pd.to_datetime(
        raw["FILED ARRIVAL TIME"], format=EUROCONTROL_DATE_FMT, errors="coerce"
    )
    actual_off = pd.to_datetime(
        raw["ACTUAL OFF BLOCK TIME"], format=EUROCONTROL_DATE_FMT, errors="coerce"
    )
    actual_arr = pd.to_datetime(
        raw["ACTUAL ARRIVAL TIME"], format=EUROCONTROL_DATE_FMT, errors="coerce"
    )

    # FL_DATE = date of scheduled (filed) off-block, normalised to midnight,
    # tz-naive. Matches BTS's FL_DATE convention (date of departure).
    out["FL_DATE"] = filed_off.dt.normalize()

    out["OP_UNIQUE_CARRIER"] = raw["AC Operator"].astype("string").str.strip()
    out["TAIL_NUM"] = raw["AC Registration"].astype("string").str.strip()
    out["OP_CARRIER_FL_NUM"] = pd.to_numeric(raw["ECTRL ID"], errors="coerce").astype("Int64")
    out["ORIGIN"] = raw["ADEP"].astype("string").str.strip()
    out["DEST"] = raw["ADES"].astype("string").str.strip()

    out["CRS_DEP_TIME"] = _hhmm_int(filed_off)
    out["DEP_TIME"] = _hhmm_int(actual_off)
    out["CRS_ARR_TIME"] = _hhmm_int(filed_arr)
    out["ARR_TIME"] = _hhmm_int(actual_arr)

    out["DEP_DELAY"] = _minutes_diff(actual_off, filed_off)
    out["ARR_DELAY"] = _minutes_diff(actual_arr, filed_arr)

    out["CRS_ELAPSED_TIME"] = (
        _minutes_diff(filed_arr, filed_off).round().astype("Int64")
    )

    # Cancellations and diversions are not flagged in R&D Archive. Rows with
    # null actuals are treated as cancelled so EC261 logic stays defensible.
    cancelled = actual_arr.isna() | actual_off.isna()
    out["CANCELLED"] = cancelled.astype("Int64")
    out["DIVERTED"] = pd.Series(0, index=raw.index, dtype="Int64")

    # nm -> mi to match BTS convention (DISTANCE is in statute miles).
    distance_nm = pd.to_numeric(
        raw["Actual Distance Flown (nm)"], errors="coerce"
    )
    out["DISTANCE"] = (distance_nm * NM_PER_MILE).round().astype("Int64")
    out["ACTUAL_DISTANCE_NM"] = distance_nm

    # Cause-code columns: not available in R&D Archive. Filling with 0 keeps
    # downstream EC261 label code happy (it computes the dominant cause across
    # these columns; with all zeros the EC261-strict label is False, which is
    # the correct fallback for a dataset with no cause codes).
    for col in (
        "CARRIER_DELAY",
        "WEATHER_DELAY",
        "NAS_DELAY",
        "SECURITY_DELAY",
        "LATE_AIRCRAFT_DELAY",
    ):
        out[col] = 0.0

    out["AIRCRAFT_TYPE"] = raw["AC Type"].astype("string").str.strip()
    out["ADEP_LAT"] = pd.to_numeric(raw["ADEP Latitude"], errors="coerce")
    out["ADEP_LON"] = pd.to_numeric(raw["ADEP Longitude"], errors="coerce")
    out["ADES_LAT"] = pd.to_numeric(raw["ADES Latitude"], errors="coerce")
    out["ADES_LON"] = pd.to_numeric(raw["ADES Longitude"], errors="coerce")
    out["ICAO_FLIGHT_TYPE"] = raw["ICAO Flight Type"].astype("string").str.strip()
    out["STATFOR_MARKET_SEGMENT"] = (
        raw["STATFOR Market Segment"].astype("string").str.strip()
    )
    out["REQUESTED_FL"] = pd.to_numeric(raw["Requested FL"], errors="coerce")

    out = out[EU_OUTPUT_COLS]
    return out.sort_values("FL_DATE").reset_index(drop=True)


def process_eurocontrol_to_parquet(
    src_root: str | Path,
    dst_dir: str | Path | None = None,
    overwrite: bool = False,
) -> list[Path]:
    """Convert every ``YYYYMM/`` drop under ``src_root`` to a parquet cache.

    Each output file is written to ``dst_dir/eurocontrol_YYYY_MM.parquet`` and
    can be loaded ~30x faster than re-parsing the CSV.

    Returns the list of parquet paths written (or already on disk when
    ``overwrite`` is False).
    """
    src_root = Path(src_root)
    dst_dir = Path(dst_dir) if dst_dir else RAW_DIR
    dst_dir.mkdir(parents=True, exist_ok=True)

    folders = sorted(
        p
        for p in src_root.iterdir()
        if p.is_dir() and re.fullmatch(r"\d{6}", p.name)
    )
    if not folders:
        raise FileNotFoundError(
            f"No YYYYMM directories under {src_root}. "
            "Expected unzipped EUROCONTROL drops like 202303/, 202403/."
        )

    written: list[Path] = []
    for folder in folders:
        yyyy, mm = folder.name[:4], folder.name[4:]
        out_path = dst_dir / f"eurocontrol_{yyyy}_{mm}.parquet"
        if out_path.exists() and not overwrite:
            written.append(out_path)
            continue
        df = load_eurocontrol_flights(folder)
        df.to_parquet(out_path, index=False)
        written.append(out_path)
    return written


def load_eu_processed(
    months: Iterable[str] | None = None,
    raw_dir: Path | None = None,
) -> pd.DataFrame:
    """Read cached EUROCONTROL parquets back into a single frame.

    Parameters
    ----------
    months : iterable of "YYYY-MM" or "YYYYMM", optional
        Restrict to a subset of months. Defaults to every parquet on disk.
    raw_dir : Path, optional
        Override the default ``data/raw`` location.
    """
    raw_dir = Path(raw_dir) if raw_dir else RAW_DIR
    paths = sorted(raw_dir.glob("eurocontrol_*.parquet"))
    if not paths:
        raise FileNotFoundError(
            f"No eurocontrol_*.parquet files in {raw_dir}. "
            "Run scripts/process_eurocontrol.py first."
        )

    if months is not None:
        wanted = {m.replace("-", "") for m in months}

        def _keep(p: Path) -> bool:
            stem = p.stem  # eurocontrol_2023_03
            parts = stem.split("_")
            return f"{parts[1]}{parts[2]}" in wanted

        paths = [p for p in paths if _keep(p)]
        if not paths:
            raise FileNotFoundError(f"None of months={months} present in {raw_dir}.")

    frames = [pd.read_parquet(p) for p in paths]
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values("FL_DATE").reset_index(drop=True)
