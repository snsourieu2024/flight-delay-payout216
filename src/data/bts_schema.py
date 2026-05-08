"""BTS PREZIP CSV → internal schema mapping.

The BTS "Reporting Carrier On-Time Performance" PREZIP files distribute their
columns under camelCase / mixed-style names (``FlightDate``, ``Reporting_Airline``,
``CarrierDelay``, ...).  The rest of the project speaks the SQL-style schema
declared in ``config.BTS_COLUMNS`` (``FL_DATE``, ``OP_UNIQUE_CARRIER``,
``CARRIER_DELAY``, ...).

This module owns the mapping in one place so the rest of the loaders, tests,
and notebooks never have to know which dialect they are reading.
"""
from __future__ import annotations

import pandas as pd

from ..config import BTS_COLUMNS


BTS_PREZIP_RENAME: dict[str, str] = {
    "FlightDate": "FL_DATE",
    "Reporting_Airline": "OP_UNIQUE_CARRIER",
    "Tail_Number": "TAIL_NUM",
    "Flight_Number_Reporting_Airline": "OP_CARRIER_FL_NUM",
    "Origin": "ORIGIN",
    "Dest": "DEST",
    "CRSDepTime": "CRS_DEP_TIME",
    "DepTime": "DEP_TIME",
    "DepDelay": "DEP_DELAY",
    "CRSArrTime": "CRS_ARR_TIME",
    "ArrTime": "ARR_TIME",
    "ArrDelay": "ARR_DELAY",
    "Cancelled": "CANCELLED",
    "Diverted": "DIVERTED",
    "CRSElapsedTime": "CRS_ELAPSED_TIME",
    "Distance": "DISTANCE",
    "CarrierDelay": "CARRIER_DELAY",
    "WeatherDelay": "WEATHER_DELAY",
    "NASDelay": "NAS_DELAY",
    "SecurityDelay": "SECURITY_DELAY",
    "LateAircraftDelay": "LATE_AIRCRAFT_DELAY",
}


def normalize_bts_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename BTS PREZIP columns to internal schema and keep only what we need.

    Columns already in internal-schema form pass through untouched, so this is
    safe to call on either format (idempotent).  Missing columns raise so the
    caller learns immediately rather than silently shipping NaNs into training.
    """
    rename = {k: v for k, v in BTS_PREZIP_RENAME.items() if k in df.columns}
    out = df.rename(columns=rename)

    missing = [c for c in BTS_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(
            "After renaming, the following BTS columns are missing: "
            f"{missing}.  Columns present: {sorted(out.columns)[:30]}..."
        )

    out = out[BTS_COLUMNS].copy()
    out["FL_DATE"] = pd.to_datetime(out["FL_DATE"], errors="coerce")

    int_cols = [
        "OP_CARRIER_FL_NUM",
        "CRS_DEP_TIME",
        "CRS_ARR_TIME",
        "CANCELLED",
        "DIVERTED",
        "CRS_ELAPSED_TIME",
        "DISTANCE",
    ]
    for c in int_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("Int64")

    float_cols = [
        "DEP_TIME",
        "DEP_DELAY",
        "ARR_TIME",
        "ARR_DELAY",
        "CARRIER_DELAY",
        "WEATHER_DELAY",
        "NAS_DELAY",
        "SECURITY_DELAY",
        "LATE_AIRCRAFT_DELAY",
    ]
    for c in float_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    str_cols = ["OP_UNIQUE_CARRIER", "TAIL_NUM", "ORIGIN", "DEST"]
    for c in str_cols:
        out[c] = out[c].astype("string").str.strip()

    return out
