"""Project-wide constants and paths.

Single source of truth for directory layout, schema columns, EC261 parameters,
and seeds. Importing from here (instead of hard-coding) keeps the code
reproducible and makes hyperparameter sensitivity analysis trivial.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLE_DIR = DATA_DIR / "sample"
CACHE_DIR = DATA_DIR / "cache"
ARTEFACTS_DIR = ROOT / "artefacts"
REPORTS_DIR = ROOT / "reports"

for _d in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR, SAMPLE_DIR, CACHE_DIR, ARTEFACTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42

# Year-based temporal split.  See plan section 3 (validation strategy).
TRAIN_YEARS = (2018, 2019, 2020, 2021, 2022)
VAL_YEARS = (2023,)
TEST_YEARS = (2024,)

# Booking horizon (days before scheduled departure).  Features used for prediction
# must be available at this horizon — anything that materialises later is leakage.
BOOKING_HORIZON_DAYS = 14


@dataclass(frozen=True)
class EC261Params:
    """Parameters for the EC261 compensation regime.

    Defaults model the post-September 2025 amended regulation.  Sensitivity
    analysis in notebook 04 sweeps `claim_success_rate`, `claim_cost_eur`,
    and `travel_cost_eur` independently.
    """

    delay_threshold_min: int = 180
    short_haul_km: int = 1500
    medium_haul_km: int = 3500
    payout_short_eur: float = 250.0
    payout_medium_eur: float = 400.0
    payout_long_eur: float = 600.0
    claim_success_rate: float = 0.65
    claim_cost_eur: float = 15.0
    travel_cost_eur: float = 50.0

    eligible_causes: tuple[str, ...] = field(
        default=("CARRIER_DELAY", "LATE_AIRCRAFT_DELAY")
    )
    exempt_causes: tuple[str, ...] = field(
        default=("WEATHER_DELAY", "NAS_DELAY", "SECURITY_DELAY")
    )


EC261 = EC261Params()


# BTS schema — column names we expect from the Reporting Carrier On-Time + Cause
# of Delay tables.  Used by loaders and the synthetic generator.
BTS_COLUMNS = [
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
]

CATEGORICAL_FEATURES = [
    "OP_UNIQUE_CARRIER",
    "ORIGIN",
    "DEST",
    "AIRCRAFT_TYPE",
]
NUMERIC_FEATURES = [
    "DISTANCE",
    "CRS_ELAPSED_TIME",
    "AIRCRAFT_AGE_YEARS",
    "ROUTE_DELAY_RATE_30D",
    "ROUTE_DELAY_RATE_90D",
    "CARRIER_DELAY_RATE_30D",
    "ORIGIN_DELAY_RATE_30D",
    "TAIL_DELAY_RATE_90D",
    "WX_PRECIP_FCST_24H",
    "WX_WIND_FCST_24H",
    "WX_VISIBILITY_FCST_24H",
    "WX_CONVECTIVE_INDEX_24H",
]
CYCLICAL_FEATURES = ["HOUR", "DAYOFWEEK", "MONTH"]
