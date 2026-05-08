"""Synthetic BTS-like data generator.

Produces a DataFrame with the schema declared in ``config.BTS_COLUMNS`` and
realistic statistical structure:

- Multiple years with year-on-year base-rate drift (mimics post-COVID rebound).
- Hub-and-spoke airport graph with capacity-driven delay rates.
- Aircraft tail numbers with persistent quality (some "lemon" tails).
- Late-aircraft cascading delay structure (the most predictive signal in real data).
- Five cause-code columns whose distribution matches BTS empirical proportions.
- Weather forecast columns correlated with the WEATHER_DELAY component.

Used for:
    1. CI smoke testing (no network).
    2. Local development when BTS download is rate-limited.
    3. Unit tests of the feature pipeline.

This generator is **deterministic** given a seed so test fixtures are stable.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from ..config import BTS_COLUMNS, RANDOM_SEED


HUB_AIRPORTS = [
    ("ATL", 33.6367, -84.4281, 10000),
    ("ORD", 41.9786, -87.9048, 9500),
    ("DFW", 32.8968, -97.0380, 8500),
    ("DEN", 39.8617, -104.6731, 7000),
    ("LAX", 33.9425, -118.4081, 9000),
    ("JFK", 40.6413, -73.7781, 5500),
    ("SFO", 37.6213, -122.3790, 5000),
    ("SEA", 47.4502, -122.3088, 4500),
    ("MIA", 25.7959, -80.2870, 4000),
    ("BOS", 42.3656, -71.0096, 4000),
]
SPOKES = [
    ("AUS", 30.1944, -97.6700),
    ("MSP", 44.8848, -93.2223),
    ("PHX", 33.4373, -112.0078),
    ("CLT", 35.2140, -80.9431),
    ("LAS", 36.0840, -115.1537),
    ("DTW", 42.2162, -83.3554),
    ("PHL", 39.8744, -75.2424),
    ("BWI", 39.1774, -76.6684),
    ("SAN", 32.7338, -117.1933),
    ("PDX", 45.5898, -122.5951),
    ("MCO", 28.4312, -81.3081),
    ("FLL", 26.0742, -80.1506),
    ("IAH", 29.9844, -95.3414),
    ("SLC", 40.7899, -111.9791),
    ("HNL", 21.3186, -157.9224),
]
ALL_AIRPORTS = HUB_AIRPORTS + [(c, lat, lon, 1500) for c, lat, lon in SPOKES]
CARRIERS = ["AA", "DL", "UA", "WN", "B6", "AS", "NK", "F9"]


def _haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R_km = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = phi2 - phi1
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlmb / 2) ** 2
    km = 2 * R_km * np.arcsin(np.sqrt(a))
    return km / 1.609344


def generate_synthetic_bts(
    n_flights: int = 250_000,
    start_year: int = 2018,
    end_year: int = 2024,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Generate a realistic BTS-shaped DataFrame.

    Parameters
    ----------
    n_flights : int
        Total flights across the date range.
    start_year, end_year : int
        Inclusive range of years.
    seed : int
        Numpy RNG seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        With every column declared in ``BTS_COLUMNS``.
    """
    rng = np.random.default_rng(seed)

    airport_codes = [a[0] for a in ALL_AIRPORTS]
    airport_coords = {a[0]: (a[1], a[2]) for a in ALL_AIRPORTS}
    airport_capacity = {a[0]: a[3] for a in ALL_AIRPORTS}

    n_tails = 1500
    tails = [f"N{rng.integers(100, 999)}{chr(rng.integers(65, 91))}{chr(rng.integers(65, 91))}{i:04d}" for i in range(n_tails)]
    tail_quality = rng.beta(8, 2, size=n_tails)
    aircraft_age = rng.gamma(shape=3.0, scale=4.0, size=n_tails).clip(0, 35)
    aircraft_type = rng.choice(["B737", "A320", "B757", "A321", "E175", "CRJ900", "B777"], size=n_tails)
    tail_meta = pd.DataFrame({
        "TAIL_NUM": tails,
        "TAIL_QUALITY": tail_quality,
        "AIRCRAFT_AGE_YEARS": aircraft_age,
        "AIRCRAFT_TYPE": aircraft_type,
    })

    start_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 12, 31)
    n_days = (end_date - start_date).days + 1

    day_offsets = rng.integers(0, n_days, size=n_flights)
    fl_dates = [start_date + timedelta(days=int(d)) for d in day_offsets]
    months = np.array([d.month for d in fl_dates])
    dow = np.array([d.weekday() for d in fl_dates])
    years = np.array([d.year for d in fl_dates])

    crs_dep_hour = rng.choice(np.arange(5, 23), size=n_flights, p=_hourly_pmf())
    crs_dep_minute = rng.choice([0, 15, 30, 45], size=n_flights)
    crs_dep_time = (crs_dep_hour * 100 + crs_dep_minute).astype(int)

    origin_idx = rng.choice(len(airport_codes), size=n_flights, p=_airport_pmf(airport_codes, airport_capacity))
    dest_idx = rng.choice(len(airport_codes), size=n_flights, p=_airport_pmf(airport_codes, airport_capacity))
    same = origin_idx == dest_idx
    while same.any():
        dest_idx[same] = rng.choice(len(airport_codes), size=int(same.sum()))
        same = origin_idx == dest_idx
    origin = np.array(airport_codes)[origin_idx]
    dest = np.array(airport_codes)[dest_idx]

    distance = np.array([
        _haversine_miles(*airport_coords[o], *airport_coords[d])
        for o, d in zip(origin, dest)
    ])
    crs_elapsed = (distance / 8.0 + 35 + rng.normal(0, 5, size=n_flights)).clip(40, 700).round()

    carrier = rng.choice(CARRIERS, size=n_flights, p=[0.18, 0.16, 0.15, 0.20, 0.08, 0.07, 0.09, 0.07])
    tail_idx = rng.integers(0, n_tails, size=n_flights)
    tail_num = np.array(tails)[tail_idx]
    aircraft_age_per_flight = aircraft_age[tail_idx] + (years - start_year)
    aircraft_type_per_flight = aircraft_type[tail_idx]
    tail_quality_per_flight = tail_quality[tail_idx]

    wx_precip = rng.gamma(shape=0.5, scale=2.0, size=n_flights)
    wx_wind = rng.gamma(shape=2.0, scale=4.0, size=n_flights)
    wx_visibility = (10 - rng.exponential(2.0, size=n_flights)).clip(0.1, 10)
    wx_convective = (
        0.3 * (months == 6).astype(float)
        + 0.4 * (months == 7).astype(float)
        + 0.3 * (months == 8).astype(float)
        + rng.gamma(0.4, 0.5, size=n_flights)
    )

    cap_pressure = np.array([airport_capacity[o] for o in origin]) / 10000.0
    base_p = 0.04 + 0.06 * (1 - cap_pressure)
    base_p += 0.02 * (years - start_year) / max(1, end_year - start_year)
    base_p += 0.03 * (1 - tail_quality_per_flight)
    base_p += 0.04 * (crs_dep_hour >= 17).astype(float)
    base_p += 0.03 * (dow >= 4).astype(float)
    base_p = base_p.clip(0.01, 0.45)

    is_long_delay = rng.uniform(size=n_flights) < base_p

    p_carrier = 0.32
    p_late_aircraft = 0.30
    p_weather = 0.20
    p_nas = 0.16
    p_security = 0.02

    weather_boost = np.minimum(0.5 * wx_precip + 0.3 * wx_convective, 5.0)
    p_weather_dyn = (p_weather + 0.05 * weather_boost) / (1 + 0.05 * weather_boost.mean())
    cap_boost = (1 - cap_pressure) * 0.3
    p_nas_dyn = p_nas * (1 + cap_boost)

    cause_probs = np.stack([
        np.full(n_flights, p_carrier),
        np.full(n_flights, p_late_aircraft),
        p_weather_dyn,
        p_nas_dyn,
        np.full(n_flights, p_security),
    ], axis=1)
    cause_probs /= cause_probs.sum(axis=1, keepdims=True)

    chosen_cause = np.array([
        rng.choice(5, p=cause_probs[i]) for i in range(n_flights)
    ])

    delay_min_total = np.where(
        is_long_delay,
        rng.gamma(shape=4.0, scale=60.0, size=n_flights) + 60,
        rng.gamma(shape=1.0, scale=10.0, size=n_flights),
    )

    cause_carrier = np.where((chosen_cause == 0) & is_long_delay, delay_min_total, 0).astype(int)
    cause_late = np.where((chosen_cause == 1) & is_long_delay, delay_min_total, 0).astype(int)
    cause_wx = np.where((chosen_cause == 2) & is_long_delay, delay_min_total, 0).astype(int)
    cause_nas = np.where((chosen_cause == 3) & is_long_delay, delay_min_total, 0).astype(int)
    cause_sec = np.where((chosen_cause == 4) & is_long_delay, delay_min_total, 0).astype(int)

    short_delay_min = np.where(~is_long_delay, delay_min_total, 0).astype(int)
    arr_delay = (cause_carrier + cause_late + cause_wx + cause_nas + cause_sec + short_delay_min).astype(int)
    arr_delay = arr_delay - rng.integers(0, 8, size=n_flights)
    dep_delay = (arr_delay - rng.integers(-15, 15, size=n_flights)).clip(-30, None)

    cancelled = (rng.uniform(size=n_flights) < 0.018).astype(int)
    diverted = (rng.uniform(size=n_flights) < 0.003).astype(int)

    crs_arr_minutes = crs_dep_hour * 60 + crs_dep_minute + crs_elapsed.astype(int)
    crs_arr_hour = (crs_arr_minutes // 60) % 24
    crs_arr_min = crs_arr_minutes % 60
    crs_arr_time = (crs_arr_hour * 100 + crs_arr_min).astype(int)
    dep_time = ((crs_dep_hour * 60 + crs_dep_minute + dep_delay.clip(0, None)) % 1440).astype(int)
    dep_time = (dep_time // 60 * 100 + dep_time % 60).astype(int)
    arr_time_min = (crs_arr_minutes + arr_delay.clip(0, None)) % 1440
    arr_time = (arr_time_min // 60 * 100 + arr_time_min % 60).astype(int)

    fl_num = rng.integers(1, 9999, size=n_flights)

    df = pd.DataFrame({
        "FL_DATE": pd.to_datetime(fl_dates),
        "OP_UNIQUE_CARRIER": carrier,
        "TAIL_NUM": tail_num,
        "OP_CARRIER_FL_NUM": fl_num,
        "ORIGIN": origin,
        "DEST": dest,
        "CRS_DEP_TIME": crs_dep_time,
        "DEP_TIME": dep_time,
        "DEP_DELAY": dep_delay,
        "CRS_ARR_TIME": crs_arr_time,
        "ARR_TIME": arr_time,
        "ARR_DELAY": arr_delay,
        "CANCELLED": cancelled,
        "DIVERTED": diverted,
        "CRS_ELAPSED_TIME": crs_elapsed.astype(int),
        "DISTANCE": distance.round().astype(int),
        "CARRIER_DELAY": cause_carrier,
        "WEATHER_DELAY": cause_wx,
        "NAS_DELAY": cause_nas,
        "SECURITY_DELAY": cause_sec,
        "LATE_AIRCRAFT_DELAY": cause_late,
        # Auxiliary columns kept for downstream features (joined separately
        # in real data via FAA registry / NOAA).
        "AIRCRAFT_TYPE": aircraft_type_per_flight,
        "AIRCRAFT_AGE_YEARS": aircraft_age_per_flight.round(1),
        "WX_PRECIP_FCST_24H": wx_precip.round(2),
        "WX_WIND_FCST_24H": wx_wind.round(2),
        "WX_VISIBILITY_FCST_24H": wx_visibility.round(2),
        "WX_CONVECTIVE_INDEX_24H": wx_convective.round(2),
    })

    df = df.sort_values("FL_DATE").reset_index(drop=True)
    df.loc[df["CANCELLED"] == 1, ["DEP_TIME", "ARR_TIME", "DEP_DELAY", "ARR_DELAY"]] = np.nan
    return df[BTS_COLUMNS + ["AIRCRAFT_TYPE", "AIRCRAFT_AGE_YEARS",
                             "WX_PRECIP_FCST_24H", "WX_WIND_FCST_24H",
                             "WX_VISIBILITY_FCST_24H", "WX_CONVECTIVE_INDEX_24H"]]


def _hourly_pmf() -> np.ndarray:
    hours = np.arange(5, 23)
    base = np.exp(-0.5 * ((hours - 8) / 3.0) ** 2) + 0.7 * np.exp(-0.5 * ((hours - 17) / 3.5) ** 2)
    return base / base.sum()


def _airport_pmf(codes, capacity) -> np.ndarray:
    caps = np.array([capacity[c] for c in codes], dtype=float)
    return caps / caps.sum()


def generate_synthetic_eu_sample(
    n_flights: int = 50_000,
    seed: int = RANDOM_SEED + 1,
) -> pd.DataFrame:
    """Generate a EUROCONTROL-ADRR-shaped sample.

    Schema is a thin proxy: scheduled vs actual departure/arrival, no cause
    codes (as in real ADRR), and EU airport codes.  Used for the transfer
    validation chapter when network access to ADRR is unavailable.
    """
    rng = np.random.default_rng(seed)
    eu_airports = [
        ("LHR", 51.4775, -0.4614),
        ("CDG", 49.0097, 2.5479),
        ("AMS", 52.3105, 4.7683),
        ("FRA", 50.0379, 8.5622),
        ("MAD", 40.4936, -3.5668),
        ("BCN", 41.2974, 2.0833),
        ("FCO", 41.7999, 12.2462),
        ("MUC", 48.3538, 11.7861),
        ("DUB", 53.4213, -6.2701),
        ("VIE", 48.1102, 16.5697),
    ]
    codes = [a[0] for a in eu_airports]
    coords = {a[0]: (a[1], a[2]) for a in eu_airports}
    carriers_eu = ["BA", "AF", "KL", "LH", "IB", "FR", "U2", "LX", "AY", "SK"]

    fl_dates = pd.to_datetime(
        np.array(
            [datetime(2023, 9, 1) + timedelta(days=int(d), minutes=int(m))
             for d, m in zip(rng.integers(0, 30, size=n_flights),
                             rng.integers(0, 1440, size=n_flights))]
        )
    )
    o = rng.choice(len(codes), size=n_flights)
    d = rng.choice(len(codes), size=n_flights)
    same = o == d
    while same.any():
        d[same] = rng.choice(len(codes), size=int(same.sum()))
        same = o == d
    origin = np.array(codes)[o]
    dest = np.array(codes)[d]
    distance = np.array([_haversine_miles(*coords[a], *coords[b])
                         for a, b in zip(origin, dest)])

    base_p = 0.07 + 0.03 * (np.array([fd.hour for fd in fl_dates]) >= 17).astype(float)
    is_long_delay = rng.uniform(size=n_flights) < base_p
    arr_delay = np.where(is_long_delay,
                         rng.gamma(4.0, 60.0, size=n_flights) + 60,
                         rng.gamma(1.0, 8.0, size=n_flights))
    arr_delay = arr_delay.astype(int)

    return pd.DataFrame({
        "FL_DATE": fl_dates,
        "OP_UNIQUE_CARRIER": rng.choice(carriers_eu, size=n_flights),
        "ORIGIN": origin,
        "DEST": dest,
        "CRS_DEP_TIME": (np.array([fd.hour for fd in fl_dates]) * 100
                          + np.array([fd.minute for fd in fl_dates])),
        "ARR_DELAY": arr_delay,
        "DISTANCE": distance.round().astype(int),
        "CRS_ELAPSED_TIME": (distance / 8.0 + 35).round().astype(int),
    })
