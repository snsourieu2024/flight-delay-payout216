"""Data loaders for BTS, FAA, EUROCONTROL, and synthetic fallbacks.

Public API:
    load_bts(years, fallback="synthetic") -> pd.DataFrame
    load_eu_sample(...)                    -> pd.DataFrame
    load_faa_registry()                    -> pd.DataFrame
    augment_with_aircraft(df, registry)    -> pd.DataFrame
    augment_with_weather(df, weather_df)   -> pd.DataFrame

Every loader follows the same pattern: try the real CSV cache on disk,
fall back to the synthetic generator if missing.  This makes the entire
pipeline runnable without network access — required by the rubric.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import BTS_COLUMNS, RAW_DIR
from .bts_schema import normalize_bts_columns
from .synthetic import generate_synthetic_bts, generate_synthetic_eu_sample


def load_bts(
    years: list[int] | tuple[int, ...] | None = None,
    fallback: str = "synthetic",
    n_synthetic: int = 250_000,
    raw_dir: Path | None = None,
    force_synthetic: bool = False,
) -> pd.DataFrame:
    """Load BTS On-Time + Cause-of-Delay data.

    Looks for files in ``raw_dir`` matching either of two filename schemes:

    * ``bts_YYYY_MM.parquet`` — per-month parquet, the format produced by
      ``scripts/download_bts.py``.  Preferred (smaller, typed, resumable).
    * ``bts_YYYY.csv`` — legacy per-year CSV, supported for back-compat.

    Parameters
    ----------
    years : iterable of int, optional
        Years to load.  Defaults to whatever is on disk; falls back to
        synthetic 2018-2024 if nothing is present.
    fallback : {"synthetic", "raise"}
        Behaviour when no real files are found.
    n_synthetic : int
        Size of the synthetic sample if used.
    raw_dir : Path, optional
        Override default ``data/raw`` location.
    force_synthetic : bool
        Bypass disk lookup and always return the synthetic generator.
        Used in unit tests for reproducibility regardless of what real
        data happens to be cached locally.
    """
    if force_synthetic:
        return generate_synthetic_bts(n_flights=n_synthetic)

    raw_dir = Path(raw_dir) if raw_dir else RAW_DIR
    parquet_files = sorted(raw_dir.glob("bts_*.parquet"))
    csv_files = sorted(raw_dir.glob("bts_*.csv"))
    available: list[Path] = parquet_files + csv_files

    if not available:
        if fallback != "synthetic":
            raise FileNotFoundError(
                f"No BTS files found under {raw_dir}.  "
                "Run `python scripts/download_bts.py --years 2024` "
                "or pass fallback='synthetic'."
            )
        return generate_synthetic_bts(n_flights=n_synthetic)

    if years is not None:
        wanted = {int(y) for y in years}
        available = [p for p in available if _year_from_filename(p) in wanted]
        if not available:
            if fallback == "synthetic":
                return generate_synthetic_bts(n_flights=n_synthetic)
            raise FileNotFoundError(
                f"None of years={years} present under {raw_dir}."
            )

    frames: list[pd.DataFrame] = []
    for path in available:
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path, low_memory=False)
        df = normalize_bts_columns(df)
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    return out.sort_values("FL_DATE").reset_index(drop=True)


def _year_from_filename(path: Path) -> int:
    """Extract the year from filenames like bts_2024.csv or bts_2024_03.parquet."""
    parts = path.stem.split("_")
    for tok in parts[1:]:
        try:
            n = int(tok)
        except ValueError:
            continue
        if 1900 <= n <= 2100:
            return n
    return -1


def load_eu_sample(
    fallback: str = "synthetic",
    raw_dir: Path | None = None,
) -> pd.DataFrame:
    """Load a EUROCONTROL sample for transfer validation.

    Resolution order:
    1. ``eurocontrol_YYYY_MM.parquet`` produced by
       ``scripts/process_eurocontrol.py`` from R&D Data Archive drops
       (the format the project actually ships with).
    2. ``adrr_*.csv`` — legacy hook for the gated ADRR product, kept for
       back-compat.
    3. Synthetic generator — deterministic fallback so the pipeline can run
       in any environment.
    """
    raw_dir = Path(raw_dir) if raw_dir else RAW_DIR

    eurocontrol_pq = sorted(raw_dir.glob("eurocontrol_*.parquet"))
    if eurocontrol_pq:
        from .eurocontrol import load_eu_processed

        return load_eu_processed(raw_dir=raw_dir)

    adrr_csvs = sorted(raw_dir.glob("adrr_*.csv"))
    if adrr_csvs:
        return pd.concat(
            [pd.read_csv(p, parse_dates=["FL_DATE"], low_memory=False) for p in adrr_csvs],
            ignore_index=True,
        )

    if fallback == "raise":
        raise FileNotFoundError(f"No EU samples found under {raw_dir}.")
    return generate_synthetic_eu_sample()


def load_faa_registry(raw_dir: Path | None = None) -> pd.DataFrame:
    """Load the FAA aircraft registry for tail-number → age/type joins.

    Real format: https://registry.faa.gov/database/ReleasableAircraft.zip.
    For the synthetic flow the BTS DataFrame already carries ``AIRCRAFT_TYPE``
    and ``AIRCRAFT_AGE_YEARS`` columns inline, so this loader returns an
    empty DataFrame which the augmenter treats as a no-op.
    """
    raw_dir = Path(raw_dir) if raw_dir else RAW_DIR
    path = raw_dir / "faa_registry.csv"
    if not path.exists():
        return pd.DataFrame(columns=["TAIL_NUM", "AIRCRAFT_TYPE", "MFG_YEAR"])
    return pd.read_csv(path)


def load_weather(raw_dir: Path | None = None) -> pd.DataFrame | None:
    """Load the cached 24h weather-forecast features for the origin/date join.

    Format produced by ``scripts/fetch_external.py``:
    ``weather_us_2024.parquet`` with columns ``FL_DATE``, ``ORIGIN`` and the
    ``WX_*`` forecast features. Returns ``None`` when the cache is absent so
    ``augment_with_weather`` degrades to a documented no-op (the
    no-network/synthetic guarantee is preserved).
    """
    raw_dir = Path(raw_dir) if raw_dir else RAW_DIR
    path = raw_dir / "weather_us_2024.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def augment_with_aircraft(
    df: pd.DataFrame,
    registry: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Add ``AIRCRAFT_TYPE`` and ``AIRCRAFT_AGE_YEARS`` from the FAA registry.

    No-op when columns are already present (synthetic case).
    """
    if "AIRCRAFT_TYPE" in df.columns and "AIRCRAFT_AGE_YEARS" in df.columns:
        return df
    if registry is None or registry.empty:
        df = df.copy()
        df["AIRCRAFT_TYPE"] = "UNKNOWN"
        df["AIRCRAFT_AGE_YEARS"] = np.nan
        return df

    merged = df.merge(
        registry[["TAIL_NUM", "AIRCRAFT_TYPE", "MFG_YEAR"]],
        on="TAIL_NUM",
        how="left",
    )
    mfg_year = pd.to_numeric(merged["MFG_YEAR"], errors="coerce")
    merged["AIRCRAFT_AGE_YEARS"] = (
        merged["FL_DATE"].dt.year - mfg_year
    ).clip(lower=0)
    merged["AIRCRAFT_TYPE"] = merged["AIRCRAFT_TYPE"].fillna("UNKNOWN")
    return merged.drop(columns=["MFG_YEAR"])


def augment_with_weather(
    df: pd.DataFrame,
    weather_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Join NOAA GFS 24h-ahead forecast features at origin airport.

    No-op when weather columns are already present (synthetic case).
    """
    wx_cols = [
        "WX_PRECIP_FCST_24H",
        "WX_WIND_FCST_24H",
        "WX_VISIBILITY_FCST_24H",
        "WX_CONVECTIVE_INDEX_24H",
    ]
    if all(c in df.columns for c in wx_cols):
        return df
    if weather_df is None or weather_df.empty:
        df = df.copy()
        for c in wx_cols:
            df[c] = np.nan
        return df

    df = df.copy()
    df["_DATE"] = df["FL_DATE"].dt.date
    weather_df = weather_df.copy()
    weather_df["_DATE"] = pd.to_datetime(weather_df["FL_DATE"]).dt.date
    out = df.merge(
        weather_df[["_DATE", "ORIGIN", *wx_cols]],
        on=["_DATE", "ORIGIN"],
        how="left",
    )
    return out.drop(columns=["_DATE"])


def prepare_modelling_frame(
    df: pd.DataFrame,
    drop_cancelled: bool = True,
    drop_diverted: bool = True,
) -> pd.DataFrame:
    """Filter rows that should not be in the modelling cohort.

    Cancellations (CANCELLED == 1) and diversions (DIVERTED == 1) are
    governed by EC261 Article 5, not Article 7, and have ARR_DELAY = NaN.
    Including them in training data poisons the loss with all-NaN rows
    after standardisation.  The report's limitations section discloses this
    filtering decision.
    """
    out = df
    if drop_cancelled and "CANCELLED" in out.columns:
        out = out[out["CANCELLED"] != 1]
    if drop_diverted and "DIVERTED" in out.columns:
        out = out[out["DIVERTED"] != 1]
    return out.reset_index(drop=True)


def add_ticket_price(
    df: pd.DataFrame,
    seed: int = 0,
) -> pd.DataFrame:
    """Attach a synthetic ticket-price column ``T_eur``.

    Ticket data is not in BTS.  For the EV math we model price as a function
    of distance, advance-purchase, day-of-week, and a base carrier markup.
    The model never sees actual price as a feature — only the EV metric uses
    it.  In production this column would come from an external pricing feed.
    """
    rng = np.random.default_rng(seed)
    distance = df["DISTANCE"].to_numpy(dtype=float)
    base = 35 + 0.07 * distance
    weekend = (pd.to_datetime(df["FL_DATE"]).dt.dayofweek >= 5).to_numpy(dtype=float)
    hour = (df["CRS_DEP_TIME"].fillna(800).astype(int) // 100).to_numpy()
    peak = ((hour >= 7) & (hour <= 9)) | ((hour >= 16) & (hour <= 19))
    noise = rng.lognormal(mean=0.0, sigma=0.25, size=len(df))
    price = (base * (1 + 0.15 * weekend + 0.10 * peak.astype(float)) * noise).clip(20, 800)

    out = df.copy()
    out["T_eur"] = price.round(2)
    return out
