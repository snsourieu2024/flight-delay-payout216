"""Fetch the two missing high-signal data sources (Step 2).

1. FAA Releasable Aircraft registry  -> data/raw/faa_registry.csv
     (the project's download_faa.py uses a default UA that Akamai 403s;
      here we send a browser UA.)
2. Open-Meteo 2024 daily weather for the busiest US airports
     -> data/raw/weather_us_2024.parquet  (FL_DATE, ORIGIN, WX_* columns
        the pipeline already expects).

Network-robust: browser UA, timeouts, airport cap, on any failure the file is
simply not written and the rest of the pipeline falls back to NaN (imputed).
Idempotent: skips a source whose output already exists.
"""
from __future__ import annotations

import io
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.config import RAW_DIR  # noqa: E402
from src.data.loaders import load_bts  # noqa: E402

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                     "AppleWebKit/537.36 (KHTML, like Gecko) "
                     "Chrome/124.0 Safari/537.36"}
TOP_AIRPORTS = 200          # covers ~99% of US flights; bounds API calls
FAA_URL = "https://registry.faa.gov/database/ReleasableAircraft.zip"
AIRPORTS_URL = "https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat"
METEO = "https://archive-api.open-meteo.com/v1/archive"


def fetch_faa() -> None:
    out = RAW_DIR / "faa_registry.csv"
    if out.exists():
        print(f"[FAA] {out.name} exists ({out.stat().st_size:,} B) - skip")
        return
    print("[FAA] downloading ReleasableAircraft.zip (browser UA) ...", flush=True)
    r = requests.get(FAA_URL, headers=UA, timeout=120)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        inner = next(n for n in zf.namelist() if n.upper().endswith("MASTER.TXT"))
        with zf.open(inner) as fh:
            df = pd.read_csv(fh, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"N-NUMBER": "TAIL_NUM_RAW",
                            "MFR MDL CODE": "MFR_MDL", "YEAR MFR": "MFG_YEAR"})
    df["TAIL_NUM"] = "N" + df["TAIL_NUM_RAW"].astype(str).str.strip()
    df["AIRCRAFT_TYPE"] = df["MFR_MDL"].astype(str).str.strip()
    df = df[["TAIL_NUM", "AIRCRAFT_TYPE", "MFG_YEAR"]]
    df.to_csv(out, index=False)
    print(f"[FAA] wrote {out.name}  ({len(df):,} rows)", flush=True)


def _airport_coords() -> pd.DataFrame:
    r = requests.get(AIRPORTS_URL, headers=UA, timeout=60)
    r.raise_for_status()
    cols = ["id", "name", "city", "country", "IATA", "ICAO", "lat", "lon",
            "alt", "tz", "dst", "tzdb", "type", "src"]
    ap = pd.read_csv(io.StringIO(r.text), header=None, names=cols)
    ap = ap[(ap["country"] == "United States") & ap["IATA"].notna()
            & (ap["IATA"] != "\\N")]
    return ap[["IATA", "lat", "lon"]].drop_duplicates("IATA")


def fetch_weather() -> None:
    out = RAW_DIR / "weather_us_2024.parquet"
    if out.exists():
        print(f"[WX] {out.name} exists - skip")
        return
    print("[WX] ranking airports by 2024 BTS volume ...", flush=True)
    bts = load_bts(fallback="synthetic")
    span_lo = str(pd.to_datetime(bts["FL_DATE"]).min().date())
    span_hi = str(pd.to_datetime(bts["FL_DATE"]).max().date())
    top = (bts["ORIGIN"].value_counts().head(TOP_AIRPORTS).index.tolist())
    coords = _airport_coords().set_index("IATA")
    have = [a for a in top if a in coords.index]
    print(f"[WX] {len(have)}/{len(top)} airports geocoded; "
          f"fetching {span_lo}..{span_hi} daily precip/wind ...", flush=True)

    frames = []
    for i, ap in enumerate(have):
        la, lo = float(coords.loc[ap, "lat"]), float(coords.loc[ap, "lon"])
        try:
            j = requests.get(METEO, timeout=60, headers=UA, params={
                "latitude": la, "longitude": lo,
                "start_date": span_lo, "end_date": span_hi,
                "daily": "precipitation_sum,wind_speed_10m_max,"
                         "precipitation_hours",
                "timezone": "UTC"}).json()["daily"]
        except Exception as e:                       # noqa: BLE001
            print(f"[WX]  {ap}: {type(e).__name__} - skipped", flush=True)
            continue
        d = pd.DataFrame(j)
        d["ORIGIN"] = ap
        frames.append(d)
        if (i + 1) % 25 == 0:
            print(f"[WX]  {i+1}/{len(have)} airports", flush=True)
        time.sleep(0.15)                              # be polite to the API

    if not frames:
        print("[WX] no weather fetched; pipeline will NaN-impute.", flush=True)
        return
    wx = pd.concat(frames, ignore_index=True)
    wx["FL_DATE"] = pd.to_datetime(wx["time"])
    wx = wx.rename(columns={
        "precipitation_sum": "WX_PRECIP_FCST_24H",
        "wind_speed_10m_max": "WX_WIND_FCST_24H",
        "precipitation_hours": "WX_VISIBILITY_FCST_24H",   # proxy: more precip-hrs => worse vis
    })
    wx["WX_VISIBILITY_FCST_24H"] = -wx["WX_VISIBILITY_FCST_24H"]  # sign so higher = better
    wx["WX_CONVECTIVE_INDEX_24H"] = wx["WX_PRECIP_FCST_24H"].clip(lower=0)
    wx = wx[["FL_DATE", "ORIGIN", "WX_PRECIP_FCST_24H", "WX_WIND_FCST_24H",
             "WX_VISIBILITY_FCST_24H", "WX_CONVECTIVE_INDEX_24H"]]
    wx.to_parquet(out, index=False)
    print(f"[WX] wrote {out.name}  ({len(wx):,} airport-days, "
          f"{wx['ORIGIN'].nunique()} airports)", flush=True)


if __name__ == "__main__":
    try:
        fetch_faa()
    except Exception as e:                            # noqa: BLE001
        print(f"[FAA] FAILED ({type(e).__name__}: {e}); "
              f"aircraft age stays NaN.", file=sys.stderr, flush=True)
    try:
        fetch_weather()
    except Exception as e:                            # noqa: BLE001
        print(f"[WX] FAILED ({type(e).__name__}: {e}); "
              f"weather stays NaN.", file=sys.stderr, flush=True)
