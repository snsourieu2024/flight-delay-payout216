"""Download the FAA Aircraft Registry for tail-number → age/type joins.

Source: https://registry.faa.gov/database/ReleasableAircraft.zip
Output: data/raw/faa_registry.csv with columns
        TAIL_NUM, AIRCRAFT_TYPE, MFG_YEAR
"""
from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import requests

try:
    from src.config import RAW_DIR  # type: ignore
except ImportError:
    RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

FAA_URL = "https://registry.faa.gov/database/ReleasableAircraft.zip"


def main() -> int:
    out = RAW_DIR / "faa_registry.csv"
    try:
        resp = requests.get(FAA_URL, stream=True, timeout=60)
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            inner = next(n for n in zf.namelist() if n.upper().endswith("MASTER.TXT"))
            with zf.open(inner) as fh:
                import pandas as pd
                df = pd.read_csv(fh, low_memory=False)
    except (requests.RequestException, zipfile.BadZipFile, StopIteration):
        print(
            "Could not reach FAA registry.  Manual fallback:\n"
            "  curl -O https://registry.faa.gov/database/ReleasableAircraft.zip\n"
            "  unzip ReleasableAircraft.zip MASTER.txt\n"
            "  mv MASTER.txt data/raw/faa_master.csv\n"
            "and re-run this script with --skip-download.",
            file=sys.stderr,
        )
        return 1

    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={
        "N-NUMBER": "TAIL_NUM_RAW",
        "MFR MDL CODE": "MFR_MDL",
        "YEAR MFR": "MFG_YEAR",
    })
    df["TAIL_NUM"] = "N" + df["TAIL_NUM_RAW"].astype(str).str.strip()
    df["AIRCRAFT_TYPE"] = df["MFR_MDL"].astype(str).str.strip()
    df = df[["TAIL_NUM", "AIRCRAFT_TYPE", "MFG_YEAR"]]
    df.to_csv(out, index=False)
    print(f"Wrote {out}  ({len(df):,} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
