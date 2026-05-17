"""Rebuild the documented modelling sample from the full frame.

Thin CLI wrapper around ``src.data.sampling.stratified_modelling_sample`` —
the same function notebook 00 uses, so the sample is identical however it is
produced. Preserves the full frame as ``flights.full2024.parquet`` for
provenance (nothing is destroyed). Re-runnable and idempotent.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parents[1]))

from src.config import PROCESSED_DIR
from src.data.sampling import DEFAULT_SAMPLE_N, LABEL_COL, stratified_modelling_sample

FULL = PROCESSED_DIR / "flights.full2024.parquet"
SAMPLE = PROCESSED_DIR / "flights.parquet"


def main() -> None:
    src = FULL if FULL.exists() else SAMPLE
    if not src.exists():
        raise FileNotFoundError(f"Neither {FULL} nor {SAMPLE} exists. Run notebook 00 first.")

    df = pd.read_parquet(src)
    print(f"[sample] source={src.name} rows={len(df):,}")
    if len(df) <= DEFAULT_SAMPLE_N:
        print(f"[sample] already <= {DEFAULT_SAMPLE_N:,}; nothing to do.")
        return

    if not FULL.exists():
        df.to_parquet(FULL, index=False)
        print(f"[sample] backed up full frame -> {FULL.name} ({len(df):,} rows)")

    sample = stratified_modelling_sample(df)
    sample.to_parquet(SAMPLE, index=False)

    def _rate(frame: pd.DataFrame) -> float:
        return float(frame[LABEL_COL].mean()) if LABEL_COL in frame.columns else float("nan")

    months = sorted(pd.to_datetime(sample["FL_DATE"]).dt.to_period("M").astype(str).unique())
    print(f"[sample] wrote {SAMPLE.name}: {len(sample):,} rows")
    print(f"[sample] base rate sample={_rate(sample):.4%} full={_rate(df):.4%}")
    print(f"[sample] months ({len(months)}): {months[0]}..{months[-1]}")


if __name__ == "__main__":
    main()
