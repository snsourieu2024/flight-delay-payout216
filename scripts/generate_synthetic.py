"""Materialise a synthetic BTS file under data/raw/.

Usage:
    python scripts/generate_synthetic.py --n 250000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parents[1]))

from src.config import RAW_DIR
from src.data.synthetic import generate_synthetic_bts


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=250_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=RAW_DIR / "bts_synthetic.csv")
    args = p.parse_args()
    df = generate_synthetic_bts(n_flights=args.n, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {args.out}  ({len(df):,} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
