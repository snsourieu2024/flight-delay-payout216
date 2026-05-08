#!/usr/bin/env python3
"""Convert EUROCONTROL R&D Archive monthly drops to parquet.

Each YYYYMM directory under the project root (or under ``--src``) is
expected to contain one ``Flights_YYYYMMDD_YYYYMMDD.csv.gz`` per the R&D
Archive schema. We parse it, normalise to BTS column conventions, and
write ``data/raw/eurocontrol_YYYY_MM.parquet`` so subsequent loads are
~30x faster.

Usage:
    python scripts/process_eurocontrol.py
    python scripts/process_eurocontrol.py --src /path/to/drops --overwrite
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import RAW_DIR  # noqa: E402
from src.data.eurocontrol import (  # noqa: E402
    find_flights_file,
    load_eurocontrol_flights,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src",
        type=Path,
        default=ROOT,
        help="Directory containing YYYYMM/ subfolders (default: repo root).",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=RAW_DIR,
        help="Where to write parquet caches (default: data/raw).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-process months whose parquet already exists.",
    )
    args = parser.parse_args()

    src: Path = args.src.resolve()
    dst: Path = args.dst.resolve()
    dst.mkdir(parents=True, exist_ok=True)

    folders = sorted(
        p for p in src.iterdir() if p.is_dir() and p.name.isdigit() and len(p.name) == 6
    )
    if not folders:
        print(f"No YYYYMM/ folders under {src}.", file=sys.stderr)
        return 1

    print(f"Found {len(folders)} EUROCONTROL drops under {src}:")
    for f in folders:
        print(f"  - {f.name}")

    total_rows = 0
    total_seconds = 0.0
    for folder in folders:
        yyyy, mm = folder.name[:4], folder.name[4:]
        out_path = dst / f"eurocontrol_{yyyy}_{mm}.parquet"
        if out_path.exists() and not args.overwrite:
            n = _row_count(out_path)
            total_rows += n
            print(f"  skip {folder.name}: {out_path.name} already exists ({n:,} rows)")
            continue

        flights_path = find_flights_file(folder)
        size_mb = flights_path.stat().st_size / (1024 * 1024)
        t0 = time.perf_counter()
        df = load_eurocontrol_flights(folder)
        t1 = time.perf_counter()
        df.to_parquet(out_path, index=False)
        t2 = time.perf_counter()
        total_rows += len(df)
        total_seconds += t2 - t0
        out_size_mb = out_path.stat().st_size / (1024 * 1024)
        print(
            f"  {folder.name}: {len(df):>9,} rows | "
            f"{size_mb:>5.1f} MB csv.gz -> {out_size_mb:>5.1f} MB parquet | "
            f"parse {t1 - t0:>4.1f}s, write {t2 - t1:>4.1f}s"
        )

    print()
    print(f"Total rows across cache: {total_rows:,}")
    if total_seconds > 0:
        print(f"Effective throughput on new files: {total_rows / total_seconds / 1e6:.2f} M rows/s")
    return 0


def _row_count(parquet_path: Path) -> int:
    import pyarrow.parquet as pq

    return pq.ParquetFile(parquet_path).metadata.num_rows


if __name__ == "__main__":
    sys.exit(main())
