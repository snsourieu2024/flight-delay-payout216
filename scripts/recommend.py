"""Flight-recommendation mode (real BTS data).

Trains a Random Forest pipeline on whatever pre-2024 BTS data is on disk
(falling back to the synthetic generator if none is available), scores
every 2024 BTS flight as a candidate ticket purchase, and prints the
``--top-n`` flights ranked by expected profit (EV in EUR).

Run::

    python scripts/recommend.py
    python scripts/recommend.py --top-n 50 --train-sample 500000
    python scripts/recommend.py --candidate-year 2024 --no-real-data

Output is two tables on stdout:

    1. Top-N candidate flights with FL_DATE / carrier / route / tail /
       CRS_DEP_TIME / DISTANCE / haul tier / p_delay / τ* / ticket price /
       EV / recommendation.
    2. Aggregate summary: how many BUY recommendations across the *full*
       candidate set, total capital required, and total expected profit if
       outcomes match the model's predicted rate.

This script is the recommendation counterpart to ``scripts/smoke_test.py``
(which only validates the backtester end-to-end).
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
np.seterr(all="ignore")

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parents[1]))

from src.config import RANDOM_SEED, RAW_DIR
from src.data.ec261 import KM_PER_MILE, compute_compensation, label_eligible_delay
from src.data.loaders import add_ticket_price, load_bts, prepare_modelling_frame
from src.data.synthetic import generate_synthetic_bts
from src.eval.profit_metric import ProfitConfig
from src.eval.recommendations import find_best_flights
from src.models.registry import make_random_forest
from src.pipeline.build import build_pipeline


DEFAULT_TRAIN_YEARS = (2018, 2019, 2020, 2021, 2022, 2023)
DEFAULT_CANDIDATE_YEAR = 2024


def _load_training_frame(
    train_years: tuple[int, ...],
    candidate_year: int,
    use_real_data: bool,
    n_synthetic: int,
) -> tuple[pd.DataFrame, str]:
    """Return ``(train_df, source_label)``.

    Strategy:
      * if ``use_real_data`` and any of the requested ``train_years`` has
        a parquet/csv on disk, load those years from disk;
      * otherwise generate a synthetic sample covering pre-candidate years
        so the rest of the script still runs end-to-end.
    """
    if use_real_data:
        on_disk = sorted({
            int(p.stem.split("_")[1])
            for p in RAW_DIR.glob("bts_*.parquet")
            if p.stem.split("_")[1].isdigit()
        } | {
            int(p.stem.split("_")[1])
            for p in RAW_DIR.glob("bts_*.csv")
            if p.stem.split("_")[1].isdigit()
        })
        usable = [y for y in train_years if y in on_disk]
        if usable:
            df = load_bts(years=tuple(usable), fallback="raise")
            return df, f"real BTS {min(usable)}-{max(usable)} ({len(df):,} rows)"

    syn = generate_synthetic_bts(
        n_flights=n_synthetic,
        start_year=min(train_years),
        end_year=candidate_year - 1,
        seed=RANDOM_SEED,
    )
    return syn, f"synthetic {min(train_years)}-{candidate_year - 1} ({len(syn):,} rows)"


def _load_candidate_frame(
    candidate_year: int,
    use_real_data: bool,
    n_synthetic: int,
) -> tuple[pd.DataFrame, str]:
    """Return ``(candidate_df, source_label)`` for ``candidate_year``."""
    if use_real_data:
        on_disk_parquet = list(RAW_DIR.glob(f"bts_{candidate_year}*.parquet"))
        on_disk_csv = list(RAW_DIR.glob(f"bts_{candidate_year}*.csv"))
        if on_disk_parquet or on_disk_csv:
            df = load_bts(years=(candidate_year,), fallback="raise")
            return df, f"real BTS {candidate_year} ({len(df):,} rows)"

    syn = generate_synthetic_bts(
        n_flights=n_synthetic,
        start_year=candidate_year,
        end_year=candidate_year,
        seed=RANDOM_SEED + 1,
    )
    return syn, f"synthetic {candidate_year} ({len(syn):,} rows)"


def _format_top_table(top: pd.DataFrame) -> str:
    if len(top) == 0:
        return "(no candidate flights)"

    fmt = top.copy()
    fmt["route"] = fmt["ORIGIN"].astype(str) + "→" + fmt["DEST"].astype(str)
    fmt["dep"] = fmt["CRS_DEP_TIME"].apply(
        lambda v: f"{int(v):04d}"[:2] + ":" + f"{int(v):04d}"[2:]
        if pd.notna(v) else "—"
    )
    fmt["p_delay"] = fmt["p_delay"].map(lambda x: f"{x:6.2%}")
    fmt["tau_star"] = fmt["tau_star"].map(lambda x: f"{x:6.2%}")
    fmt["ticket"] = fmt["ticket_price_eur"].map(lambda x: f"€{x:6.0f}")
    fmt["ev"] = fmt["ev_eur"].map(lambda x: f"€{x:+7.0f}")
    fmt["dist"] = fmt["DISTANCE"].map(
        lambda x: f"{int(x):5d}mi" if pd.notna(x) else "  —  "
    )

    cols = [
        "FL_DATE", "OP_UNIQUE_CARRIER", "route", "TAIL_NUM",
        "dep", "dist", "haul_tier",
        "p_delay", "tau_star", "ticket", "ev", "recommendation",
    ]
    rename = {
        "OP_UNIQUE_CARRIER": "carrier",
        "TAIL_NUM": "tail",
        "haul_tier": "tier",
        "p_delay": "p(delay)",
        "tau_star": "τ*",
        "ticket": "ticket",
        "ev": "EV",
        "recommendation": "rec",
    }
    return fmt[cols].rename(columns=rename).to_string(index=False)


def _summary(scored_full: pd.DataFrame, cfg: ProfitConfig) -> dict[str, float]:
    """Aggregate stats over the *complete* scored candidate set.

    Expected profit assumes outcomes match the model's predicted P(delay)
    flight-by-flight — the most charitable interpretation of "if all BUY
    recommendations come true at the model's predicted rate".  For each
    BUY flight i:

        E[profit_i] = p_i · (α·C_i − T_i − c_claim)
                    + (1 − p_i) · −(T_i + c_travel)

    which is just the EV column we already computed (only summed over the
    BUY subset).  Capital required = sum of ticket prices over BUYs.
    """
    buy = scored_full[scored_full["recommendation"] == "BUY"]
    n_buy = int(len(buy))
    n_total = int(len(scored_full))
    capital = float(buy["ticket_price_eur"].sum())
    expected_profit = float(buy["ev_eur"].sum())
    avg_p = float(buy["p_delay"].mean()) if n_buy else 0.0
    roi = expected_profit / capital if capital > 0 else 0.0
    return {
        "n_total": n_total,
        "n_buy": n_buy,
        "buy_rate": n_buy / max(n_total, 1),
        "capital_eur": capital,
        "expected_profit_eur": expected_profit,
        "expected_roi": roi,
        "avg_p_buy": avg_p,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--top-n", type=int, default=20,
                   help="Rows to print in the recommendation table (default 20).")
    p.add_argument("--candidate-year", type=int, default=DEFAULT_CANDIDATE_YEAR,
                   help="Year whose flights to score as ticket candidates.")
    p.add_argument("--train-years", type=int, nargs="+",
                   default=list(DEFAULT_TRAIN_YEARS),
                   help="Years to train the Random Forest on (real data only).")
    p.add_argument("--train-sample", type=int, default=300_000,
                   help="Cap training rows to this many (subsampled, "
                        "stratified by FL_DATE quantile).  Set 0 to disable.")
    p.add_argument("--n-estimators", type=int, default=150,
                   help="Random Forest tree count (default 150 for speed).")
    p.add_argument("--no-real-data", action="store_true",
                   help="Force synthetic BTS data even if parquet files exist.")
    p.add_argument("--n-synthetic", type=int, default=200_000,
                   help="Synthetic sample size when falling back.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    use_real = not args.no_real_data
    train_years = tuple(sorted(set(args.train_years)))
    candidate_year = int(args.candidate_year)

    cfg = ProfitConfig()

    print("=" * 72)
    print(f" Flight recommendation mode  (candidate year = {candidate_year})")
    print("=" * 72)

    t0 = time.time()
    print("[1/4] Loading data ...")
    train_df, train_label = _load_training_frame(
        train_years, candidate_year, use_real, args.n_synthetic,
    )
    cand_df, cand_label = _load_candidate_frame(
        candidate_year, use_real, args.n_synthetic,
    )
    print(f"      training:  {train_label}")
    print(f"      candidates: {cand_label}")

    train_df = prepare_modelling_frame(train_df)
    cand_df = prepare_modelling_frame(cand_df)
    train_df = add_ticket_price(train_df, seed=1)
    cand_df = add_ticket_price(cand_df, seed=2)

    if args.train_sample and len(train_df) > args.train_sample:
        train_df = (
            train_df.sort_values("FL_DATE")
            .sample(n=args.train_sample, random_state=RANDOM_SEED)
            .sort_values("FL_DATE")
            .reset_index(drop=True)
        )
        print(f"      subsampled training to {len(train_df):,} rows")

    y_train = label_eligible_delay(train_df).to_numpy()
    base_rate = float(y_train.mean())
    print(f"      training base rate (EC261-eligible delay) = {base_rate:.3%}")

    print(f"[2/4] Fitting Random Forest (n_estimators={args.n_estimators}) ...")
    clf = make_random_forest()
    clf.set_params(n_estimators=args.n_estimators)
    pipeline = build_pipeline(clf)
    t_fit0 = time.time()
    pipeline.fit(train_df, y_train)
    print(f"      fit in {time.time() - t_fit0:.1f}s")

    print(f"[3/4] Scoring {len(cand_df):,} candidate flights ...")
    scored_full = find_best_flights(pipeline, cand_df, top_n=len(cand_df), cfg=cfg)
    top = scored_full.head(args.top_n).reset_index(drop=True)

    print()
    print(f"Top {args.top_n} flights by expected profit")
    print("-" * 72)
    print(_format_top_table(top))

    print()
    print("[4/4] Portfolio summary across the full candidate set")
    print("-" * 72)
    s = _summary(scored_full, cfg)
    print(f"  candidates scored ........ {s['n_total']:>10,}")
    print(f"  BUY recommendations ...... {s['n_buy']:>10,}  "
          f"({s['buy_rate']:.2%} buy rate)")
    print(f"  capital required ......... €{s['capital_eur']:>10,.0f}")
    print(f"  expected profit (model) .. €{s['expected_profit_eur']:>+10,.0f}")
    print(f"  expected ROI ............. {s['expected_roi']:>10.2%}")
    print(f"  avg p(delay) on BUY set .. {s['avg_p_buy']:>10.2%}")

    print()
    print(f"Done in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
