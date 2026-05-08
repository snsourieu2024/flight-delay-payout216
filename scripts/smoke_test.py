"""End-to-end smoke test on synthetic data.

Runs the full pipeline (data → features → model → metrics → threshold sweep)
on a 60 k-row synthetic sample.  Should complete in well under a minute on
a laptop.  Used by CI to fail fast on any regression.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
np.seterr(all="ignore")

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parents[1]))

from src.data.ec261 import KM_PER_MILE, label_eligible_delay
from src.data.loaders import add_ticket_price, load_bts, prepare_modelling_frame
from src.eval.profit_metric import ProfitConfig, total_roi
from src.eval.threshold import bankroll_constrained_profit, profit_curve
from src.models.registry import make_logistic_regression, make_random_forest
from src.pipeline.build import build_pipeline
from src.pipeline.splits import temporal_split


def main() -> int:
    from src.config import RAW_DIR

    t0 = time.time()
    has_real = any(RAW_DIR.glob("bts_*.parquet")) or any(RAW_DIR.glob("bts_*.csv"))
    print(f"Loading BTS data ({'real CSV/parquet on disk' if has_real else 'synthetic fallback'}) ...")
    df = load_bts(fallback="synthetic", n_synthetic=60_000)
    df = prepare_modelling_frame(df)
    df = add_ticket_price(df, seed=1)
    y = label_eligible_delay(df).to_numpy()
    print(f"  shape={df.shape}, base rate={y.mean():.3%}, "
          f"date range={df['FL_DATE'].min().date()} → {df['FL_DATE'].max().date()}")

    split = temporal_split(df)
    print(f"  train={len(split.train_idx):,}  val={len(split.val_idx):,}  test={len(split.test_idx):,}")

    X_train = df.iloc[split.train_idx].reset_index(drop=True)
    y_train = y[split.train_idx]
    X_test = df.iloc[split.test_idx].reset_index(drop=True)
    y_test = y[split.test_idx]
    T_test = X_test["T_eur"].to_numpy()
    d_test = X_test["DISTANCE"].to_numpy() * KM_PER_MILE

    for name, factory in [("LogReg", make_logistic_regression), ("RF", make_random_forest)]:
        print(f"Fitting {name} ...")
        clf = factory()
        if name == "RF":
            clf.set_params(n_estimators=80)
        pipe = build_pipeline(clf)
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]

        roi = total_roi(y_test, proba, T_test, d_test, ProfitConfig())
        print(f"  ROI(per-flight tau*) = {roi['roi']:.3%}, profit=€{roi['profit_total_eur']:.0f}, n_buys={roi['n_buys']}")

        curve = profit_curve(y_test, proba, T_test, d_test, ProfitConfig())
        best = curve.loc[curve["roi"].idxmax()]
        print(f"  ROI(best global tau={best['threshold']:.2f}) = {best['roi']:.3%}, profit=€{best['profit_total_eur']:.0f}")

        bk = bankroll_constrained_profit(y_test, proba, T_test, d_test, bankroll_eur=10_000)
        print(f"  bankroll €10k -> profit=€{bk['profit_total_eur']:.0f}, ROI={bk['roi']:.3%}, n_buys={bk['n_buys']}")

    elapsed = time.time() - t0
    print(f"\nSmoke test passed in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
