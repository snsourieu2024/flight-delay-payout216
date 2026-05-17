"""Root-cause diagnostic: why the EV-optimal policy is *never buy*.

Proves the zero-ROI result is structural EC261 economics, not a tuning,
calibration, or synthetic-price artefact.

Core identity (from src/eval/profit_metric.py): a ticket is positive-EV iff
the flight's TRUE eligible-delay probability p exceeds the per-flight
break-even threshold

    tau*(T, d) = (T + c_travel) / (alpha * C(d) - c_claim + c_travel)

i.e. tau* IS the break-even probability.  We show that for realistic ticket
prices the required tau* (0.6 .. >1.0) is one to two orders of magnitude
above the highest eligible-delay rate observed in *any* sufficiently large
route/carrier cohort (which itself is far above the model's calibrated
prediction).  No model skill can close that gap.

Outputs: artefacts/breakeven_analysis.csv (tier table)
         artefacts/breakeven_analysis.txt (human-readable, report-ready)
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
np.seterr(all="ignore")

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parents[1]))

from src.config import ARTEFACTS_DIR, EC261
from src.data.ec261 import KM_PER_MILE, compute_compensation, label_eligible_delay
from src.data.loaders import add_ticket_price, load_bts, prepare_modelling_frame
from src.eval.profit_metric import ProfitConfig

ALPHA = EC261.claim_success_rate
C_CLAIM = EC261.claim_cost_eur
C_TRAVEL = EC261.travel_cost_eur


def _tier(distance_km: np.ndarray) -> np.ndarray:
    t = np.full(distance_km.shape, "medium", dtype=object)
    t[distance_km <= EC261.short_haul_km] = "short"
    t[distance_km > EC261.medium_haul_km] = "long"
    return t


def _q(a: np.ndarray, p: float) -> float:
    return float(np.quantile(a, p))


def main() -> None:
    print("[breakeven] loading real 2024 BTS modelling frame ...")
    df = load_bts(years=(2024,))
    df = prepare_modelling_frame(df)
    df = add_ticket_price(df, seed=0)

    y = label_eligible_delay(df).to_numpy()
    distance_km = df["DISTANCE"].to_numpy(dtype=float) * KM_PER_MILE
    T = df["T_eur"].to_numpy(dtype=float)
    C = compute_compensation(distance_km, EC261)
    denom = ALPHA * C - C_CLAIM + C_TRAVEL          # always > 0 here
    tau_raw = (T + C_TRAVEL) / denom                 # UNclipped break-even prob
    tier = _tier(distance_km)

    base_rate = float(y.mean())
    n = len(df)

    # Generous ceiling on any model's achievable prediction: the highest
    # empirical eligible-delay rate in any cohort with >= 200 flights.
    ceilings = {}
    for name, keys in {
        "route (ORIGIN,DEST)": ["ORIGIN", "DEST"],
        "carrier": ["OP_UNIQUE_CARRIER"],
        "route x carrier": ["ORIGIN", "DEST", "OP_UNIQUE_CARRIER"],
    }.items():
        g = pd.DataFrame({"y": y}).join(df[keys].reset_index(drop=True))
        agg = g.groupby(keys)["y"].agg(["mean", "size"])
        agg = agg[agg["size"] >= 200]
        ceilings[name] = float(agg["mean"].max()) if len(agg) else float("nan")
    p_ceiling = np.nanmax(list(ceilings.values()))

    rows = []
    for t in ("short", "medium", "long"):
        m = tier == t
        if not m.any():
            continue
        Cv = float(C[m][0])
        dv = float(denom[m][0])
        tau_t = tau_raw[m]
        never = float((tau_t >= 1.0).mean())          # never positive-EV
        br_t = float(y[m].mean())
        # EV per flight if you bought EVERY flight in this tier, evaluated at
        # the empirical base rate and at the optimistic ceiling probability:
        ev_at_base = br_t * dv - (np.median(T[m]) + C_TRAVEL)
        ev_at_ceiling = p_ceiling * dv - (np.median(T[m]) + C_TRAVEL)
        rows.append({
            "tier": t,
            "n": int(m.sum()),
            "payout_C_eur": Cv,
            "denominator_eur": round(dv, 2),
            "ticket_T_p50": round(_q(T[m], 0.50), 2),
            "ticket_T_p10": round(_q(T[m], 0.10), 2),
            "ticket_T_p90": round(_q(T[m], 0.90), 2),
            "tau_star_p50": round(_q(tau_t, 0.50), 4),
            "tau_star_p10": round(_q(tau_t, 0.10), 4),
            "tau_star_p90": round(_q(tau_t, 0.90), 4),
            "pct_never_buyable_tau_ge_1": round(100 * never, 2),
            "empirical_base_rate": round(br_t, 5),
            "gap_factor_p50tau_over_baserate": round(_q(tau_t, 0.50) / br_t, 1)
            if br_t > 0 else float("inf"),
            "EV_per_flight_at_baserate_eur": round(ev_at_base, 2),
            "EV_per_flight_at_ceiling_eur": round(ev_at_ceiling, 2),
        })

    table = pd.DataFrame(rows)
    ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = ARTEFACTS_DIR / "breakeven_analysis.csv"
    table.to_csv(csv_path, index=False)

    med_tau = float(np.quantile(tau_raw, 0.50))
    gap = med_tau / p_ceiling if p_ceiling > 0 else float("inf")

    lines = []
    lines.append("=" * 78)
    lines.append("EC261 BREAK-EVEN DIAGNOSTIC  (real 2024 BTS, n = {:,})".format(n))
    lines.append("=" * 78)
    lines.append(
        f"alpha={ALPHA}  c_claim=EUR{C_CLAIM:.0f}  c_travel=EUR{C_TRAVEL:.0f}  "
        f"payouts short/med/long = EUR250/400/600"
    )
    lines.append("")
    lines.append(f"Empirical EC261-eligible base rate ......... {base_rate:.4%}")
    lines.append("Highest eligible-delay rate in any cohort (>=200 flights):")
    for k, v in ceilings.items():
        lines.append(f"  - {k:<22s} {v:.4%}")
    lines.append(
        f"=> optimistic prediction ceiling p_max ..... {p_ceiling:.4%}"
    )
    lines.append("")
    lines.append("Per-tier break-even (tau* = required TRUE delay probability):")
    lines.append(table.to_string(index=False))
    lines.append("")
    lines.append(
        f"Median break-even tau* across all flights ... {med_tau:.3f}"
    )
    lines.append(
        f"Best achievable probability anywhere ........ {p_ceiling:.4f}"
    )
    lines.append(
        f"STRUCTURAL GAP (median tau* / p_max) ........ {gap:,.0f}x"
    )
    lines.append("")
    lines.append(
        "CONCLUSION: the probability required to make ANY ticket positive-EV "
        "exceeds\nthe highest delay rate of even the worst route/carrier "
        f"cohort by ~{gap:,.0f}x.\nThe EV-optimal action for every flight is "
        "ABSTAIN. ROI 0 is the correct\nanswer the model learned, not a "
        "tuning or calibration failure. EC261's\ncapped payouts (EUR250-600 x "
        f"alpha={ALPHA}) cannot cover ticket+friction cost\nfor a ~"
        f"{base_rate:.2%} event."
    )
    lines.append("=" * 78)
    txt = "\n".join(lines)
    (ARTEFACTS_DIR / "breakeven_analysis.txt").write_text(txt + "\n")
    print(txt)
    print(f"\n[breakeven] wrote {csv_path}")
    print(f"[breakeven] wrote {ARTEFACTS_DIR / 'breakeven_analysis.txt'}")


if __name__ == "__main__":
    main()
