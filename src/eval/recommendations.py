"""Flight recommendation mode.

Given a trained pipeline and a DataFrame of candidate flights, rank them by
expected profit (in EUR) under the EC261 portfolio strategy and surface the
top-N "buy" candidates.

The math is *not* re-implemented here — it delegates to
:func:`src.eval.profit_metric.per_flight_threshold` (the per-flight
break-even cutoff τ\\*) and to the EV formula already encoded in
:func:`src.eval.threshold.rank_by_unit_profit`:

    EV_i = p_i · (α·C_i − T_i − c_claim)
         + (1 − p_i) · −(T_i + c_travel)

A flight is recommended ``BUY`` iff ``p_delay > τ*``, i.e. when the model's
predicted delay probability clears the EV-positive cutoff for that
particular ticket-price / distance combination.

Only the columns the user actually needs to make a buying decision are
returned (date, carrier, route, tail, scheduled departure, distance, haul
tier, p_delay, τ*, ticket price, EV, recommendation) — sorted by EV
descending.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.ec261 import KM_PER_MILE, compute_compensation
from .profit_metric import ProfitConfig, per_flight_threshold


REC_COLUMNS: tuple[str, ...] = (
    "FL_DATE",
    "OP_UNIQUE_CARRIER",
    "ORIGIN",
    "DEST",
    "TAIL_NUM",
    "CRS_DEP_TIME",
    "DISTANCE",
    "haul_tier",
    "p_delay",
    "tau_star",
    "ticket_price_eur",
    "ev_eur",
    "recommendation",
)


def _haul_tier(distance_km: np.ndarray, params) -> np.ndarray:
    """Return per-flight EC261 compensation tier as ``short|medium|long``.

    Aligned with :func:`src.data.ec261.compute_compensation`:
        short  : d ≤ short_haul_km
        medium : short_haul_km < d ≤ medium_haul_km
        long   : d  > medium_haul_km
    """
    d = np.asarray(distance_km, dtype=float)
    tier = np.full(d.shape, "medium", dtype=object)
    tier[d <= params.short_haul_km] = "short"
    tier[d > params.medium_haul_km] = "long"
    return tier


def find_best_flights(
    pipeline,
    df: pd.DataFrame,
    top_n: int = 20,
    cfg: ProfitConfig | None = None,
) -> pd.DataFrame:
    """Rank candidate flights by expected profit and return the top ``top_n``.

    Parameters
    ----------
    pipeline
        A fitted estimator/pipeline exposing ``predict_proba(X)``.  The
        canonical Pipeline built by :func:`src.pipeline.build.build_pipeline`
        accepts a *raw* BTS DataFrame and handles preprocessing internally.
    df : pandas.DataFrame
        Candidate flights.  Must contain at least the columns
        ``FL_DATE, OP_UNIQUE_CARRIER, ORIGIN, DEST, CRS_DEP_TIME, DISTANCE,
        T_eur``.  ``TAIL_NUM`` is included if present, otherwise filled with
        ``"—"``.
    top_n : int, default 20
        Number of rows to return (after sorting by EV desc).  Pass
        ``len(df)`` to score the full candidate set.
    cfg : ProfitConfig, optional
        EC261 + cost parameters.  Defaults match the report's headline
        configuration (α=0.65, c_claim=€15, c_travel=€50).

    Returns
    -------
    pandas.DataFrame
        ``top_n`` rows with columns :data:`REC_COLUMNS`, sorted by
        ``ev_eur`` descending.  ``recommendation`` is ``"BUY"`` when
        ``p_delay > tau_star``, else ``"SKIP"``.
    """
    cfg = cfg or ProfitConfig()

    if "T_eur" not in df.columns:
        raise KeyError(
            "find_best_flights requires a `T_eur` ticket-price column. "
            "Run src.data.loaders.add_ticket_price(df) first."
        )
    for required in ("FL_DATE", "OP_UNIQUE_CARRIER", "ORIGIN", "DEST",
                     "CRS_DEP_TIME", "DISTANCE"):
        if required not in df.columns:
            raise KeyError(
                f"find_best_flights requires `{required}` in candidate frame."
            )

    if len(df) == 0:
        return pd.DataFrame(columns=list(REC_COLUMNS))

    df = df.reset_index(drop=True)

    proba = pipeline.predict_proba(df)
    p_delay = np.asarray(proba)[:, 1].astype(float)

    T = df["T_eur"].to_numpy(dtype=float)
    distance_mi = pd.to_numeric(df["DISTANCE"], errors="coerce").to_numpy(dtype=float)
    distance_km = distance_mi * KM_PER_MILE

    tau_star = per_flight_threshold(T, distance_km, cfg)
    C = compute_compensation(distance_km, cfg.params)

    # Identical EV formulation to rank_by_unit_profit() in threshold.py.
    ev = (
        p_delay * (cfg.alpha * C - T - cfg.claim_cost)
        + (1.0 - p_delay) * (-(T + cfg.travel_cost))
    )

    tail = (
        df["TAIL_NUM"].astype("string").fillna("—")
        if "TAIL_NUM" in df.columns
        else pd.Series(["—"] * len(df))
    )

    out = pd.DataFrame({
        "FL_DATE": pd.to_datetime(df["FL_DATE"]).dt.date,
        "OP_UNIQUE_CARRIER": df["OP_UNIQUE_CARRIER"].astype("string"),
        "ORIGIN": df["ORIGIN"].astype("string"),
        "DEST": df["DEST"].astype("string"),
        "TAIL_NUM": tail.values,
        "CRS_DEP_TIME": pd.to_numeric(df["CRS_DEP_TIME"], errors="coerce")
            .astype("Int64"),
        "DISTANCE": pd.to_numeric(df["DISTANCE"], errors="coerce")
            .astype("Int64"),
        "haul_tier": _haul_tier(distance_km, cfg.params),
        "p_delay": p_delay,
        "tau_star": tau_star,
        "ticket_price_eur": np.round(T, 2),
        "ev_eur": np.round(ev, 2),
        "recommendation": np.where(p_delay > tau_star, "BUY", "SKIP"),
    })

    out = out.sort_values("ev_eur", ascending=False, kind="mergesort")
    return out.head(int(top_n)).reset_index(drop=True)
