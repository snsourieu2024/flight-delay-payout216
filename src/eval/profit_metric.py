"""Custom profit / ROI metric with the corrected cost matrix.

This is the headline contribution of the project (and the answer to the
professor's "second-layer analysis" feedback).  The cost matrix used here
is intentionally *different* from Option 2 of the original prompt set: that
prompt scored False Negatives at €0, which is wrong for a portfolio
strategy where every FN forgoes a positive-EV opportunity.

Definitions
-----------

For every flight i with predicted probability p̂_i, ticket price T_i,
distance-tiered compensation C_i = C(d_i), and binary outcome y_i:

    buy_i = 1 iff p̂_i > τ_i*

where the **per-flight** optimal threshold is

    τ_i* = (T_i + c_travel) / (α · C_i − c_claim + c_travel)

α            empirical claim-success probability  (default 0.65)
c_claim      cost of pursuing the claim (€15 self-filed)
c_travel     unrecoverable opportunity cost of taking the flight (€50 default)

Per-flight realised profit:

    profit_i =
        +(α · C_i − T_i − c_claim)   if  buy_i = 1 and y_i = 1   (TP)
        −(T_i + c_travel)             if  buy_i = 1 and y_i = 0   (FP)
        −γ · E[profit | y=1]          if  buy_i = 0 and y_i = 1   (FN, opportunity cost)
        0                              if  buy_i = 0 and y_i = 0   (TN)

The opportunity-cost weight γ ∈ [0, 1] models the fraction of FN flights
that the bankroll *could* have funded.  For an unconstrained bankroll γ=1;
for a tight bankroll where most FN candidates would have been crowded out
anyway, γ→0.  The default γ=0 reproduces the simple Option-2 metric for
comparison; the report uses γ=1 as the headline.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import EC261, EC261Params
from ..data.ec261 import compute_compensation, KM_PER_MILE


@dataclass
class ProfitConfig:
    params: EC261Params = EC261
    opportunity_cost_gamma: float = 1.0
    use_per_flight_threshold: bool = True
    global_threshold: float = 0.5

    @property
    def alpha(self) -> float:
        return self.params.claim_success_rate

    @property
    def claim_cost(self) -> float:
        return self.params.claim_cost_eur

    @property
    def travel_cost(self) -> float:
        return self.params.travel_cost_eur


def per_flight_threshold(
    ticket_price_eur: np.ndarray,
    distance_km: np.ndarray,
    cfg: ProfitConfig | None = None,
) -> np.ndarray:
    """Compute τ*(T_i, d_i) — the per-flight EV-positive cutoff."""
    cfg = cfg or ProfitConfig()
    C = compute_compensation(distance_km, cfg.params)
    denom = cfg.alpha * C - cfg.claim_cost + cfg.travel_cost
    denom = np.where(denom <= 0, np.inf, denom)
    tau = (ticket_price_eur + cfg.travel_cost) / denom
    return np.clip(tau, 0.0, 1.0)


def buy_decisions(
    y_prob: np.ndarray,
    ticket_price_eur: np.ndarray,
    distance_km: np.ndarray,
    cfg: ProfitConfig | None = None,
) -> np.ndarray:
    """Return a 0/1 array: 1 = buy this ticket."""
    cfg = cfg or ProfitConfig()
    if cfg.use_per_flight_threshold:
        thresholds = per_flight_threshold(ticket_price_eur, distance_km, cfg)
        return (y_prob > thresholds).astype(int)
    return (y_prob > cfg.global_threshold).astype(int)


def realised_profit_per_flight(
    y_true: np.ndarray,
    buy: np.ndarray,
    ticket_price_eur: np.ndarray,
    distance_km: np.ndarray,
    cfg: ProfitConfig | None = None,
) -> np.ndarray:
    cfg = cfg or ProfitConfig()
    y_true = np.asarray(y_true).astype(int)
    buy = np.asarray(buy).astype(int)
    T = np.asarray(ticket_price_eur, dtype=float)
    C = compute_compensation(distance_km, cfg.params)

    profit = np.zeros_like(T, dtype=float)

    tp_mask = (buy == 1) & (y_true == 1)
    profit[tp_mask] = cfg.alpha * C[tp_mask] - T[tp_mask] - cfg.claim_cost

    fp_mask = (buy == 1) & (y_true == 0)
    profit[fp_mask] = -(T[fp_mask] + cfg.travel_cost)

    if cfg.opportunity_cost_gamma > 0:
        fn_mask = (buy == 0) & (y_true == 1)
        avg_realised_pos = (
            (cfg.alpha * C[fn_mask] - T[fn_mask] - cfg.claim_cost).mean()
            if fn_mask.any()
            else 0.0
        )
        profit[fn_mask] = -cfg.opportunity_cost_gamma * max(0.0, avg_realised_pos)

    return profit


def total_roi(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    ticket_price_eur: np.ndarray,
    distance_km: np.ndarray,
    cfg: ProfitConfig | None = None,
) -> dict[str, float]:
    """Compute total profit and ROI for a given probability set.

    Returns a dict with profit_total, capital_at_risk, roi, n_buys.
    """
    cfg = cfg or ProfitConfig()
    buy = buy_decisions(y_prob, ticket_price_eur, distance_km, cfg)
    profit = realised_profit_per_flight(
        y_true, buy, ticket_price_eur, distance_km, cfg
    )
    capital = (buy * ticket_price_eur).sum()
    return {
        "profit_total_eur": float(profit.sum()),
        "capital_at_risk_eur": float(capital),
        "roi": float(profit.sum() / capital) if capital > 0 else 0.0,
        "n_buys": int(buy.sum()),
        "n_total": int(len(y_true)),
        "buy_rate": float(buy.mean()),
    }


def profit_scorer(cfg: ProfitConfig | None = None):
    """Return a Sklearn-compatible scorer.

    Use as ``cv=..., scoring=profit_scorer()`` in GridSearchCV.  The X passed
    to the scorer must be the *raw* DataFrame (the Pipeline handles
    transformations) so it still has DISTANCE and the price column.
    """
    cfg = cfg or ProfitConfig()

    def _score(estimator, X: pd.DataFrame, y) -> float:
        if "T_eur" not in X.columns:
            raise KeyError("profit_scorer requires X to have a `T_eur` ticket-price column")
        y_prob = estimator.predict_proba(X)[:, 1]
        distance_km = X["DISTANCE"].to_numpy(dtype=float) * KM_PER_MILE
        return total_roi(y, y_prob, X["T_eur"].to_numpy(dtype=float), distance_km, cfg)["roi"]

    _score.__name__ = "profit_roi"
    return _score
