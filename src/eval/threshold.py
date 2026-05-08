"""Threshold optimisation — the "second layer" analysis.

Three artefacts produced here, all referenced by the report and slides:

1. ``profit_curve``: total ROI vs global threshold τ ∈ [0, 1].
2. ``per_flight_threshold_grid``: heatmap of τ*(T, d) over a 2D grid.
3. ``rank_by_unit_profit``: bankroll-constrained policy that ranks flights
   by expected EV-per-euro and buys top-k.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .profit_metric import (
    ProfitConfig,
    buy_decisions,
    per_flight_threshold,
    realised_profit_per_flight,
)
from ..data.ec261 import compute_compensation, KM_PER_MILE


def profit_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    ticket_price_eur: np.ndarray,
    distance_km: np.ndarray,
    cfg: ProfitConfig | None = None,
    grid: np.ndarray | None = None,
) -> pd.DataFrame:
    """Sweep a *global* threshold and report ROI + profit at each point.

    Useful as a sanity check against the per-flight threshold strategy:
    if per-flight beats the best global threshold, the second-layer
    analysis was worth it.
    """
    cfg = cfg or ProfitConfig()
    grid = grid if grid is not None else np.linspace(0.0, 1.0, 101)
    rows = []
    for tau in grid:
        cfg_g = ProfitConfig(
            params=cfg.params,
            opportunity_cost_gamma=cfg.opportunity_cost_gamma,
            use_per_flight_threshold=False,
            global_threshold=float(tau),
        )
        buy = buy_decisions(y_prob, ticket_price_eur, distance_km, cfg_g)
        profit = realised_profit_per_flight(
            y_true, buy, ticket_price_eur, distance_km, cfg_g
        )
        capital = float((buy * ticket_price_eur).sum())
        rows.append({
            "threshold": float(tau),
            "n_buys": int(buy.sum()),
            "buy_rate": float(buy.mean()),
            "profit_total_eur": float(profit.sum()),
            "capital_at_risk_eur": capital,
            "roi": float(profit.sum() / capital) if capital > 0 else 0.0,
        })
    return pd.DataFrame(rows)


def per_flight_threshold_grid(
    ticket_grid_eur: np.ndarray | None = None,
    distance_grid_km: np.ndarray | None = None,
    cfg: ProfitConfig | None = None,
) -> pd.DataFrame:
    """Tidy DataFrame of τ*(T, d) values for a heatmap."""
    cfg = cfg or ProfitConfig()
    ticket_grid_eur = (
        ticket_grid_eur if ticket_grid_eur is not None
        else np.linspace(20, 500, 25)
    )
    distance_grid_km = (
        distance_grid_km if distance_grid_km is not None
        else np.linspace(200, 8000, 25)
    )
    T, D = np.meshgrid(ticket_grid_eur, distance_grid_km)
    tau = per_flight_threshold(T.ravel(), D.ravel(), cfg).reshape(T.shape)
    return pd.DataFrame(tau, index=distance_grid_km, columns=ticket_grid_eur)


def rank_by_unit_profit(
    y_prob: np.ndarray,
    ticket_price_eur: np.ndarray,
    distance_km: np.ndarray,
    cfg: ProfitConfig | None = None,
) -> np.ndarray:
    """Rank flights by expected profit per euro at risk (descending).

    Used by the bankroll-constrained policy: buy in this order until the
    monthly budget is exhausted.
    """
    cfg = cfg or ProfitConfig()
    C = compute_compensation(distance_km, cfg.params)
    ev = (
        y_prob * (cfg.alpha * C - ticket_price_eur - cfg.claim_cost)
        + (1 - y_prob) * (-(ticket_price_eur + cfg.travel_cost))
    )
    score = ev / np.maximum(ticket_price_eur, 1e-6)
    return np.argsort(-score)


def bankroll_constrained_profit(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    ticket_price_eur: np.ndarray,
    distance_km: np.ndarray,
    bankroll_eur: float,
    cfg: ProfitConfig | None = None,
) -> dict[str, float]:
    """Greedily allocate a fixed bankroll across positive-EV flights.

    Returns total profit, ROI, and the number of tickets actually bought.
    """
    cfg = cfg or ProfitConfig()
    order = rank_by_unit_profit(y_prob, ticket_price_eur, distance_km, cfg)
    spent = 0.0
    bought = np.zeros_like(y_prob, dtype=int)
    for i in order:
        if y_prob[i] <= per_flight_threshold(
            np.array([ticket_price_eur[i]]),
            np.array([distance_km[i]]),
            cfg,
        )[0]:
            break
        if spent + ticket_price_eur[i] > bankroll_eur:
            continue
        bought[i] = 1
        spent += ticket_price_eur[i]

    profit = realised_profit_per_flight(y_true, bought, ticket_price_eur, distance_km, cfg)
    return {
        "profit_total_eur": float(profit.sum()),
        "capital_deployed_eur": float(spent),
        "roi": float(profit.sum() / spent) if spent > 0 else 0.0,
        "n_buys": int(bought.sum()),
        "bankroll_eur": float(bankroll_eur),
    }
