"""Tests for threshold optimisation."""
from __future__ import annotations

import numpy as np

from src.eval.profit_metric import ProfitConfig
from src.eval.threshold import (
    bankroll_constrained_profit,
    per_flight_threshold_grid,
    profit_curve,
    rank_by_unit_profit,
)


def test_profit_curve_threshold_grid_is_monotone():
    rng = np.random.default_rng(0)
    n = 1000
    y = (rng.uniform(size=n) < 0.05).astype(int)
    proba = np.clip(0.05 + 0.6 * y + rng.normal(0, 0.1, n), 0, 1)
    ticket = rng.uniform(50, 200, n)
    distance = rng.uniform(500, 5000, n)

    curve = profit_curve(y, proba, ticket, distance)
    assert (curve["threshold"].diff().dropna() > 0).all()
    # n_buys is monotonically non-increasing as the threshold rises
    assert (curve["n_buys"].diff().dropna() <= 0).all()


def test_profit_curve_zero_at_high_threshold_with_no_opp_cost():
    """Disable opportunity-cost (γ=0) → zero profit when nobody buys."""
    rng = np.random.default_rng(0)
    n = 500
    y = (rng.uniform(size=n) < 0.05).astype(int)
    proba = np.clip(0.05 + 0.6 * y + rng.normal(0, 0.1, n), 0, 1)
    ticket = rng.uniform(50, 200, n)
    distance = rng.uniform(500, 5000, n)
    cfg = ProfitConfig(opportunity_cost_gamma=0.0)
    curve = profit_curve(y, proba, ticket, distance, cfg)
    assert curve.iloc[-1]["profit_total_eur"] == 0
    assert curve.iloc[-1]["n_buys"] == 0


def test_rank_by_unit_profit_orders_high_ev_first():
    proba = np.array([0.9, 0.1, 0.9, 0.1])
    ticket = np.array([100.0, 100.0, 100.0, 100.0])
    distance = np.array([2000.0, 2000.0, 6000.0, 6000.0])
    order = rank_by_unit_profit(proba, ticket, distance)
    assert order[0] == 2  # high prob + long distance
    assert order[-1] in (1, 3)  # low prob


def test_bankroll_constrained_caps_capital():
    rng = np.random.default_rng(1)
    n = 200
    y = (rng.uniform(size=n) < 0.2).astype(int)
    proba = np.clip(0.2 + 0.5 * y + rng.normal(0, 0.05, n), 0, 1)
    ticket = rng.uniform(50, 200, n)
    distance = rng.uniform(500, 5000, n)
    out = bankroll_constrained_profit(y, proba, ticket, distance, bankroll_eur=1_000.0)
    assert out["capital_deployed_eur"] <= 1_000.0


def test_per_flight_threshold_grid_shape():
    grid = per_flight_threshold_grid(
        ticket_grid_eur=np.array([50.0, 100.0]),
        distance_grid_km=np.array([1000.0, 3000.0, 6000.0]),
    )
    assert grid.shape == (3, 2)
