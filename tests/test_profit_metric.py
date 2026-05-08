"""Tests for the custom profit metric."""
from __future__ import annotations

import numpy as np

from src.eval.profit_metric import (
    ProfitConfig,
    buy_decisions,
    per_flight_threshold,
    realised_profit_per_flight,
    total_roi,
)


def test_per_flight_threshold_increases_with_ticket_price():
    """Cheaper ticket → lower required confidence; expensive ticket → higher."""
    distance_km = np.array([2000.0, 2000.0, 2000.0])
    ticket = np.array([30.0, 100.0, 300.0])
    cfg = ProfitConfig()
    tau = per_flight_threshold(ticket, distance_km, cfg)
    assert tau[0] < tau[1] < tau[2]


def test_per_flight_threshold_decreases_with_distance():
    """Long-haul → higher payout → lower required confidence."""
    distance_km = np.array([800.0, 2500.0, 6000.0])
    ticket = np.array([100.0, 100.0, 100.0])
    cfg = ProfitConfig()
    tau = per_flight_threshold(ticket, distance_km, cfg)
    assert tau[0] > tau[1] > tau[2]


def test_buy_decisions_use_per_flight_threshold():
    proba = np.array([0.40, 0.40, 0.40])
    distance_km = np.array([800.0, 2500.0, 6000.0])
    ticket = np.array([100.0, 100.0, 100.0])
    cfg = ProfitConfig()
    buy = buy_decisions(proba, ticket, distance_km, cfg)
    # Long-haul gets bought first
    assert buy[2] >= buy[1] >= buy[0]


def test_tp_pays_compensation_minus_costs():
    cfg = ProfitConfig()
    profit = realised_profit_per_flight(
        y_true=np.array([1]),
        buy=np.array([1]),
        ticket_price_eur=np.array([100.0]),
        distance_km=np.array([2000.0]),
        cfg=cfg,
    )
    expected = cfg.alpha * 400.0 - 100.0 - cfg.claim_cost
    assert abs(profit[0] - expected) < 1e-6


def test_fp_loses_ticket_plus_travel_cost():
    cfg = ProfitConfig()
    profit = realised_profit_per_flight(
        y_true=np.array([0]),
        buy=np.array([1]),
        ticket_price_eur=np.array([100.0]),
        distance_km=np.array([2000.0]),
        cfg=cfg,
    )
    assert abs(profit[0] - (-(100.0 + cfg.travel_cost))) < 1e-6


def test_fn_zero_when_gamma_zero():
    cfg = ProfitConfig(opportunity_cost_gamma=0.0)
    profit = realised_profit_per_flight(
        y_true=np.array([1]),
        buy=np.array([0]),
        ticket_price_eur=np.array([100.0]),
        distance_km=np.array([2000.0]),
        cfg=cfg,
    )
    assert profit[0] == 0.0


def test_fn_negative_when_gamma_one():
    cfg = ProfitConfig(opportunity_cost_gamma=1.0)
    profit = realised_profit_per_flight(
        y_true=np.array([1, 1]),
        buy=np.array([0, 1]),
        ticket_price_eur=np.array([100.0, 100.0]),
        distance_km=np.array([2000.0, 2000.0]),
        cfg=cfg,
    )
    assert profit[0] < 0


def test_total_roi_includes_capital_at_risk():
    proba = np.array([0.95, 0.05, 0.95, 0.05])
    y = np.array([1, 0, 0, 1])
    ticket = np.array([100.0, 100.0, 100.0, 100.0])
    distance = np.array([2000.0, 2000.0, 2000.0, 2000.0])
    out = total_roi(y, proba, ticket, distance)
    assert out["n_buys"] >= 2  # At least the high-prob flights
    assert out["capital_at_risk_eur"] >= 200.0
