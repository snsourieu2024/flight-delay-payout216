"""Tests for the flight-recommendation mode."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.ec261 import KM_PER_MILE
from src.eval.profit_metric import ProfitConfig, per_flight_threshold
from src.eval.recommendations import REC_COLUMNS, find_best_flights


class _StubPipeline:
    """Minimal stand-in: returns a fixed P(delay) per row."""

    def __init__(self, p: np.ndarray):
        self.p = np.asarray(p, dtype=float)

    def predict_proba(self, X) -> np.ndarray:
        n = len(X)
        if len(self.p) != n:
            raise AssertionError(f"stub got {n} rows, expected {len(self.p)}")
        return np.column_stack([1.0 - self.p, self.p])


def _make_candidates(n: int = 5) -> pd.DataFrame:
    return pd.DataFrame({
        "FL_DATE": pd.to_datetime(["2024-03-01"] * n),
        "OP_UNIQUE_CARRIER": ["AA", "DL", "UA", "WN", "B6"][:n],
        "ORIGIN": ["JFK", "ORD", "ATL", "DFW", "LAX"][:n],
        "DEST": ["LAX", "MIA", "SEA", "BOS", "JFK"][:n],
        "TAIL_NUM": [f"N{1000 + i:04d}" for i in range(n)],
        "CRS_DEP_TIME": [800, 1200, 1830, 600, 2200][:n],
        "DISTANCE": [2475, 1197, 2182, 1562, 2475][:n],  # miles
        "T_eur": [180.0, 90.0, 220.0, 120.0, 60.0][:n],
    })


def test_returns_expected_columns_and_sort_order():
    df = _make_candidates(5)
    p = np.array([0.30, 0.05, 0.65, 0.20, 0.40])
    out = find_best_flights(_StubPipeline(p), df, top_n=3)

    assert list(out.columns) == list(REC_COLUMNS)
    assert len(out) == 3
    assert (out["ev_eur"].diff().dropna() <= 0).all()


def test_recommendation_matches_per_flight_threshold():
    df = _make_candidates(5)
    p = np.array([0.30, 0.05, 0.65, 0.20, 0.40])
    cfg = ProfitConfig()
    out = find_best_flights(_StubPipeline(p), df, top_n=len(df), cfg=cfg)

    distance_km = out["DISTANCE"].astype(float).to_numpy() * KM_PER_MILE
    expected_tau = per_flight_threshold(
        out["ticket_price_eur"].to_numpy(dtype=float),
        distance_km,
        cfg,
    )
    np.testing.assert_allclose(out["tau_star"].to_numpy(), expected_tau, atol=1e-9)

    expected_rec = np.where(out["p_delay"].to_numpy() > expected_tau, "BUY", "SKIP")
    assert (out["recommendation"].to_numpy() == expected_rec).all()


def test_haul_tier_aligns_with_ec261_thresholds():
    df = pd.DataFrame({
        "FL_DATE": pd.to_datetime(["2024-01-01"] * 3),
        "OP_UNIQUE_CARRIER": ["AA", "AA", "AA"],
        "ORIGIN": ["JFK", "JFK", "JFK"],
        "DEST": ["BOS", "LAX", "LHR"],
        "TAIL_NUM": ["N1", "N2", "N3"],
        "CRS_DEP_TIME": [800, 800, 800],
        # 200mi → ≤1500 km (short); 2000mi → 1500–3500 km (medium);
        # 5000mi → >3500 km (long).
        "DISTANCE": [200, 2000, 5000],
        "T_eur": [50.0, 200.0, 500.0],
    })
    p = np.array([0.5, 0.5, 0.5])
    out = find_best_flights(_StubPipeline(p), df, top_n=len(df))
    by_dest = dict(zip(out["DEST"], out["haul_tier"]))
    assert by_dest["BOS"] == "short"
    assert by_dest["LAX"] == "medium"
    assert by_dest["LHR"] == "long"


def test_empty_input_returns_empty_frame_with_schema():
    df = pd.DataFrame(columns=[
        "FL_DATE", "OP_UNIQUE_CARRIER", "ORIGIN", "DEST",
        "CRS_DEP_TIME", "DISTANCE", "T_eur",
    ])
    out = find_best_flights(_StubPipeline(np.array([])), df, top_n=10)
    assert list(out.columns) == list(REC_COLUMNS)
    assert len(out) == 0


def test_missing_ticket_price_raises():
    df = _make_candidates(2).drop(columns=["T_eur"])
    try:
        find_best_flights(_StubPipeline(np.array([0.3, 0.4])), df)
    except KeyError as e:
        assert "T_eur" in str(e)
    else:
        raise AssertionError("expected KeyError for missing T_eur")


def test_ev_formula_matches_threshold_module():
    """Sanity check: per-row EV = p·(αC − T − c_claim) + (1−p)·−(T + c_travel)."""
    from src.data.ec261 import compute_compensation

    df = _make_candidates(4)
    p = np.array([0.10, 0.50, 0.25, 0.80])
    cfg = ProfitConfig()
    out = find_best_flights(_StubPipeline(p), df, top_n=len(df), cfg=cfg)

    # Re-derive EV from the returned table and check it matches the formula.
    T = out["ticket_price_eur"].to_numpy(dtype=float)
    distance_km = out["DISTANCE"].astype(float).to_numpy() * KM_PER_MILE
    C = compute_compensation(distance_km, cfg.params)
    p_out = out["p_delay"].to_numpy()
    expected_ev = (
        p_out * (cfg.alpha * C - T - cfg.claim_cost)
        + (1.0 - p_out) * (-(T + cfg.travel_cost))
    )
    np.testing.assert_allclose(out["ev_eur"].to_numpy(), np.round(expected_ev, 2),
                               atol=0.01)
