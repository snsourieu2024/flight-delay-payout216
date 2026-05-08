"""Failure-mode analysis: bin test errors by airport, hour, distance, weather.

Produces tables that go straight into the report's *Interpretation and
Reflection* section (rubric: 10 pts on advanced methods, 20 pts on
evaluation, 10 pts on report clarity).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import EC261, EC261Params
from .profit_metric import ProfitConfig, buy_decisions, realised_profit_per_flight
from ..data.ec261 import KM_PER_MILE


def error_table_by(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    by: str,
    threshold: float = 0.5,
    top_k: int = 20,
) -> pd.DataFrame:
    y_pred = (y_prob > threshold).astype(int)
    fp = (y_pred == 1) & (y_true == 0)
    fn = (y_pred == 0) & (y_true == 1)
    tp = (y_pred == 1) & (y_true == 1)
    tn = (y_pred == 0) & (y_true == 0)
    grouped = (
        pd.DataFrame({
            "by": df[by].astype(str).values,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "y": y_true,
        })
        .groupby("by")
        .agg(n=("y", "size"), pos=("y", "sum"),
             tp=("tp", "sum"), fp=("fp", "sum"),
             fn=("fn", "sum"), tn=("tn", "sum"))
        .sort_values("n", ascending=False)
    )
    grouped["base_rate"] = grouped["pos"] / grouped["n"]
    grouped["precision"] = grouped["tp"] / (grouped["tp"] + grouped["fp"]).replace(0, 1)
    grouped["recall"] = grouped["tp"] / (grouped["tp"] + grouped["fn"]).replace(0, 1)
    return grouped.head(top_k)


def loss_makers(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    ticket_price_eur: np.ndarray,
    cfg: ProfitConfig | None = None,
    top_k: int = 20,
) -> pd.DataFrame:
    """Return the top-k flights that contributed the most negative profit.

    Useful for case-study paragraphs in the report.
    """
    cfg = cfg or ProfitConfig()
    distance_km = df["DISTANCE"].to_numpy(dtype=float) * KM_PER_MILE
    buy = buy_decisions(y_prob, ticket_price_eur, distance_km, cfg)
    profit = realised_profit_per_flight(
        y_true, buy, ticket_price_eur, distance_km, cfg
    )
    out = df.assign(
        y_true=y_true, y_prob=y_prob, buy=buy,
        profit_eur=profit, ticket_eur=ticket_price_eur,
    )
    return out.sort_values("profit_eur").head(top_k)
