#!/usr/bin/env python3
"""Headline analysis of the EUROCONTROL R&D Archive cache.

Produces three artefacts:

* ``reports/figures/eurocontrol_delay_distribution.png``
* ``reports/figures/eurocontrol_yoy_march.png``
* ``reports/figures/eurocontrol_seasonality.png``

and one structured dump:

* ``reports/eurocontrol_summary.json``
* ``reports/eurocontrol_summary.txt``  (human-readable mirror of the JSON)

These feed:

* ``reports/eu_data_analysis.md`` (full chapter)
* ``reports/final_report.md`` (the "*fill in*" placeholders in section 4.3)
* ``README.md`` (storage table + EU validation strategy paragraph)

Run from repo root after ``scripts/process_eurocontrol.py``::

    python scripts/analyse_eurocontrol.py
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.config import EC261, REPORTS_DIR  # noqa: E402
from src.data.ec261 import KM_PER_MILE, compute_compensation  # noqa: E402
from src.data.eurocontrol import load_eu_processed  # noqa: E402


FIG_DIR = REPORTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

EC261_THRESHOLD_MIN = EC261.delay_threshold_min  # 180


def main() -> int:
    df = load_eu_processed()
    print(f"Loaded {len(df):,} flights from {df['FL_DATE'].min():%Y-%m-%d} "
          f"to {df['FL_DATE'].max():%Y-%m-%d}")

    df = df.copy()
    df["YEAR_MONTH"] = df["FL_DATE"].dt.to_period("M").astype(str)
    df["DELAYED_15M"] = (df["ARR_DELAY"] >= 15).astype(int)
    df["DELAYED_60M"] = (df["ARR_DELAY"] >= 60).astype(int)
    df["DELAYED_3H"] = (df["ARR_DELAY"] >= EC261_THRESHOLD_MIN).astype(int)
    df["DISTANCE_KM"] = df["DISTANCE"] * KM_PER_MILE

    df["DISTANCE_TIER"] = pd.cut(
        df["DISTANCE_KM"],
        bins=[-0.01, EC261.short_haul_km, EC261.medium_haul_km, np.inf],
        labels=["short_haul", "medium_haul", "long_haul"],
    )

    df["EC261_PAYOUT_EUR"] = compute_compensation(df["DISTANCE_KM"].to_numpy())

    summary: dict = {}
    summary["coverage"] = _coverage_block(df)
    summary["overall"] = _overall_block(df)
    summary["per_month"] = _per_month_block(df)
    summary["yoy_march"] = _yoy_march_block(df)
    summary["distance_tiers"] = _distance_tier_block(df)
    summary["top_operators"] = _top_operators_block(df)
    summary["top_routes"] = _top_routes_block(df)
    summary["hour_of_day"] = _hour_of_day_block(df)
    summary["expected_value"] = _expected_value_block(df)

    _save_dump(summary)
    _make_distribution_plot(df)
    _make_yoy_plot(df)
    _make_seasonality_plot(df)

    print()
    print("Summary written to reports/eurocontrol_summary.json")
    print("Figures written to reports/figures/eurocontrol_*.png")
    return 0


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------


def _operated_view(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with no actual times (cancellations / data gaps).

    All downstream delay statistics use this view because ARR_DELAY is NaN
    for cancellations and would silently bias means.
    """
    return df[df["CANCELLED"] != 1].copy()


def _summary_stats(arr_delay: pd.Series) -> dict:
    a = arr_delay.dropna().to_numpy()
    if a.size == 0:
        return {k: None for k in ("mean", "median", "p75", "p90", "p95", "p99", "max")}
    return {
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "p75": float(np.percentile(a, 75)),
        "p90": float(np.percentile(a, 90)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "max": float(a.max()),
    }


def _coverage_block(df: pd.DataFrame) -> dict:
    months = sorted(df["YEAR_MONTH"].unique())
    rows_per_month = (
        df.groupby("YEAR_MONTH").size().rename("n_flights").to_dict()
    )
    return {
        "months_present": months,
        "rows_per_month": rows_per_month,
        "total_rows": int(len(df)),
        "earliest": str(df["FL_DATE"].min().date()),
        "latest": str(df["FL_DATE"].max().date()),
        "n_unique_origins": int(df["ORIGIN"].nunique()),
        "n_unique_destinations": int(df["DEST"].nunique()),
        "n_unique_operators": int(df["OP_UNIQUE_CARRIER"].nunique()),
        "n_unique_aircraft_types": int(df["AIRCRAFT_TYPE"].nunique()),
    }


def _overall_block(df: pd.DataFrame) -> dict:
    op = _operated_view(df)
    cancelled = int((df["CANCELLED"] == 1).sum())
    delayed_15m = int(op["DELAYED_15M"].sum())
    delayed_60m = int(op["DELAYED_60M"].sum())
    delayed_3h = int(op["DELAYED_3H"].sum())
    return {
        "n_flights": int(len(df)),
        "n_operated_flights": int(len(op)),
        "n_cancelled_or_missing_actuals": cancelled,
        "cancelled_or_missing_rate": cancelled / max(1, len(df)),
        "n_delayed_15m": delayed_15m,
        "delayed_15m_rate": delayed_15m / max(1, len(op)),
        "n_delayed_60m": delayed_60m,
        "delayed_60m_rate": delayed_60m / max(1, len(op)),
        "n_delayed_3h": delayed_3h,
        "delayed_3h_rate": delayed_3h / max(1, len(op)),
        "arr_delay_stats_min": _summary_stats(op["ARR_DELAY"]),
        "dep_delay_stats_min": _summary_stats(op["DEP_DELAY"]),
    }


def _per_month_block(df: pd.DataFrame) -> dict:
    out: dict = {}
    for ym, group in df.groupby("YEAR_MONTH"):
        op = _operated_view(group)
        out[ym] = {
            "n_flights": int(len(group)),
            "n_operated": int(len(op)),
            "n_cancelled_or_missing": int((group["CANCELLED"] == 1).sum()),
            "delayed_15m_rate": float(op["DELAYED_15M"].mean())
            if len(op) > 0
            else None,
            "delayed_60m_rate": float(op["DELAYED_60M"].mean())
            if len(op) > 0
            else None,
            "delayed_3h_rate": float(op["DELAYED_3H"].mean())
            if len(op) > 0
            else None,
            "mean_arr_delay_min": float(op["ARR_DELAY"].mean())
            if len(op) > 0
            else None,
            "median_arr_delay_min": float(op["ARR_DELAY"].median())
            if len(op) > 0
            else None,
            "p90_arr_delay_min": float(np.percentile(op["ARR_DELAY"].dropna(), 90))
            if len(op) > 0
            else None,
            "share_short_haul": float(
                (group["DISTANCE_TIER"] == "short_haul").mean()
            ),
            "share_medium_haul": float(
                (group["DISTANCE_TIER"] == "medium_haul").mean()
            ),
            "share_long_haul": float(
                (group["DISTANCE_TIER"] == "long_haul").mean()
            ),
            "mean_distance_km": float(group["DISTANCE_KM"].mean()),
        }
    return out


def _yoy_march_block(df: pd.DataFrame) -> dict:
    """March 2023 vs March 2024 — the headline year-over-year comparison."""
    march_2023 = _operated_view(df[df["YEAR_MONTH"] == "2023-03"])
    march_2024 = _operated_view(df[df["YEAR_MONTH"] == "2024-03"])
    if march_2023.empty or march_2024.empty:
        return {}

    def _bucket(g: pd.DataFrame) -> dict:
        return {
            "n_flights": int(len(g)),
            "delayed_15m_rate": float(g["DELAYED_15M"].mean()),
            "delayed_60m_rate": float(g["DELAYED_60M"].mean()),
            "delayed_3h_rate": float(g["DELAYED_3H"].mean()),
            "delayed_3h_count": int(g["DELAYED_3H"].sum()),
            "mean_arr_delay_min": float(g["ARR_DELAY"].mean()),
            "median_arr_delay_min": float(g["ARR_DELAY"].median()),
            "p90_arr_delay_min": float(np.percentile(g["ARR_DELAY"].dropna(), 90)),
            "p95_arr_delay_min": float(np.percentile(g["ARR_DELAY"].dropna(), 95)),
            "share_long_haul": float((g["DISTANCE_TIER"] == "long_haul").mean()),
            "mean_distance_km": float(g["DISTANCE_KM"].mean()),
        }

    a = _bucket(march_2023)
    b = _bucket(march_2024)
    return {
        "march_2023": a,
        "march_2024": b,
        "delta_delayed_15m_rate_pp": (b["delayed_15m_rate"] - a["delayed_15m_rate"]) * 100,
        "delta_delayed_60m_rate_pp": (b["delayed_60m_rate"] - a["delayed_60m_rate"]) * 100,
        "delta_delayed_3h_rate_pp": (b["delayed_3h_rate"] - a["delayed_3h_rate"]) * 100,
        "delta_mean_delay_min": b["mean_arr_delay_min"] - a["mean_arr_delay_min"],
        "delta_p90_delay_min": b["p90_arr_delay_min"] - a["p90_arr_delay_min"],
        "delta_n_flights_pct": (b["n_flights"] - a["n_flights"]) / a["n_flights"] * 100,
    }


def _distance_tier_block(df: pd.DataFrame) -> dict:
    op = _operated_view(df)
    out: dict = {}
    for tier in ["short_haul", "medium_haul", "long_haul"]:
        sub = op[op["DISTANCE_TIER"] == tier]
        if sub.empty:
            continue
        out[tier] = {
            "n": int(len(sub)),
            "share": float(len(sub) / len(op)),
            "delayed_3h_rate": float(sub["DELAYED_3H"].mean()),
            "ec261_payout_eur": float(sub["EC261_PAYOUT_EUR"].iloc[0]),
            "expected_payout_eur": float(
                sub["DELAYED_3H"].mean() * sub["EC261_PAYOUT_EUR"].iloc[0]
            ),
            "mean_distance_km": float(sub["DISTANCE_KM"].mean()),
        }
    return out


def _top_operators_block(df: pd.DataFrame, top_n: int = 15) -> dict:
    op = _operated_view(df)
    g = (
        op.groupby("OP_UNIQUE_CARRIER")
        .agg(
            n=("ARR_DELAY", "size"),
            mean_delay=("ARR_DELAY", "mean"),
            delayed_3h_rate=("DELAYED_3H", "mean"),
        )
        .sort_values("n", ascending=False)
    )
    by_volume = g.head(top_n)
    by_delay = g[g["n"] >= 5_000].sort_values("delayed_3h_rate", ascending=False).head(top_n)
    by_punctuality = g[g["n"] >= 5_000].sort_values("delayed_3h_rate").head(top_n)
    return {
        "by_volume": by_volume.reset_index().to_dict("records"),
        "worst_3h_rate": by_delay.reset_index().to_dict("records"),
        "best_3h_rate": by_punctuality.reset_index().to_dict("records"),
    }


def _top_routes_block(df: pd.DataFrame, top_n: int = 20) -> dict:
    op = _operated_view(df)
    op = op.assign(ROUTE=op["ORIGIN"] + "-" + op["DEST"])
    g = (
        op.groupby("ROUTE")
        .agg(
            n=("ARR_DELAY", "size"),
            mean_delay=("ARR_DELAY", "mean"),
            delayed_3h_rate=("DELAYED_3H", "mean"),
        )
        .sort_values("n", ascending=False)
    )
    return {
        "by_volume": g.head(top_n).reset_index().to_dict("records"),
        "worst_3h_rate_min10k": g[g["n"] >= 10_000]
        .sort_values("delayed_3h_rate", ascending=False)
        .head(top_n)
        .reset_index()
        .to_dict("records"),
    }


def _hour_of_day_block(df: pd.DataFrame) -> dict:
    op = _operated_view(df)
    op = op.assign(HOUR=(op["CRS_DEP_TIME"].fillna(0).astype(int) // 100) % 24)
    g = (
        op.groupby("HOUR")
        .agg(
            n=("ARR_DELAY", "size"),
            mean_delay=("ARR_DELAY", "mean"),
            delayed_3h_rate=("DELAYED_3H", "mean"),
        )
        .sort_index()
    )
    return g.reset_index().to_dict("records")


def _expected_value_block(df: pd.DataFrame) -> dict:
    """Upper-bound EC261 economics, assuming every 180+ min delay is eligible.

    Real eligibility is gated on cause-codes which the R&D Archive does not
    provide. The number reported here is therefore the *ceiling* — a real
    decision rule has to discount by a claim-success rate (default 65%).
    """
    op = _operated_view(df)
    payout = op["EC261_PAYOUT_EUR"].to_numpy()
    delayed = op["DELAYED_3H"].to_numpy()

    expected_payout_per_flight = float((payout * delayed).mean())
    expected_payout_with_alpha = expected_payout_per_flight * EC261.claim_success_rate
    return {
        "claim_success_alpha": EC261.claim_success_rate,
        "expected_payout_per_flight_eur_no_alpha": expected_payout_per_flight,
        "expected_payout_per_flight_eur_with_alpha": expected_payout_with_alpha,
        "n_flights_used": int(len(op)),
        "interpretation": (
            "Upper-bound expected payout per flight if every >=180min arrival "
            "delay were carrier-attributable. Real EC261 eligibility is a "
            "subset of this; the report's section 4.3 quantifies the gap."
        ),
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _make_distribution_plot(df: pd.DataFrame) -> None:
    op = _operated_view(df)
    delays = op["ARR_DELAY"].dropna().to_numpy()
    delays_clip = delays.clip(-60, 360)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].hist(delays_clip, bins=84, color="#0b5394", alpha=0.85, edgecolor="white", linewidth=0.4)
    axes[0].axvline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    axes[0].axvline(EC261_THRESHOLD_MIN, color="#cc0000", linestyle="--", linewidth=1.2,
                    label=f"EC261 trigger ({EC261_THRESHOLD_MIN}min)")
    axes[0].set_xlabel("Arrival delay (min, clipped to [-60, 360])")
    axes[0].set_ylabel("Number of flights")
    axes[0].set_title("Arrival-delay distribution — full EUROCONTROL cache")
    axes[0].legend(loc="upper right")

    months = sorted(op["YEAR_MONTH"].unique())
    rates_15 = [op[op["YEAR_MONTH"] == m]["DELAYED_15M"].mean() * 100 for m in months]
    rates_60 = [op[op["YEAR_MONTH"] == m]["DELAYED_60M"].mean() * 100 for m in months]
    x = np.arange(len(months))
    w = 0.35
    bars1 = axes[1].bar(x - w / 2, rates_15, width=w, color="#0b5394", label=">=15min")
    bars2 = axes[1].bar(x + w / 2, rates_60, width=w, color="#cc0000", label=">=60min")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(months, rotation=20)
    axes[1].set_ylabel("Share of flights (%)")
    axes[1].set_title(f"Tactical-residual delay rates per month (n={len(op):,})")
    axes[1].legend()
    for bar, rate in zip(bars1, rates_15):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{rate:.1f}", ha="center", va="bottom", fontsize=8)
    for bar, rate in zip(bars2, rates_60):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{rate:.1f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "eurocontrol_delay_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _make_yoy_plot(df: pd.DataFrame) -> None:
    a = _operated_view(df[df["YEAR_MONTH"] == "2023-03"])
    b = _operated_view(df[df["YEAR_MONTH"] == "2024-03"])
    if a.empty or b.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    bins = np.linspace(-30, 240, 55)
    axes[0].hist(a["ARR_DELAY"].clip(-60, 360), bins=bins, alpha=0.55, label=f"March 2023 (n={len(a):,})", color="#999999")
    axes[0].hist(b["ARR_DELAY"].clip(-60, 360), bins=bins, alpha=0.55, label=f"March 2024 (n={len(b):,})", color="#0b5394")
    axes[0].axvline(EC261_THRESHOLD_MIN, color="#cc0000", linestyle="--", linewidth=1.0, label="EC261 trigger")
    axes[0].set_xlabel("Arrival delay (min)")
    axes[0].set_ylabel("Number of flights")
    axes[0].set_title("Year-over-year delay distribution (March)")
    axes[0].legend()

    metrics = ["mean (min)", "p90 (min)", "p95 (min)", "% >=15m", "% >=60m"]
    a_vals = [
        a["ARR_DELAY"].mean(),
        np.percentile(a["ARR_DELAY"].dropna(), 90),
        np.percentile(a["ARR_DELAY"].dropna(), 95),
        a["DELAYED_15M"].mean() * 100,
        a["DELAYED_60M"].mean() * 100,
    ]
    b_vals = [
        b["ARR_DELAY"].mean(),
        np.percentile(b["ARR_DELAY"].dropna(), 90),
        np.percentile(b["ARR_DELAY"].dropna(), 95),
        b["DELAYED_15M"].mean() * 100,
        b["DELAYED_60M"].mean() * 100,
    ]
    x = np.arange(len(metrics))
    w = 0.35
    axes[1].bar(x - w / 2, a_vals, width=w, label="March 2023", color="#999999")
    axes[1].bar(x + w / 2, b_vals, width=w, label="March 2024", color="#0b5394")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(metrics)
    axes[1].set_title("YoY headline metrics")
    axes[1].set_ylabel("Minutes / percent")
    axes[1].legend()
    for xi, av, bv in zip(x, a_vals, b_vals):
        axes[1].text(xi - w / 2, av, f"{av:.1f}", ha="center", va="bottom", fontsize=8)
        axes[1].text(xi + w / 2, bv, f"{bv:.1f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "eurocontrol_yoy_march.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _make_seasonality_plot(df: pd.DataFrame) -> None:
    op = _operated_view(df)
    g = (
        op.groupby("YEAR_MONTH")
        .agg(
            mean_delay=("ARR_DELAY", "mean"),
            p90_delay=("ARR_DELAY", lambda x: np.percentile(x.dropna(), 90)),
            rate_15=("DELAYED_15M", "mean"),
            n=("ARR_DELAY", "size"),
        )
        .sort_index()
    )

    fig, ax1 = plt.subplots(figsize=(8, 4.4))
    ax2 = ax1.twinx()

    months = list(g.index)
    ax1.plot(months, g["mean_delay"], "o-", color="#0b5394", label="Mean arrival delay (min)")
    ax1.plot(months, g["p90_delay"], "s--", color="#3d85c6", label="P90 arrival delay (min)")
    ax1.set_ylabel("Arrival delay vs filed plan (min)")
    ax1.set_xlabel("Month")

    ax2.bar(
        months,
        g["rate_15"] * 100,
        alpha=0.18,
        color="#cc0000",
        label="Share delayed >=15min (%)",
    )
    ax2.set_ylabel("Share delayed >=15min (%)")

    lines, labels = ax1.get_legend_handles_labels()
    bars, blabels = ax2.get_legend_handles_labels()
    ax1.legend(lines + bars, labels + blabels, loc="upper left")
    ax1.set_title("EUROCONTROL: monthly tactical-residual delay metrics")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "eurocontrol_seasonality.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Output dump
# ---------------------------------------------------------------------------


def _save_dump(summary: dict) -> None:
    json_path = REPORTS_DIR / "eurocontrol_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, default=_json_default))

    txt_path = REPORTS_DIR / "eurocontrol_summary.txt"
    lines: list[str] = []
    lines.append("EUROCONTROL R&D Archive — analysis dump")
    lines.append("=" * 60)
    lines.append("")

    cov = summary["coverage"]
    lines.append(f"Coverage: {cov['earliest']} – {cov['latest']}")
    lines.append(f"Months  : {', '.join(cov['months_present'])}")
    lines.append(f"Flights : {cov['total_rows']:,} total")
    lines.append(
        f"Vocab   : {cov['n_unique_origins']} ADEPs · "
        f"{cov['n_unique_destinations']} ADESs · "
        f"{cov['n_unique_operators']} operators · "
        f"{cov['n_unique_aircraft_types']} aircraft types"
    )
    lines.append("")

    ovr = summary["overall"]
    lines.append("Overall (note: ARR_DELAY here is actual − latest filed plan,")
    lines.append("not actual − published schedule; ATFM slots are absorbed pre-takeoff)")
    lines.append("-" * 60)
    lines.append(f"  operated flights        : {ovr['n_operated_flights']:,}")
    lines.append(
        f"  cancelled/missing actuals: {ovr['n_cancelled_or_missing_actuals']:,} "
        f"({ovr['cancelled_or_missing_rate'] * 100:.2f}%)"
    )
    lines.append(
        f"  flights delayed >=15m   : {ovr['n_delayed_15m']:,} "
        f"({ovr['delayed_15m_rate'] * 100:.2f}%)  [CODA threshold]"
    )
    lines.append(
        f"  flights delayed >=60m   : {ovr['n_delayed_60m']:,} "
        f"({ovr['delayed_60m_rate'] * 100:.2f}%)"
    )
    lines.append(
        f"  flights delayed >=3h    : {ovr['n_delayed_3h']:,} "
        f"({ovr['delayed_3h_rate'] * 100:.2f}%)  [EC261 trigger]"
    )
    arr = ovr["arr_delay_stats_min"]
    lines.append(
        f"  ARR_DELAY (min)         : mean={arr['mean']:.2f} median={arr['median']:.1f} "
        f"p90={arr['p90']:.1f} p95={arr['p95']:.1f} p99={arr['p99']:.1f}"
    )
    lines.append("")

    lines.append("Per-month")
    lines.append("-" * 60)
    lines.append(
        f"{'YYYY-MM':>8}  {'n_op':>9}  {'mean':>6}  {'med':>5}  {'p90':>5}  "
        f"{'>=15m%':>7}  {'>=60m%':>7}  {'>=3h%':>6}  {'long%':>6}"
    )
    for ym, m in summary["per_month"].items():
        lines.append(
            f"{ym:>8}  {m['n_operated']:>9,}  {m['mean_arr_delay_min']:>6.2f}  "
            f"{m['median_arr_delay_min']:>5.1f}  {m['p90_arr_delay_min']:>5.1f}  "
            f"{m['delayed_15m_rate'] * 100:>6.2f}%  "
            f"{m['delayed_60m_rate'] * 100:>6.2f}%  "
            f"{m['delayed_3h_rate'] * 100:>5.2f}%  {m['share_long_haul'] * 100:>5.1f}%"
        )
    lines.append("")

    if summary["yoy_march"]:
        y = summary["yoy_march"]
        lines.append("Year-over-year — March 2023 vs March 2024")
        lines.append("-" * 60)
        lines.append(
            f"  flights      : {y['march_2023']['n_flights']:>9,} -> "
            f"{y['march_2024']['n_flights']:>9,} ({y['delta_n_flights_pct']:+.2f}%)"
        )
        lines.append(
            f"  mean delay   : {y['march_2023']['mean_arr_delay_min']:>9.2f} -> "
            f"{y['march_2024']['mean_arr_delay_min']:>9.2f} ({y['delta_mean_delay_min']:+.2f} min)"
        )
        lines.append(
            f"  p90 delay    : {y['march_2023']['p90_arr_delay_min']:>9.1f} -> "
            f"{y['march_2024']['p90_arr_delay_min']:>9.1f} ({y['delta_p90_delay_min']:+.1f} min)"
        )
        lines.append(
            f"  >=15m rate   : {y['march_2023']['delayed_15m_rate'] * 100:>9.2f}%"
            f" -> {y['march_2024']['delayed_15m_rate'] * 100:>9.2f}% "
            f"({y['delta_delayed_15m_rate_pp']:+.2f}pp)"
        )
        lines.append(
            f"  >=60m rate   : {y['march_2023']['delayed_60m_rate'] * 100:>9.2f}%"
            f" -> {y['march_2024']['delayed_60m_rate'] * 100:>9.2f}% "
            f"({y['delta_delayed_60m_rate_pp']:+.2f}pp)"
        )
        lines.append(
            f"  >=3h rate    : {y['march_2023']['delayed_3h_rate'] * 100:>9.2f}%"
            f" -> {y['march_2024']['delayed_3h_rate'] * 100:>9.2f}% "
            f"({y['delta_delayed_3h_rate_pp']:+.2f}pp)"
        )
        lines.append("")

    lines.append("Distance tiers (EC261 Article 7)")
    lines.append("-" * 60)
    for tier, t in summary["distance_tiers"].items():
        lines.append(
            f"  {tier:<11}: n={t['n']:>10,} ({t['share'] * 100:>5.2f}%) "
            f"3h-rate={t['delayed_3h_rate'] * 100:>5.2f}% "
            f"payout=€{t['ec261_payout_eur']:>4.0f} "
            f"expected=€{t['expected_payout_eur']:>5.2f}/flight"
        )
    lines.append("")

    lines.append("Top 10 operators by volume")
    lines.append("-" * 60)
    for r in summary["top_operators"]["by_volume"][:10]:
        lines.append(
            f"  {r['OP_UNIQUE_CARRIER']:<6} n={r['n']:>9,}  "
            f"mean_delay={r['mean_delay']:>5.1f}min  "
            f">=3h={r['delayed_3h_rate'] * 100:>5.2f}%"
        )
    lines.append("")

    lines.append("Top 10 routes by volume")
    lines.append("-" * 60)
    for r in summary["top_routes"]["by_volume"][:10]:
        lines.append(
            f"  {r['ROUTE']:<10} n={r['n']:>7,}  "
            f"mean_delay={r['mean_delay']:>5.1f}min  "
            f">=3h={r['delayed_3h_rate'] * 100:>5.2f}%"
        )
    lines.append("")

    ev = summary["expected_value"]
    lines.append("Expected-value ceiling under EC261 (no cause-code filter)")
    lines.append("-" * 60)
    lines.append(
        f"  no-alpha           : €{ev['expected_payout_per_flight_eur_no_alpha']:.4f}/flight"
    )
    lines.append(
        f"  with alpha={ev['claim_success_alpha']}: "
        f"€{ev['expected_payout_per_flight_eur_with_alpha']:.4f}/flight"
    )
    lines.append("")

    txt_path.write_text("\n".join(lines))
    print()
    print("\n".join(lines))


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        if math.isnan(obj):
            return None
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    raise TypeError(f"Cannot JSON-encode {type(obj)}")


if __name__ == "__main__":
    sys.exit(main())
