"""Deterministic, stratified modelling-sample construction.

The full real 2024 BTS frame is ~7 M rows — infeasible to tune on a laptop
within the project's time budget. The modelling pipeline therefore consumes a
fixed-size, reproducible sample. This module is the single source of truth for
how that sample is built (imported by notebook 00 and by
``scripts/make_sample.py``) so the documented run path is self-contained and
needs no hidden manual step.

The sample is stratified jointly on calendar month and the EC261-eligible
label so that (a) all twelve months stay represented in time order and
(b) the ~1.2 % positive base rate is preserved — the class imbalance is the
scientific point and must survive sampling.
"""
from __future__ import annotations

import pandas as pd

from ..config import RANDOM_SEED

DEFAULT_SAMPLE_N = 150_000
LABEL_COL = "y_eligible_delay"


def stratified_modelling_sample(
    df: pd.DataFrame,
    n: int = DEFAULT_SAMPLE_N,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Return a seeded, month×label-stratified sample of ``df``.

    Parameters
    ----------
    df : DataFrame
        Full modelling frame. Must contain ``FL_DATE``; uses
        ``y_eligible_delay`` for label stratification when present.
    n : int
        Target row count. If ``len(df) <= n`` the frame is returned
        unchanged (no upsampling).
    seed : int
        RNG seed for reproducibility (defaults to the project seed).

    Returns
    -------
    DataFrame with the original columns, row-subsampled, index reset.
    """
    if len(df) <= n:
        return df.reset_index(drop=True)

    month = pd.to_datetime(df["FL_DATE"]).dt.to_period("M").astype(str)
    label = df[LABEL_COL] if LABEL_COL in df.columns else pd.Series(0, index=df.index)
    frac = n / len(df)

    sample = (
        df.groupby([month.rename("_m"), label.rename("_y")], group_keys=False, observed=True)
        .sample(frac=frac, random_state=seed)
        .reset_index(drop=True)
    )
    return sample
