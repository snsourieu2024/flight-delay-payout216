"""EC261 compensation logic.

Implements the post-September-2025 amended regulation.  Two operations:

1. ``compute_compensation`` — distance-tiered payout per Article 7.
2. ``label_eligible_delay`` — the binary target, gated on both the 3-hour
   arrival-delay trigger AND the carrier-attributable cause requirement
   (i.e. *not* an "extraordinary circumstance" in the meaning of Art. 5(3)).

The labelling decision is the most consequential modelling choice in this
project.  Most students label raw delay; the 2025 amended Annex explicitly
removes weather, ATC, and ANSP-strike delays from the airline's liability,
so a model trained on raw delay would predict events that do not pay out.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import EC261, EC261Params

KM_PER_MILE = 1.609344


def compute_compensation(
    distance_km: np.ndarray | pd.Series | float,
    params: EC261Params = EC261,
) -> np.ndarray:
    """Distance-tiered compensation per EC261 Article 7.

    Parameters
    ----------
    distance_km : array-like
        Great-circle distance in kilometres.  BTS reports DISTANCE in miles
        — convert via ``KM_PER_MILE`` before calling.
    params : EC261Params
        Allows sensitivity analysis without editing this function.

    Returns
    -------
    np.ndarray of float, EUR.
    """
    d = np.asarray(distance_km, dtype=float)
    out = np.full_like(d, params.payout_medium_eur)
    out[d <= params.short_haul_km] = params.payout_short_eur
    out[d > params.medium_haul_km] = params.payout_long_eur
    return out


def label_eligible_delay(
    df: pd.DataFrame,
    params: EC261Params = EC261,
) -> pd.Series:
    """Build the binary EC261-eligible delay label.

    Eligibility rule (post-2025 amendment):
        y = 1 iff
            ARR_DELAY >= 180 minutes
            AND the dominant delay cause is in {CARRIER_DELAY, LATE_AIRCRAFT_DELAY}.

    "Dominant cause" = the cause with the most attributed minutes.  When all
    cause-code columns are zero/null (i.e. flight was on-time or cause not
    reported), eligibility is False by construction.

    BTS rows where CANCELLED == 1 produce ARR_DELAY = NaN and are labelled
    False — under EC261 cancellations are a separate compensation regime
    (Article 5) which we do not model here.  The report's limitations
    section notes this.
    """
    arr_delay = df["ARR_DELAY"].to_numpy(dtype=float)
    long_delay = arr_delay >= params.delay_threshold_min

    cause_cols = list(params.eligible_causes) + list(params.exempt_causes)
    causes = df[cause_cols].fillna(0).to_numpy(dtype=float)

    dominant_idx = np.argmax(causes, axis=1)
    has_any_cause = causes.sum(axis=1) > 0
    eligible_indices = set(range(len(params.eligible_causes)))
    is_eligible_cause = np.isin(dominant_idx, list(eligible_indices)) & has_any_cause

    label = (long_delay & is_eligible_cause).astype(int)
    label[np.isnan(arr_delay)] = 0
    return pd.Series(label, index=df.index, name="y_eligible_delay")


def compensation_eur(df: pd.DataFrame, params: EC261Params = EC261) -> pd.Series:
    """Per-flight EC261 compensation amount in EUR.

    Distance is converted from miles (BTS) to km on the fly.  This is a
    *potential* payout — it is multiplied by the empirical claim-success
    rate ``params.claim_success_rate`` inside the profit metric.
    """
    distance_km = df["DISTANCE"].to_numpy(dtype=float) * KM_PER_MILE
    return pd.Series(
        compute_compensation(distance_km, params=params),
        index=df.index,
        name="C_eur",
    )
