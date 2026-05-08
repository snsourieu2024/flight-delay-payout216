"""End-to-end pipeline test on a small synthetic sample."""
from __future__ import annotations

import warnings

import numpy as np

from src.data.ec261 import label_eligible_delay
from src.data.loaders import add_ticket_price, load_bts, prepare_modelling_frame
from src.models.registry import make_logistic_regression
from src.pipeline.build import build_pipeline
from src.pipeline.splits import temporal_split


def test_pipeline_end_to_end_runs_and_predicts():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    df = load_bts(force_synthetic=True, n_synthetic=10_000)
    df = prepare_modelling_frame(df)
    df = add_ticket_price(df)
    y = label_eligible_delay(df).to_numpy()

    split = temporal_split(df)
    X_tr = df.iloc[split.train_idx].reset_index(drop=True)
    X_te = df.iloc[split.test_idx].reset_index(drop=True)
    y_tr = y[split.train_idx]

    pipe = build_pipeline(make_logistic_regression())
    pipe.fit(X_tr, y_tr)
    proba = pipe.predict_proba(X_te)[:, 1]
    assert proba.shape == (len(X_te),)
    assert np.all((proba >= 0) & (proba <= 1))
