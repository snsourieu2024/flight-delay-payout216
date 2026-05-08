"""SHAP explainer helpers.

Wraps SHAP's TreeExplainer / KernelExplainer behind a single
``explain_pipeline`` function so the notebook never has to know which
flavour applies to which estimator.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def _is_tree_model(clf) -> bool:
    name = type(clf).__name__
    return name in {
        "RandomForestClassifier",
        "DecisionTreeClassifier",
        "XGBClassifier",
        "GradientBoostingClassifier",
        "LGBMClassifier",
    }


def explain_pipeline(
    pipeline: Pipeline,
    X_raw: pd.DataFrame,
    sample_size: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """Compute SHAP values on a sample of raw rows.

    Returns a dict with keys: ``shap_values``, ``feature_names``, ``X_transformed``.
    Caller is responsible for plotting (notebook handles that with shap.summary_plot).
    """
    try:
        import shap
    except ImportError as e:
        raise ImportError("Install shap: `pip install shap`") from e

    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(len(X_raw), size=min(sample_size, len(X_raw)), replace=False)
    X_sample = X_raw.iloc[sample_idx].copy()

    pre = Pipeline(steps=pipeline.steps[:-1])
    clf = pipeline.steps[-1][1]

    X_trans = pre.transform(X_sample)
    feature_names = _resolve_feature_names(pre, X_trans)

    if _is_tree_model(clf):
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_trans)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
    else:
        background = X_trans[: min(100, X_trans.shape[0])] if hasattr(X_trans, "__getitem__") else X_trans
        explainer = shap.KernelExplainer(
            lambda Z: clf.predict_proba(Z)[:, 1], background
        )
        shap_values = explainer.shap_values(X_trans, nsamples=100)

    return {
        "shap_values": shap_values,
        "feature_names": feature_names,
        "X_transformed": X_trans,
        "X_sample": X_sample,
    }


def _resolve_feature_names(pre: Pipeline, X_trans) -> list[str]:
    try:
        return list(pre.get_feature_names_out())
    except Exception:
        n = X_trans.shape[1] if hasattr(X_trans, "shape") else None
        return [f"f{i}" for i in range(n)] if n is not None else []
