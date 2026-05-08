"""Hyperparameter tuning configurations.

Three search strategies are provided, matched to model complexity and
computational budget:

- ``logreg_grid``      : exhaustive small grid (fast).
- ``rf_random``        : RandomizedSearchCV, 30 iterations (variance-reducing).
- ``xgb_bayes``        : BayesSearchCV, 50 iterations (sample-efficient).

Each returns a tuple of ``(param_grid_or_space, search_kwargs)`` so the
notebooks can pick the right CV class.

Search budget rationale (cited in the report):
    Grid LR:   2 × 3 × 2  = 12 fits
    Random RF: 30 fits
    Bayes XGB: 50 fits
    Total:     ~92 fits × 4 CV folds = 368 fits ≈ 6h on a laptop with hist tree method.
"""
from __future__ import annotations

from typing import Any

LOGREG_GRID: dict[str, list[Any]] = {
    "clf__C": [0.1, 1.0, 10.0],
    "clf__penalty": ["l2"],
    "clf__solver": ["lbfgs", "saga"],
}

RF_RANDOM_DIST: dict[str, Any] = {
    "clf__n_estimators": [200, 300, 500, 800],
    "clf__max_depth": [None, 8, 16, 24],
    "clf__min_samples_leaf": [10, 25, 50, 100, 200],
    "clf__max_features": ["sqrt", "log2", 0.5],
}

XGB_BAYES_SPACE = {
    "clf__n_estimators": (100, 1000),
    "clf__max_depth": (3, 10),
    "clf__learning_rate": (1e-3, 3e-1, "log-uniform"),
    "clf__subsample": (0.5, 1.0),
    "clf__colsample_bytree": (0.5, 1.0),
    "clf__reg_lambda": (1e-3, 10.0, "log-uniform"),
    "clf__min_child_weight": (1, 50),
}


def n_iter_xgb(default: int = 50) -> int:
    """Return the Bayesian search budget for XGBoost.

    Adjustable via env var ``XGB_BAYES_N_ITER`` for CI / quick experiments.
    """
    import os

    return int(os.environ.get("XGB_BAYES_N_ITER", default))


def n_iter_rf(default: int = 30) -> int:
    import os

    return int(os.environ.get("RF_RANDOM_N_ITER", default))
