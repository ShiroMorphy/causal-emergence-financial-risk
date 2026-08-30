"""
Econometric Modeling and Forecasting Package
============================================
In-sample predictive regressions, Logit early-warning models, Out-of-sample evaluations,
and block bootstrap inference.
"""

from .predictive_regressions import run_predictive_regression_hac
from .early_warning_logit import evaluate_early_warning_classifier
from .oos_forecasting import run_expanding_window_oos, compute_clark_west_test, compute_diebold_mariano_test
from .bootstrap_inference import stationary_block_bootstrap_cefi

__all__ = [
    "run_predictive_regression_hac",
    "evaluate_early_warning_classifier",
    "run_expanding_window_oos",
    "compute_clark_west_test",
    "compute_diebold_mariano_test",
    "stationary_block_bootstrap_cefi"
]
