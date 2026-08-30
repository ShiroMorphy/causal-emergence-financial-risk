"""
Tests for Econometrics and Forecast Evaluation Engine
=====================================================
"""

import numpy as np
import pandas as pd
from econometrics.predictive_regressions import run_predictive_regression_hac
from econometrics.early_warning_logit import (
    evaluate_early_warning_classifier,
    compute_roc_auc,
    compute_pr_auc,
    evaluate_oos_expanding_logit
)
from econometrics.oos_forecasting import compute_diebold_mariano_test, compute_clark_west_test


def test_predictive_regression_hac_recovery():
    """Verify that OLS + HAC recovers known linear relationships."""
    np.random.seed(123)
    T = 200
    x = np.random.randn(T, 1)
    X = np.column_stack([np.ones((T, 1)), x])
    true_beta = np.array([1.5, 2.5])
    eps = np.random.randn(T) * 0.5
    y = X @ true_beta + eps

    res = run_predictive_regression_hac(y, X, feature_names=["const", "x"])
    assert np.isclose(res["params"]["const"], 1.5, atol=0.2)
    assert np.isclose(res["params"]["x"], 2.5, atol=0.2)
    assert res["r2"] > 0.8
    assert res["pvalues"]["x"] < 0.001


def test_roc_auc_perfect_separation():
    """Verify AUC is 1.0 under perfect separation."""
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_score = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    auc = compute_roc_auc(y_true, y_score)
    assert auc == 1.0


def test_pr_auc_calculation():
    """Verify PR-AUC calculation is bounded in [0, 1]."""
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.2, 0.4, 0.6, 0.8])
    pr_auc = compute_pr_auc(y_true, y_score)
    assert 0.0 <= pr_auc <= 1.0


def test_expanding_window_logit_oos():
    """Verify expanding window OOS logit produces valid probabilistic forecasts."""
    np.random.seed(42)
    T = 300
    X = np.column_stack([np.ones(T), np.random.randn(T, 2)])
    y = (X[:, 1] + 0.5 * X[:, 2] + np.random.randn(T) > 0).astype(int)

    oos_res = evaluate_oos_expanding_logit(y, X, initial_train_size=100, feature_names=["c", "x1", "x2"])
    assert oos_res["auc_roc_oos"] > 0.6
    assert 0.0 <= oos_res["brier_score_oos"] <= 1.0
    assert len(oos_res["p_hat_oos"]) == 200


def test_diebold_mariano_equal_forecasts():
    """Verify DM test does not reject the null of equal predictive ability when forecasts are identical."""
    np.random.seed(42)
    e1 = np.random.randn(500)
    e2 = e1 + 1e-6 * np.random.randn(500)
    dm_stat, p_val = compute_diebold_mariano_test(e1, e2)
    assert p_val > 0.05
