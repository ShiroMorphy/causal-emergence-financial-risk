"""
Out-of-Sample Pseudo-Real-Time Forecasting Engine
=================================================
Implements expanding-window OOS evaluation, Diebold-Mariano (1995) tests with
Harvey-Leybourne-Newbold small sample correction, and Clark-West (2007) tests for nested models.
"""

from typing import Dict, Tuple
import math
import numpy as np


def compute_diebold_mariano_test(
    e1: np.ndarray,
    e2: np.ndarray,
    h: int = 1
) -> Tuple[float, float]:
    """
    Computes Diebold-Mariano (1995) test statistic for equal predictive accuracy:
        d_t = e1_t^2 - e2_t^2
        DM = bar{d} / sqrt( V_HAC(bar{d}) )

    With Harvey, Leybourne, and Newbold (1997) finite-sample correction.

    Returns
    -------
    dm_stat : float
    p_value : float
    """
    T = len(e1)
    d = e1 ** 2 - e2 ** 2
    d_bar = np.mean(d)

    # Autocovariance up to lag h-1
    gamma_0 = np.var(d, ddof=0)
    gamma_sum = 0.0
    for lag in range(1, h):
        gamma_l = np.mean((d[lag:] - d_bar) * (d[:-lag] - d_bar))
        gamma_sum += 2.0 * gamma_l

    var_d = (gamma_0 + gamma_sum) / T
    if var_d <= 0:
        return 0.0, 1.0

    dm_stat = float(d_bar / np.sqrt(var_d))

    # HLN small sample adjustment factor
    hln_factor = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    dm_stat_adj = dm_stat * hln_factor

    # Two-sided p-value
    p_val = 1.0 - math.erf(abs(dm_stat_adj) / math.sqrt(2.0))
    return dm_stat_adj, p_val


def compute_clark_west_test(
    y_actual: np.ndarray,
    y_pred_nested: np.ndarray,
    y_pred_extended: np.ndarray
) -> Tuple[float, float]:
    """
    Computes Clark-West (2007) adjusted MSPE test for nested model comparisons:
        f_t = (y_t - y_hat_nested_t)^2 - [ (y_t - y_hat_extended_t)^2 - (y_hat_nested_t - y_hat_extended_t)^2 ]
        CW = bar{f} / sqrt( Var(bar{f}) )

    One-sided test: H0: MSPE_nested <= MSPE_extended vs H1: MSPE_extended < MSPE_nested

    Returns
    -------
    cw_stat : float
    p_value : float
    """
    T = len(y_actual)
    e_nested = y_actual - y_pred_nested
    e_extended = y_actual - y_pred_extended
    adj_term = (y_pred_nested - y_pred_extended) ** 2

    # f_t sequence
    f = e_nested ** 2 - (e_extended ** 2 - adj_term)
    f_bar = np.mean(f)
    var_f = np.var(f, ddof=1) / T

    if var_f <= 0:
        return 0.0, 0.5

    cw_stat = float(f_bar / np.sqrt(var_f))
    # One-sided standard normal p-value
    p_val = 0.5 * (1.0 - math.erf(cw_stat / math.sqrt(2.0)))
    return cw_stat, p_val


def run_expanding_window_oos(
    y: np.ndarray,
    X_base: np.ndarray,
    X_ext: np.ndarray,
    initial_train_size: int
) -> Dict:
    """
    Runs an expanding window pseudo-out-of-sample forecast comparison.
    """
    T = len(y)
    n_oos = T - initial_train_size
    preds_base = np.zeros(n_oos)
    preds_ext = np.zeros(n_oos)
    y_actual = y[initial_train_size:]

    for i, t in enumerate(range(initial_train_size, T)):
        # Fit models on expanding window [0 : t]
        y_train = y[:t]

        # Base model OLS
        X_b_train = X_base[:t]
        beta_base = np.linalg.lstsq(X_b_train, y_train, rcond=None)[0]
        preds_base[i] = X_base[t] @ beta_base

        # Extended model OLS
        X_e_train = X_ext[:t]
        beta_ext = np.linalg.lstsq(X_e_train, y_train, rcond=None)[0]
        preds_ext[i] = X_ext[t] @ beta_ext

    # Evaluation metrics
    e_base = y_actual - preds_base
    e_ext = y_actual - preds_ext

    rmse_base = float(np.sqrt(np.mean(e_base ** 2)))
    rmse_ext = float(np.sqrt(np.mean(e_ext ** 2)))
    mae_base = float(np.mean(np.abs(e_base)))
    mae_ext = float(np.mean(np.abs(e_ext)))

    dm_stat, dm_pval = compute_diebold_mariano_test(e_base, e_ext)
    cw_stat, cw_pval = compute_clark_west_test(y_actual, preds_base, preds_ext)

    return {
        "rmse_base": rmse_base,
        "rmse_ext": rmse_ext,
        "rmse_ratio": rmse_ext / (rmse_base + 1e-12),
        "mae_base": mae_base,
        "mae_ext": mae_ext,
        "dm_stat": dm_stat,
        "dm_pvalue": dm_pval,
        "cw_stat": cw_stat,
        "cw_pvalue": cw_pval,
        "n_oos_obs": n_oos
    }
