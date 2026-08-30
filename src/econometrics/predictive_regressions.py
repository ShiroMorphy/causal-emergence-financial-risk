"""
Predictive In-Sample Regressions with Newey-West HAC Errors
==========================================================
Estimates linear models: y_{t+h} = alpha + beta * CEFI_t + gamma' Z_t + eps_{t+h}
with Heteroskedasticity and Autocorrelation Consistent (HAC) standard errors.
"""

from typing import Dict, List, Optional
import numpy as np



def compute_newey_west_covariance(X: np.ndarray, residuals: np.ndarray, max_lags: int) -> np.ndarray:
    """
    Computes Newey-West (1987) HAC covariance matrix for OLS coefficients:
        V_HAC = (X'X)^-1 ( Gamma_0 + sum_{l=1}^L w_l (Gamma_l + Gamma_l') ) (X'X)^-1
        where w_l = 1 - l / (L + 1)
    """
    T, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)

    # u_t * x_t (Shape: T, k)
    u_X = residuals[:, np.newaxis] * X

    # Gamma_0
    omega = u_X.T @ u_X

    # Lagged autocovariances
    for lag in range(1, max_lags + 1):
        weight = 1.0 - (lag / (max_lags + 1))
        gamma_l = u_X[lag:].T @ u_X[:-lag]
        omega += weight * (gamma_l + gamma_l.T)

    v_hac = XtX_inv @ omega @ XtX_inv
    return v_hac


def run_predictive_regression_hac(
    y: np.ndarray,
    X: np.ndarray,
    feature_names: Optional[List[str]] = None,
    max_lags: Optional[int] = None
) -> Dict:
    """
    Fits OLS regression y = X beta + eps and returns HAC standard errors, t-stats, and p-values.

    Parameters
    ----------
    y : np.ndarray of shape (T,)
    X : np.ndarray of shape (T, k) (should include intercept column of ones)
    feature_names : Optional[List[str]]
    max_lags : Optional[int] (Default: int(4 * (T / 100)^(2/9)))

    Returns
    -------
    results : Dict containing 'params', 'bse_hac', 'tvalues', 'pvalues', 'r2', 'r2_adj'
    """
    T, k = X.shape
    if max_lags is None:
        max_lags = int(np.ceil(4.0 * (T / 100.0) ** (2.0 / 9.0)))

    # OLS parameter estimates
    params = np.linalg.lstsq(X, y, rcond=None)[0]
    y_pred = X @ params
    residuals = y - y_pred

    # Standard errors
    v_hac = compute_newey_west_covariance(X, residuals, max_lags)
    bse_hac = np.sqrt(np.clip(np.diagonal(v_hac), a_min=1e-12, a_max=None))

    t_values = params / bse_hac
    # Two-sided standard normal p-values using math.erf
    import math
    p_values = [1.0 - math.erf(abs(float(t)) / math.sqrt(2.0)) for t in t_values]

    # Goodness of fit
    tss = np.sum((y - np.mean(y)) ** 2)
    rss = np.sum(residuals ** 2)
    r2 = float(1.0 - rss / tss) if tss > 0 else 0.0
    r2_adj = float(1.0 - (1.0 - r2) * (T - 1) / (T - k)) if (T - k) > 0 else r2


    if feature_names is None:
        feature_names = [f"X_{i}" for i in range(k)]

    return {
        "params": dict(zip(feature_names, params)),
        "bse_hac": dict(zip(feature_names, bse_hac)),
        "tvalues": dict(zip(feature_names, t_values)),
        "pvalues": dict(zip(feature_names, p_values)),
        "r2": r2,
        "r2_adj": r2_adj,
        "nobs": T
    }
