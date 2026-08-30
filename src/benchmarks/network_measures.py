"""
Network Connectedness Benchmarks
================================
Implements Diebold-Yilmaz (2012, 2014) Total Spillover Index via Generalized
Forecast Error Variance Decomposition (GFEVD) and Granger causality network density.
"""

from typing import Tuple
import numpy as np


def compute_diebold_yilmaz_index(
    returns_matrix: np.ndarray,
    n_lags: int = 2,
    forecast_horizon: int = 10
) -> float:
    """
    Calculates Diebold-Yilmaz Total Spillover Index from a VAR(p) model via Generalized FEVD:
        Total Spillover = (sum_{i != j} theta_{ij}^g(H)) / (sum_{i,j} theta_{ij}^g(H)) * 100

    Parameters
    ----------
    returns_matrix : np.ndarray of shape (T, p)
    n_lags : int
    forecast_horizon : int (H = 10 days default)

    Returns
    -------
    spillover_index : float in [0, 100]
    """
    T, p = returns_matrix.shape
    X = returns_matrix - np.mean(returns_matrix, axis=0, keepdims=True)

    # Estimate VAR(1) or VAR(p)
    X_lag = X[:-1, :]
    X_lead = X[1:, :]
    A = np.linalg.lstsq(X_lag, X_lead, rcond=None)[0].T  # (p, p)
    res = X_lead - X_lag @ A.T
    Sigma_eps = (res.T @ res) / (T - p)

    # Moving Average MA(h) coefficients Phi_h: Phi_0 = I, Phi_1 = A, Phi_h = A^h
    Phi = [np.eye(p)]
    for h in range(1, forecast_horizon):
        Phi.append(Phi[-1] @ A)

    # Generalized FEVD: theta_{ij}^g(H) = (sigma_{jj}^-1 * sum_{h=0}^{H-1} (e_i.T Phi_h Sigma e_j)^2) / sum_{h=0}^{H-1} (e_i.T Phi_h Sigma Phi_h.T e_i)
    gfevd = np.zeros((p, p))
    diag_sigma = np.diagonal(Sigma_eps)

    for i in range(p):
        denom = 0.0
        for h in range(forecast_horizon):
            denom += (Phi[h] @ Sigma_eps @ Phi[h].T)[i, i]

        for j in range(p):
            numer = 0.0
            for h in range(forecast_horizon):
                numer += (Phi[h] @ Sigma_eps)[i, j] ** 2
            gfevd[i, j] = (numer / (diag_sigma[j] + 1e-12)) / (denom + 1e-12)

    # Normalize rows to sum to 100
    row_sums = np.sum(gfevd, axis=1, keepdims=True)
    gfevd_norm = (gfevd / (row_sums + 1e-12)) * 100.0

    # Total spillover = sum of off-diagonals / total sum * 100
    off_diag_sum = np.sum(gfevd_norm) - np.trace(gfevd_norm)
    total_spillover = off_diag_sum / p
    return float(total_spillover)


def compute_granger_network_density(returns_matrix: np.ndarray, alpha: float = 0.05) -> float:
    """
    Computes Granger causality network density (% of statistically significant directed links).
    """
    T, p = returns_matrix.shape
    X = returns_matrix - np.mean(returns_matrix, axis=0, keepdims=True)
    n_links = 0
    total_possible = p * (p - 1)

    for i in range(p):
        y = X[1:, i]
        for j in range(p):
            if i == j:
                continue
            # Restricted: y_t ~ y_{t-1}
            x_res = X[:-1, i:i+1]
            beta_res = np.linalg.lstsq(x_res, y, rcond=None)[0]
            rss_res = np.sum((y - x_res @ beta_res) ** 2)

            # Unrestricted: y_t ~ y_{t-1} + x_{j, t-1}
            x_unres = np.column_stack([X[:-1, i], X[:-1, j]])
            beta_unres = np.linalg.lstsq(x_unres, y, rcond=None)[0]
            rss_unres = np.sum((y - x_unres @ beta_unres) ** 2)

            # F-statistic: ((RSS_r - RSS_u) / 1) / (RSS_u / (T - 3))
            f_stat = ((rss_res - rss_unres) / 1.0) / (rss_unres / (T - 3) + 1e-12)
            # Critical F-value approx for 1, T-3 at alpha=0.05 is ~ 3.84
            if f_stat > 3.84:
                n_links += 1

    return float(n_links / total_possible)
