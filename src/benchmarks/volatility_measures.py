"""
Volatility and Correlation Benchmarks
=====================================
Calculates realized volatility, average pairwise cross-sectional correlation,
and PCA first component explained variance ratio.
"""

import numpy as np
import pandas as pd


def compute_realized_volatility(returns_series: np.ndarray, annualized: bool = True) -> float:
    """
    Computes annualized realized volatility: RV = sqrt(252 * sum(r_t^2) / T)
    """
    T = len(returns_series)
    sum_sq = np.sum(returns_series ** 2)
    rv = np.sqrt(sum_sq / T)
    if annualized:
        rv *= np.sqrt(252.0)
    return float(rv)


def compute_average_correlation(returns_matrix: np.ndarray) -> float:
    """
    Calculates the average off-diagonal pairwise Pearson correlation coefficient:
        bar{rho} = 2 / (p*(p-1)) * sum_{i < j} rho_{ij}
    """
    corr_matrix = np.corrcoef(returns_matrix, rowvar=False)
    p = corr_matrix.shape[0]
    # Extract upper triangle indices
    triu_indices = np.triu_indices(p, k=1)
    avg_corr = np.mean(corr_matrix[triu_indices])
    return float(avg_corr)


def compute_first_pc_variance_ratio(returns_matrix: np.ndarray) -> float:
    """
    Calculates the proportion of total variance explained by the first principal component:
        lambda_1 / sum_{i=1}^p lambda_i
    """
    cov_matrix = np.cov(returns_matrix, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov_matrix)
    total_var = np.sum(eigvals)
    if total_var <= 0:
        return 0.0
    return float(np.max(eigvals) / total_var)
