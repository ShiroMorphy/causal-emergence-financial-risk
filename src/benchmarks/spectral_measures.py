"""
Spectral Dimensionality and Entropy Benchmarks
==============================================
Calculates the Effective Rank of Roy & Vetterli (2007) and spectral von Neumann entropy.
"""

import numpy as np


def compute_effective_rank(returns_matrix: np.ndarray, eps: float = 1e-12) -> float:
    """
    Computes the Effective Rank (Roy & Vetterli, 2007) of the data covariance matrix:
        EffRank = exp( H(p_1, ..., p_p) )
        where p_i = sigma_i / sum(sigma_k) are the normalized singular values.

    Represents the continuous effective number of active dimensions in the correlation structure.
    """
    # Singular value decomposition
    _, s, _ = np.linalg.svd(returns_matrix - np.mean(returns_matrix, axis=0, keepdims=True), full_matrices=False)
    s_sum = np.sum(s)
    if s_sum <= 0:
        return 1.0

    p_norm = s / s_sum
    # Shannon entropy of normalized singular values
    entropy = -np.sum(p_norm * np.log(p_norm + eps))
    return float(np.exp(entropy))


def compute_spectral_entropy(returns_matrix: np.ndarray, eps: float = 1e-12) -> float:
    """
    Computes normalized Spectral Entropy of the correlation matrix:
        H_spec = - 1 / ln(p) * sum_{i=1}^p lambda_i_norm * ln(lambda_i_norm)
    """
    corr = np.corrcoef(returns_matrix, rowvar=False)
    p = corr.shape[0]
    eigvals = np.linalg.eigvalsh(corr)
    eigvals = np.clip(eigvals, a_min=eps, a_max=None)
    eig_norm = eigvals / np.sum(eigvals)

    entropy = -np.sum(eig_norm * np.log(eig_norm))
    norm_entropy = entropy / np.log(p)
    return float(norm_entropy)
