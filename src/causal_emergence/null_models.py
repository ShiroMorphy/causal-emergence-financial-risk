"""
Hierarchical Null Models for Rigorous Causal Emergence Inference
===============================================================
Implements 4 matched null generators:
1. H_0^{circ}: Circular time-shift null (preserves marginal dynamics, destroys cross-correlations).
2. H_0^{diag}: Diagonal VAR(1) null (independent AR(1) processes, zero network coupling).
3. H_0^{static}: Synchronous time-permutation null (preserves exact instantaneous cross-sectional
   covariance Sigma_x, destroys intertemporal causality x_t -> x_{t+1}).
4. H_0^{diag+contemp}: Diagonal VAR(1) with FULL contemporaneous noise covariance Sigma_eps
   (preserves own-lag persistence AND contemporaneous cross-correlations, isolates pure cross-lag causal network).
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from .micro_var import fit_micro_var1
from .analytical_ei import compute_continuous_ei, compute_emergence_spectrum
from .stiefel_optimizer import optimize_coarse_graining_stiefel


def generate_circular_null_data(X: np.ndarray) -> np.ndarray:
    """
    H_0^{circ}: Circularly shifts each column independently by a random lag.
    Preserves marginal autocorrelation and variance, destroys cross-correlations and cross-lags.
    """
    T, p = X.shape
    X_null = np.zeros_like(X)
    for col in range(p):
        shift = np.random.randint(10, T - 10)
        X_null[:, col] = np.roll(X[:, col], shift)
    return X_null


def generate_diagonal_var_null_data(X: np.ndarray, A: np.ndarray, Sigma_eps: np.ndarray) -> np.ndarray:
    """
    H_0^{diag}: Simulates an uncoupled diagonal VAR(1) with matching diagonal AR(1) coefficients
    and diagonal innovation variances. Zero cross-asset coupling.
    """
    T, p = X.shape
    diag_A = np.diag(np.diag(A))
    diag_S_eps = np.diag(np.diag(Sigma_eps))
    
    X_null = np.zeros((T, p))
    std_eps = np.sqrt(np.clip(np.diag(diag_S_eps), 1e-12, None))
    innovations = np.random.randn(T, p) * std_eps
    
    x_curr = np.random.randn(p) * std_eps
    for _ in range(50):
        x_curr = diag_A @ x_curr + np.random.randn(p) * std_eps
        
    for t in range(T):
        x_curr = diag_A @ x_curr + innovations[t]
        X_null[t] = x_curr
        
    return X_null


def generate_static_correlation_null_data(X: np.ndarray) -> np.ndarray:
    """
    H_0^{static}: Synchronously permutes time steps across all assets.
    Preserves EXACT contemporaneous covariance Sigma_x, but destroys intertemporal causality x_t -> x_{t+1}.
    """
    T, p = X.shape
    perm = np.random.permutation(T)
    return X[perm, :]


def generate_diag_plus_contemp_null_data(X: np.ndarray, A: np.ndarray, Sigma_eps: np.ndarray) -> np.ndarray:
    """
    H_0^{diag+contemp}: Simulates a VAR(1) with diagonal transition matrix D = diag(A)
    and FULL contemporaneous innovation covariance matrix Sigma_eps.
    Preserves:
    1. Own-lag autocorrelation (persistence) of each individual asset.
    2. Exact contemporaneous cross-asset shock covariance structure.
    Destroys ONLY:
    Off-diagonal cross-lag dynamical coupling (A_ij = 0 for i != j).
    """
    T, p = X.shape
    diag_A = np.diag(np.diag(A))
    
    # Generate correlated innovations from Sigma_eps
    Sigma_clean = 0.5 * (Sigma_eps + Sigma_eps.T) + 1e-10 * np.trace(Sigma_eps)/float(p) * np.eye(p)
    L_eps = np.linalg.cholesky(Sigma_clean)
    innovations = np.random.randn(T, p) @ L_eps.T
    
    X_null = np.zeros((T, p))
    x_curr = np.random.randn(p) @ L_eps.T
    # Warmup
    for _ in range(50):
        x_curr = diag_A @ x_curr + np.random.randn(p) @ L_eps.T
        
    for t in range(T):
        x_curr = diag_A @ x_curr + innovations[t]
        X_null[t] = x_curr
        
    return X_null


def evaluate_single_null_realization(
    X_null: np.ndarray,
    q_candidates: List[int],
    kappa_do: float = 1.0,
    n_restarts: int = 2,
    max_iter: int = 30
) -> Tuple[float, int]:
    """
    Runs the exact same estimation and Stiefel optimization pipeline on surrogate data.
    """
    W_len, p = X_null.shape
    A_n, S_eps_n = fit_micro_var1(X_null)
    S_x_n = np.cov(X_null, rowvar=False)

    ei_m_n = compute_continuous_ei(A_n, S_eps_n, Sigma_x=S_x_n, kappa_do=kappa_do)
    macro_eis_n = {}
    for q in q_candidates:
        _, ei_q_n = optimize_coarse_graining_stiefel(
            A_n, S_eps_n, q=q, Sigma_x=S_x_n, kappa_do=kappa_do,
            n_restarts=n_restarts, max_iter=max_iter
        )
        macro_eis_n[q] = ei_q_n

    cefi_n, q_star_n, _, _ = compute_emergence_spectrum(ei_m_n, macro_eis_n, p_micro=p)
    return cefi_n, q_star_n


def run_null_ensemble(
    X_window: np.ndarray,
    null_type: str,
    n_sims: int = 100,
    q_candidates: Optional[List[int]] = None,
    kappa_do: float = 1.0,
    n_restarts: int = 2,
    max_iter: int = 30
) -> Dict:
    """
    Computes empirical null distribution F_0(CEFI) under the specified null model.
    """
    T, p = X_window.shape
    if q_candidates is None:
        q_candidates = list(range(1, p))

    A_obs, S_eps_obs = fit_micro_var1(X_window)

    null_cefis = []
    null_q_stars = []

    for _ in range(n_sims):
        if null_type == "circular":
            X_surr = generate_circular_null_data(X_window)
        elif null_type == "diagonal":
            X_surr = generate_diagonal_var_null_data(X_window, A_obs, S_eps_obs)
        elif null_type == "static":
            X_surr = generate_static_correlation_null_data(X_window)
        elif null_type == "diag_contemp":
            X_surr = generate_diag_plus_contemp_null_data(X_window, A_obs, S_eps_obs)
        else:
            raise ValueError(f"Unknown null_type: {null_type}")

        cefi_b, q_b = evaluate_single_null_realization(
            X_surr, q_candidates=q_candidates, kappa_do=kappa_do,
            n_restarts=n_restarts, max_iter=max_iter
        )
        null_cefis.append(cefi_b)
        null_q_stars.append(q_b)

    null_cefis = np.array(null_cefis)
    null_q_stars = np.array(null_q_stars)

    return {
        "null_type": null_type,
        "n_sims": n_sims,
        "mean_cefi": float(np.mean(null_cefis)),
        "std_cefi": float(np.std(null_cefis)),
        "q95_cefi": float(np.percentile(null_cefis, 95)),
        "q99_cefi": float(np.percentile(null_cefis, 99)),
        "null_cefis": null_cefis,
        "null_q_stars": null_q_stars
    }
