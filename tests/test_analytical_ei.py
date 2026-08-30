"""
Unit Tests for Scale-Invariant Analytical Continuous Causal Emergence Engine
===========================================================================
Verifies exact scale-invariance, Stiefel manifold orthogonality, exact conditional
Markov dynamics projection, emergence metrics, and published benchmarks (Liu et al. 2024, PRE 2025 SVD).
"""

import numpy as np
from causal_emergence.analytical_ei import (
    compute_continuous_ei,
    compute_macro_dynamics,
    compute_macro_ei,
    compute_emergence_spectrum
)
from causal_emergence.stiefel_optimizer import (
    initialize_stiefel_matrix,
    optimize_coarse_graining_stiefel
)
from causal_emergence.micro_var import fit_micro_var1
from causal_emergence.svd_ce import compute_svd_causal_emergence
from causal_emergence.liu_exact_ce import compute_liu_exact_emergence


def test_analytical_ei_identity_system():
    """For A = I_p and Sigma_eps = I_p with kappa_do = 1.0:
       EI = 0.5 * ln det(I_p + I_p) = 0.5 * p * ln(2)
    """
    p = 5
    A = np.eye(p)
    Sigma_eps = np.eye(p)
    Sigma_x = np.eye(p)
    ei = compute_continuous_ei(A, Sigma_eps, Sigma_x=Sigma_x, kappa_do=1.0)
    expected_ei = 0.5 * p * np.log(2.0)
    assert np.isclose(ei, expected_ei, atol=1e-6)


def test_strict_scale_invariance():
    """Verify that scaling data x -> c * x leaves EI completely unchanged."""
    np.random.seed(42)
    p = 6
    X_dec = np.random.randn(200, p) * 0.01
    X_pct = X_dec * 100.0  # 100x scaling

    A_dec, S_eps_dec = fit_micro_var1(X_dec)
    A_pct, S_eps_pct = fit_micro_var1(X_pct)

    S_x_dec = np.cov(X_dec, rowvar=False)
    S_x_pct = np.cov(X_pct, rowvar=False)

    ei_dec = compute_continuous_ei(A_dec, S_eps_dec, Sigma_x=S_x_dec)
    ei_pct = compute_continuous_ei(A_pct, S_eps_pct, Sigma_x=S_x_pct)

    assert np.isclose(ei_dec, ei_pct, atol=1e-6)


def test_stiefel_matrix_orthogonality():
    """Verify that W initialized on Stiefel satisfies W @ W.T = I_q."""
    p, q = 10, 3
    W = initialize_stiefel_matrix(p, q, method="random")
    assert W.shape == (q, p)
    gram = W @ W.T
    assert np.allclose(gram, np.eye(q), atol=1e-6)


def test_macro_dynamics_exact_projection():
    """Verify exact conditional Markov projection dimension and positive-definiteness."""
    p, q = 6, 2
    A = 0.5 * np.eye(p)
    Sigma_eps = 0.8 * np.eye(p)
    Sigma_x = 1.2 * np.eye(p)
    W = initialize_stiefel_matrix(p, q, method="random")

    A_y, Sigma_eps_y, Sigma_x_y = compute_macro_dynamics(A, Sigma_eps, W, Sigma_x=Sigma_x)
    assert A_y.shape == (q, q)
    assert Sigma_eps_y.shape == (q, q)
    assert Sigma_x_y.shape == (q, q)
    assert np.all(np.linalg.eigvalsh(Sigma_eps_y) > 0)


def test_emergence_spectrum():
    """Test CEFI density and q* extraction logic."""
    ei_micro = 6.0
    macro_ei_dict = {1: 1.5, 2: 3.6, 3: 3.9, 4: 4.4}
    cefi, q_star, deltas, cefi_raw = compute_emergence_spectrum(ei_micro, macro_ei_dict, p_micro=6)

    assert np.isclose(cefi, 0.8)
    assert q_star == 2
    assert np.isclose(deltas[2], 0.8)


def test_pre2025_svd_analytical_benchmark():
    """
    Validates SVD-based Causal Emergence (Liu, Pan, Wang, Yang, Yuan, Zhang, PRE 2025).
    For diagonal system A = diag(2, 0.1), Sigma_eps = diag(1, 1), optimal macro is q=1.
    """
    A = np.diag([2.0, 0.1])
    Sigma_eps = np.eye(2)
    ce_svd, q_star_svd, s_vals, deltas = compute_svd_causal_emergence(A, Sigma_eps)

    assert q_star_svd == 1
    assert ce_svd > 0.0
    assert np.isclose(s_vals[0], 2.0)
    assert np.isclose(s_vals[1], 0.1)


def test_liu2024_exact_delta_j_benchmark():
    """
    Validates Liu, Yuan & Zhang (2024) exact Delta J formulation on 2D coupled system.
    """
    A = np.array([[0.8, 0.4], [0.4, 0.8]])
    Sigma_eps = 0.5 * np.eye(2)
    W_dict = {1: np.array([[1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)]])}

    delta_J, q_star, delta_J_dict = compute_liu_exact_emergence(A, Sigma_eps, W_dict)
    assert q_star == 1
    assert isinstance(delta_J, float)
