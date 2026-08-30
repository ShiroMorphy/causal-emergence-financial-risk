"""
Non-Parametric Bootstrap Inference Engine
=========================================
Implements Stationary Block Bootstrap (Politis & Romano, 1994) for dependent
financial time series and surrogate data testing against the null of no emergence.
"""

from typing import Tuple
import numpy as np
from causal_emergence.dynamic_pipeline import run_single_window_ce



def generate_stationary_bootstrap_indices(T: int, p_geom: float = 0.05) -> np.ndarray:
    """
    Generates time series indices of length T using the Stationary Bootstrap
    with geometrically distributed block lengths (Politis & Romano, 1994).
    """
    indices = np.zeros(T, dtype=int)
    current_idx = np.random.randint(0, T)

    for t in range(T):
        if np.random.rand() < p_geom:
            current_idx = np.random.randint(0, T)
        else:
            current_idx = (current_idx + 1) % T
        indices[t] = current_idx

    return indices


def stationary_block_bootstrap_cefi(
    window_data: np.ndarray,
    n_bootstraps: int = 200,
    mean_block_length: int = 20,
    q_candidates: list = None
) -> Tuple[float, float, float]:
    """
    Calculates non-parametric 95% confidence intervals for CEFI in a given time window.

    Returns
    -------
    cefi_estimate : float
    ci_lower : float (2.5 percentile)
    ci_upper : float (97.5 percentile)
    """
    T = window_data.shape[0]
    p_geom = 1.0 / mean_block_length

    # Point estimate
    base_res = run_single_window_ce(window_data, q_candidates=q_candidates)
    cefi_point = base_res["cefi"]

    boot_cefis = []
    for _ in range(n_bootstraps):
        boot_idx = generate_stationary_bootstrap_indices(T, p_geom=p_geom)
        boot_data = window_data[boot_idx]
        res_b = run_single_window_ce(boot_data, q_candidates=q_candidates, n_restarts=1, max_iter=20)
        boot_cefis.append(res_b["cefi"])

    ci_lower = float(np.percentile(boot_cefis, 2.5))
    ci_upper = float(np.percentile(boot_cefis, 97.5))

    return cefi_point, ci_lower, ci_upper
