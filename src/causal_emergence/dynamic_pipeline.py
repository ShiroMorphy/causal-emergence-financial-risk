"""
Dynamic Rolling-Window Pipeline for Scale-Invariant Causal Emergence
====================================================================
Extracts time series of CEFI_t, Excess CEFI_t, and q_t^* across rolling windows with
complete dimensional grid evaluation and parallel multi-core execution.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from .micro_var import fit_micro_var1
from .analytical_ei import compute_continuous_ei, compute_macro_ei, compute_emergence_spectrum
from .stiefel_optimizer import optimize_coarse_graining_stiefel


def compute_finite_sample_null_bias(
    W: int,
    p: int,
    q_candidates: List[int],
    n_sims: int = 10,
    kappa_do: float = 1.0
) -> Dict[int, float]:
    """
    Computes the expected finite-sample optimization bias E_{H0}[EI_q / q - EI_p / p]
    under the null hypothesis of pure independent Gaussian noise (zero true interaction).
    """
    null_biases = {q: [] for q in q_candidates}

    for _ in range(n_sims):
        X_null = np.random.randn(W, p)
        A_null, S_eps_null = fit_micro_var1(X_null)
        S_x_null = np.cov(X_null, rowvar=False)

        ei_p_null = compute_continuous_ei(A_null, S_eps_null, Sigma_x=S_x_null, kappa_do=kappa_do)
        ei_p_dens = ei_p_null / float(p)

        for q in q_candidates:
            _, ei_q_null = optimize_coarse_graining_stiefel(
                A_null, S_eps_null, q=q, Sigma_x=S_x_null, kappa_do=kappa_do,
                n_restarts=1, max_iter=25
            )
            bias_q = (ei_q_null / float(q)) - ei_p_dens
            null_biases[q].append(bias_q)

    mean_null_bias = {q: float(np.mean(null_biases[q])) for q in q_candidates}
    return mean_null_bias


def run_single_window_ce(
    window_data: np.ndarray,
    q_candidates: Optional[List[int]] = None,
    kappa_do: float = 1.0,
    null_bias_dict: Optional[Dict[int, float]] = None,
    n_restarts: int = 12,
    max_iter: int = 100
) -> Dict:
    """
    Computes scale-invariant Effective Information (micro and macro) and CEFI for a single time slice.
    """
    W, p = window_data.shape
    if q_candidates is None:
        q_candidates = list(range(1, p))

    # Empirical state covariance and VAR(1)
    Sigma_x = np.cov(window_data, rowvar=False)
    A_micro, Sigma_micro = fit_micro_var1(window_data)

    ei_micro = compute_continuous_ei(
        A_micro, Sigma_micro, Sigma_x=Sigma_x, kappa_do=kappa_do
    )

    macro_eis = {}
    optimal_Ws = {}

    for q in q_candidates:
        W_opt, ei_q = optimize_coarse_graining_stiefel(
            A_micro, Sigma_micro, q=q, Sigma_x=Sigma_x, kappa_do=kappa_do,
            n_restarts=n_restarts, max_iter=max_iter
        )
        macro_eis[q] = ei_q
        optimal_Ws[q] = W_opt

    cefi_density, q_star, deltas, cefi_raw = compute_emergence_spectrum(
        ei_micro, macro_eis, p_micro=p
    )

    # Compute excess CEFI over finite-sample null bias
    if null_bias_dict is not None and q_star in null_bias_dict:
        expected_null = null_bias_dict[q_star]
        cefi_excess = float(cefi_density - expected_null)
    else:
        cefi_excess = cefi_density

    return {
        "ei_micro": ei_micro,
        "ei_micro_density": ei_micro / float(p),
        "macro_eis": macro_eis,
        "cefi": cefi_density,
        "cefi_excess": cefi_excess,
        "cefi_raw": cefi_raw,
        "q_star": q_star,
        "deltas": deltas,
        "best_W": optimal_Ws[q_star]
    }


def compute_dynamic_cefi_series(
    returns_df: Any,
    window_length: int = 500,
    step_size: int = 2,
    q_candidates: Optional[List[int]] = None,
    kappa_do: float = 1.0,
    n_restarts: int = 12,
    max_iter: int = 100,
    n_jobs: int = -1
) -> Any:
    """
    Generates historical daily time series of CEFI_t, Excess CEFI_t, and q_t^* across rolling windows.
    Evaluates complete dimensional grid q in 1 .. p-1 with 12 restarts / 100 iterations.
    """
    import pandas as pd
    from joblib import Parallel, delayed

    T_total, p = returns_df.shape
    dates = returns_df.index

    # Default to complete dimensional spectrum 1 .. p-1
    if q_candidates is None:
        q_candidates = list(range(1, p))

    print(f"Precomputing finite-sample optimization null baseline for W={window_length}, p={p}...")
    null_bias_dict = compute_finite_sample_null_bias(
        W=window_length, p=p, q_candidates=q_candidates, n_sims=8, kappa_do=kappa_do
    )

    tasks = []
    task_dates = []
    for t_end in range(window_length, T_total + 1, step_size):
        date_t = dates[t_end - 1]
        window_slice = returns_df.iloc[t_end - window_length : t_end].values
        tasks.append(window_slice)
        task_dates.append(date_t)

    print(f"Executing {len(tasks)} rolling windows in parallel (n_jobs={n_jobs}, restarts={n_restarts}, iters={max_iter}, q={q_candidates[0]}..{q_candidates[-1]})...")

    def _process_slice(window_slice, date_t):
        res = run_single_window_ce(
            window_slice,
            q_candidates=q_candidates,
            kappa_do=kappa_do,
            null_bias_dict=null_bias_dict,
            n_restarts=n_restarts,
            max_iter=max_iter
        )
        row = {
            "date": date_t,
            "ei_micro": res["ei_micro"],
            "ei_micro_density": res["ei_micro_density"],
            "macro_ei_max": res["macro_eis"][res["q_star"]],
            "macro_ei_max_density": res["macro_eis"][res["q_star"]] / float(res["q_star"]),
            "cefi": res["cefi"],
            "cefi_excess": res["cefi_excess"],
            "cefi_raw": res["cefi_raw"],
            "q_star": res["q_star"]
        }
        for q, delta in res["deltas"].items():
            row[f"delta_ei_q_{q}"] = delta
        return row

    results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_process_slice)(w, d) for w, d in zip(tasks, task_dates)
    )

    df_out = pd.DataFrame(results)
    if "date" in df_out.columns:
        df_out.set_index("date", inplace=True)
    return df_out
