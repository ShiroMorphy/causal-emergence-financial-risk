#!/usr/bin/env python3
"""
Script 09: Theoretical Framework Robustness & Cross-Method Validation
====================================================================
Compares:
1. Scale-Invariant Continuous Gaussian Stiefel CEFI (Framework A)
2. Liu, Yuan & Zhang (2024) exact bounded-uniform Delta J
3. Kaiwei Liu, Linli Pan, Zhipeng Wang, Mingzhe Yang, Bing Yuan, Jiang Zhang (Phys. Rev. E, 2025) SVD Emergence
4. Framework B: Conditional Expectation with full micro-uncertainty covariance
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from joblib import Parallel, delayed

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from causal_emergence.micro_var import fit_micro_var1
from causal_emergence.analytical_ei import compute_continuous_ei, compute_emergence_spectrum
from causal_emergence.stiefel_optimizer import optimize_coarse_graining_stiefel
from causal_emergence.svd_ce import compute_svd_causal_emergence
from causal_emergence.liu_exact_ce import compute_liu_exact_emergence


def evaluate_window_all_frameworks(window_slice: np.ndarray, q_candidates: list):
    p = window_slice.shape[1]
    A_micro, Sigma_eps = fit_micro_var1(window_slice)
    Sigma_x = np.cov(window_slice, rowvar=False)

    # 1. Framework A: Stiefel Optimization (Gaussian Scale-Invariant)
    ei_micro_A = compute_continuous_ei(A_micro, Sigma_eps, Sigma_x=Sigma_x, kappa_do=1.0)
    macro_eis_A = {}
    W_dict_A = {}
    for q in q_candidates:
        W_opt, ei_q_A = optimize_coarse_graining_stiefel(
            A_micro, Sigma_eps, q=q, Sigma_x=None, kappa_do=1.0, n_restarts=2, max_iter=25
        )
        macro_eis_A[q] = ei_q_A
        W_dict_A[q] = W_opt

    cefi_A, q_A, _, _ = compute_emergence_spectrum(ei_micro_A, macro_eis_A, p_micro=p)

    # 2. Liu, Yuan & Zhang (2024) exact Delta J
    delta_J_liu, q_liu, _ = compute_liu_exact_emergence(
        A_micro, Sigma_eps, W_dict=W_dict_A, Sigma_x=Sigma_x
    )

    # 3. PRE-2025 SVD Analytical Emergence (Liu et al., Phys. Rev. E, 2025)
    cefi_svd, q_svd, s_vals, svd_deltas = compute_svd_causal_emergence(
        A_micro, Sigma_eps, Sigma_x=Sigma_x, kappa_do=1.0
    )

    # Compute Subspace Principal Angle / Projection Distance for optimal dimension
    # W_svd is formed by top q left singular vectors of K = L^{-1} A
    Sigma_clean = 0.5 * (Sigma_eps + Sigma_eps.T) + 1e-10 * np.trace(Sigma_eps)/float(p) * np.eye(p)
    L_mat = np.linalg.cholesky(Sigma_clean)
    K = np.linalg.solve(L_mat, A_micro)
    U_svd, _, _ = np.linalg.svd(K, full_matrices=False)
    W_svd_q = U_svd[:, :q_A].T  # Shape (q_A, p)

    W_opt_q = W_dict_A[q_A]      # Shape (q_A, p)
    # Grassmann projection distance: || W_opt^T W_opt - W_svd^T W_svd ||_F
    proj_diff = (W_opt_q.T @ W_opt_q) - (W_svd_q.T @ W_svd_q)
    subspace_dist = float(np.linalg.norm(proj_diff, ord="fro"))
    objective_gap = float(abs(cefi_A - svd_deltas[q_A]))

    # 4. Framework B: Conditional Uncertainty Covariance
    macro_eis_B = {}
    for q in q_candidates:
        W_q = W_dict_A[q]
        W_cov = W_q @ Sigma_x @ W_q.T
        K_cond = Sigma_x @ W_q.T @ np.linalg.pinv(W_cov)
        A_M_B = W_q @ A_micro @ K_cond
        res_cov = Sigma_x - Sigma_x @ W_q.T @ np.linalg.pinv(W_cov) @ W_q @ Sigma_x
        Sigma_M_B = W_q @ Sigma_eps @ W_q.T + W_q @ A_micro @ res_cov @ A_micro.T @ W_q.T

        ei_q_B = compute_continuous_ei(A_M_B, Sigma_M_B, Sigma_x=W_cov, kappa_do=1.0)
        macro_eis_B[q] = ei_q_B

    cefi_B, q_B, _, _ = compute_emergence_spectrum(ei_micro_A, macro_eis_B, p_micro=p)

    return {
        "cefi_A": cefi_A, "q_A": q_A,
        "delta_J_liu": delta_J_liu, "q_liu": q_liu,
        "cefi_svd": cefi_svd, "q_svd": q_svd,
        "cefi_B": cefi_B, "q_B": q_B,
        "subspace_dist": subspace_dist,
        "objective_gap": objective_gap
    }


def main():
    parser = argparse.ArgumentParser(description="Cross-Framework Historical Robustness.")
    parser.add_argument("--input-file", type=str, default="data/raw/ff30_daily_returns.csv")
    parser.add_argument("--output-file", type=str, default="reports/tables/table_cross_framework_comparison.tex")
    parser.add_argument("--step", type=int, default=10)
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()

    df = pd.read_csv(args.input_file, parse_dates=["Date"], index_col="Date")
    T, p = df.shape
    q_all = list(range(1, p))

    window_slices = [df.iloc[t_end - 500 : t_end].values for t_end in range(500, T + 1, args.step)]
    slice_dates = [df.index[t_end - 1] for t_end in range(500, T + 1, args.step)]

    print(f"Running Cross-Method Validation on {len(window_slices)} historical slices (1992-2026)...")

    results = Parallel(n_jobs=args.n_jobs)(
        delayed(evaluate_window_all_frameworks)(w, q_all) for w in window_slices
    )
    res_df = pd.DataFrame(results, index=slice_dates)
    os.makedirs("data/features", exist_ok=True)
    res_df.to_csv("data/features/framework_comparison_series.csv")

    # Cross-Method Metrics
    ref = res_df["cefi_A"]

    ref_q = res_df["q_A"]

    comparisons = [
        ("Liu et al. (2024) Uniform Delta J", res_df["delta_J_liu"], res_df["q_liu"]),
        ("Liu, Pan, Wang, Yang, Yuan, Zhang (PRE 2025) SVD", res_df["cefi_svd"], res_df["q_svd"]),
        ("Framework B (Conditional Uncertainty Covariance)", res_df["cefi_B"], res_df["q_B"])
    ]

    summary_rows = []
    for name, s_cefi, s_q in comparisons:
        p_corr = float(np.corrcoef(ref, s_cefi)[0, 1])
        s_corr, _ = spearmanr(ref, s_cefi)
        exact_q = float(np.mean(ref_q == s_q) * 100.0)
        near_q = float(np.mean(np.abs(ref_q - s_q) <= 1) * 100.0)

        summary_rows.append({
            "Theoretical Framework / Benchmark": name,
            "Pearson Corr vs CEFI": f"{p_corr:.3f}",
            "Spearman Corr vs CEFI": f"{s_corr:.3f}",
            "Exact q* Match (%)": f"{exact_q:.1f}%",
            "Relaxed q* Match (+/-1) (%)": f"{near_q:.1f}%"
        })

    summary_df = pd.DataFrame(summary_rows).set_index("Theoretical Framework / Benchmark")
    
    mean_obj_gap = float(res_df["objective_gap"].mean())
    mean_subspace_dist = float(res_df["subspace_dist"].mean())

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        f.write("% Cross-Framework and Literature Benchmarking (1992-2026)\n")
        f.write(summary_df.to_latex(caption="Cross-Method Benchmarking: Comparing CEFI with Liu et al. (2024), PRE (2025) SVD, and Framework B", label="tab:cross_framework"))

    print("\n" + "=" * 95)
    print("RESUMEN DE ROBUSTEZ Y VALIDACIÓN CRUZADA CONTRA LA LITERATURA:")
    print("=" * 95)
    print(summary_df.to_string())
    print(f"\nMean Stiefel-SVD Objective Function Gap:  {mean_obj_gap:.5f} nats")
    print(f"Mean Grassmann Subspace Frobenius Distance: {mean_subspace_dist:.4f}")


if __name__ == "__main__":
    main()
