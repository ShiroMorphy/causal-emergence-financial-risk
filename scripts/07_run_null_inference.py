#!/usr/bin/env python3
"""
Script 07: Matched Null Model Inference for Causal Emergence
============================================================
Evaluates 4 matched surrogate null ensembles:
1. H_0^{circ}: Circular time-shift null
2. H_0^{diag}: Decoupled diagonal VAR(1) null
3. H_0^{static}: Static correlation mode null (preserves exact Sigma_x)
4. H_0^{diag+contemp}: Diagonal VAR(1) with full contemporaneous Sigma_eps

IMPORTANT: The empirical statistic CEFI_obs is computed ONCE deterministically
per window with fixed seed and deterministic restarts, and compared consistently
against all 4 surrogate distributions.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
from joblib import Parallel, delayed

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from causal_emergence.micro_var import fit_micro_var1
from causal_emergence.analytical_ei import compute_continuous_ei, compute_emergence_spectrum
from causal_emergence.stiefel_optimizer import optimize_coarse_graining_stiefel
from causal_emergence.null_models import (
    generate_circular_null_data,
    generate_diagonal_var_null_data,
    generate_static_correlation_null_data,
    generate_diag_plus_contemp_null_data,
    evaluate_single_null_realization
)


def compute_observed_cefi_deterministic(window_data: np.ndarray, q_candidates: list, seed: int = 42) -> tuple:
    """Computes the observed empirical CEFI and q* once with fixed random seed and robust restarts."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    T, p = window_data.shape
    A_obs, S_eps_obs = fit_micro_var1(window_data)
    S_x_obs = np.cov(window_data, rowvar=False)

    ei_m_obs = compute_continuous_ei(A_obs, S_eps_obs, Sigma_x=S_x_obs, kappa_do=1.0)
    macro_eis_obs = {}
    for q in q_candidates:
        _, ei_q = optimize_coarse_graining_stiefel(
            A_obs, S_eps_obs, q=q, Sigma_x=S_x_obs, kappa_do=1.0, n_restarts=4, max_iter=35
        )
        macro_eis_obs[q] = ei_q

    cefi_obs, q_obs, _, _ = compute_emergence_spectrum(ei_m_obs, macro_eis_obs, p_micro=p)
    return cefi_obs, q_obs, A_obs, S_eps_obs


def evaluate_null_ensemble_for_window(
    window_data: np.ndarray,
    cefi_obs: float,
    q_obs: int,
    A_obs: np.ndarray,
    S_eps_obs: np.ndarray,
    null_name: str,
    B: int = 500,
    q_candidates: list = None,
    n_jobs: int = -1
) -> dict:
    """Evaluates B surrogate realizations of the specified null against the fixed observed CEFI."""
    T, p = window_data.shape
    if q_candidates is None:
        q_candidates = list(range(1, p))

    def _gen_and_eval(seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if null_name == "circular":
            X_surr = generate_circular_null_data(window_data)
        elif null_name == "diagonal":
            X_surr = generate_diagonal_var_null_data(window_data, A_obs, S_eps_obs)
        elif null_name == "static":
            X_surr = generate_static_correlation_null_data(window_data)
        elif null_name == "diag_contemp":
            X_surr = generate_diag_plus_contemp_null_data(window_data, A_obs, S_eps_obs)
        else:
            raise ValueError(f"Unknown null: {null_name}")

        return evaluate_single_null_realization(
            X_surr, q_candidates=q_candidates, kappa_do=1.0, n_restarts=2, max_iter=30
        )

    seeds = np.random.randint(1000, 9999999, size=B)
    null_results = Parallel(n_jobs=n_jobs)(delayed(_gen_and_eval)(s) for s in seeds)

    null_cefis = np.array([r[0] for r in null_results])
    null_q_stars = np.array([r[1] for r in null_results])

    mu_0 = float(np.mean(null_cefis))
    std_0 = float(np.std(null_cefis))
    q95_0 = float(np.percentile(null_cefis, 95))
    q99_0 = float(np.percentile(null_cefis, 99))

    # Exact non-parametric empirical p-value: (1 + sum(T_b >= T_obs)) / (B + 1)
    p_emp = float((1.0 + np.sum(null_cefis >= cefi_obs)) / (B + 1.0))
    z_score = float((cefi_obs - mu_0) / max(std_0, 1e-12))

    return {
        "null_name": null_name,
        "B": B,
        "cefi_obs": cefi_obs,
        "q_obs": q_obs,
        "mu_null": mu_0,
        "std_null": std_0,
        "q95_null": q95_0,
        "q99_null": q99_0,
        "z_score": z_score,
        "p_emp": p_emp,
        "modal_q_null": int(pd.Series(null_q_stars).mode()[0])
    }


def main():
    parser = argparse.ArgumentParser(description="Run matched null inference.")
    parser.add_argument("--input-file", type=str, default="data/raw/ff30_daily_returns.csv")
    parser.add_argument("--B", type=int, default=500, help="Replications per null model.")
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()

    df = pd.read_csv(args.input_file, parse_dates=["Date"], index_col="Date")
    p = df.shape[1]
    q_all = list(range(1, p))

    test_dates = [
        ("Calm Period (2005)", "2005-01-03", "2006-12-30"),
        ("2008 GFC Peak", "2007-09-01", "2009-08-31"),
        ("2020 COVID Shock", "2019-03-01", "2021-02-28")
    ]

    print("\n" + "=" * 95)
    print(f"INFERENCIA CON MODELOS NULOS EMPAREJADOS (B={args.B}, q=1..29, CEFI_obs ÚNICO)")
    print("=" * 95)

    summary_rows = []

    for label, start, end in test_dates:
        mask = (df.index >= pd.to_datetime(start)) & (df.index <= pd.to_datetime(end))
        window_slice = df.loc[mask].iloc[:500].values
        if len(window_slice) < 500:
            continue

        # 1. Compute CEFI_obs ONCE deterministically
        cefi_obs, q_obs, A_obs, S_eps_obs = compute_observed_cefi_deterministic(window_slice, q_all, seed=42)

        print(f"\n>>> Régimen: [{label}] | CEFI_obs = {cefi_obs:.4f} (q* = {q_obs}) (Fijo para todos los nulls)")

        # 2. Evaluate all 4 nulls against this exact same cefi_obs
        for null_type in ["circular", "diagonal", "static", "diag_contemp"]:
            res = evaluate_null_ensemble_for_window(
                window_slice, cefi_obs, q_obs, A_obs, S_eps_obs,
                null_name=null_type, B=args.B, q_candidates=q_all, n_jobs=args.n_jobs
            )
            print(f"  [{res['null_name'].upper():12s}]: CEFI_obs={res['cefi_obs']:.4f} | E[CEFI_0]={res['mu_null']:+.4f} (Q95={res['q95_null']:+.4f}) | z_dev={res['z_score']:+.2f} | p_emp={res['p_emp']:.4f} | Modal q_0*={res['modal_q_null']}")

            summary_rows.append({
                "Regime": label,
                "Null_Model": null_type,
                "CEFI_Obs": res["cefi_obs"],
                "q_Obs": res["q_obs"],
                "E_CEFI_0": res["mu_null"],
                "Std_0": res["std_null"],
                "Q95_0": res["q95_null"],
                "Z_Deviation": res["z_score"],
                "p_emp": res["p_emp"],
                "Modal_q_0": res["modal_q_null"]
            })

    summary_df = pd.DataFrame(summary_rows)
    os.makedirs("reports/tables", exist_ok=True)
    summary_df.to_csv("reports/tables/table_matched_null_inference.csv", index=False)
    print(f"\nResultados guardados en reports/tables/table_matched_null_inference.csv")


if __name__ == "__main__":
    main()
