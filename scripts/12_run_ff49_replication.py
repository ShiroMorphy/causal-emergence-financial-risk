#!/usr/bin/env python3
"""
Script 12: Cross-Universe Replication on Fama-French 49 Industry Portfolios (FF49)
==================================================================================
Evaluates Causal Emergence across the complete dimensional grid q in 1..48 and
runs matched null tests on FF49 (p = 49).
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from joblib import Parallel, delayed

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from utils.data_fetcher import download_fama_french_49_industry
from causal_emergence.dynamic_pipeline import run_single_window_ce
from causal_emergence.null_models import (
    generate_static_correlation_null_data,
    generate_diag_plus_contemp_null_data,
    evaluate_single_null_realization
)
from causal_emergence.micro_var import fit_micro_var1
from causal_emergence.analytical_ei import compute_continuous_ei, compute_emergence_spectrum
from causal_emergence.stiefel_optimizer import optimize_coarse_graining_stiefel


def run_null_for_ff49_window(window_slice: np.ndarray, null_name: str, B: int = 100, q_candidates: list = None, n_jobs: int = -1):
    p = window_slice.shape[1]
    A_obs, S_eps_obs = fit_micro_var1(window_slice)
    S_x_obs = np.cov(window_slice, rowvar=False)
    ei_m_obs = compute_continuous_ei(A_obs, S_eps_obs, Sigma_x=S_x_obs, kappa_do=1.0)
    macro_eis_obs = {}
    for q in q_candidates:
        _, ei_q = optimize_coarse_graining_stiefel(
            A_obs, S_eps_obs, q=q, Sigma_x=S_x_obs, kappa_do=1.0, n_restarts=2, max_iter=25
        )
        macro_eis_obs[q] = ei_q
    cefi_obs, q_obs, _, _ = compute_emergence_spectrum(ei_m_obs, macro_eis_obs, p_micro=p)

    def _eval_null(seed):
        np.random.seed(seed)
        if null_name == "static":
            X_s = generate_static_correlation_null_data(window_slice)
        elif null_name == "diag_contemp":
            X_s = generate_diag_plus_contemp_null_data(window_slice, A_obs, S_eps_obs)
        else:
            raise ValueError()
        return evaluate_single_null_realization(X_s, q_candidates=q_candidates, kappa_do=1.0, n_restarts=2, max_iter=25)

    seeds = np.random.randint(1000, 9999999, size=B)
    null_res = Parallel(n_jobs=n_jobs)(delayed(_eval_null)(s) for s in seeds)
    null_cefis = np.array([r[0] for r in null_res])

    mu_0 = float(np.mean(null_cefis))
    q95_0 = float(np.percentile(null_cefis, 95))
    p_emp = float((1.0 + np.sum(null_cefis >= cefi_obs)) / (B + 1.0))
    z_dev = float((cefi_obs - mu_0) / max(float(np.std(null_cefis)), 1e-12))

    return {"cefi_obs": cefi_obs, "q_obs": q_obs, "mu_0": mu_0, "q95_0": q95_0, "z_dev": z_dev, "p_emp": p_emp}


def main():
    parser = argparse.ArgumentParser(description="Replicate on FF49 with complete grid q=1..48.")
    parser.add_argument("--output-csv", type=str, default="data/features/cefi_ff49_daily_series.csv")
    parser.add_argument("--step", type=int, default=10)
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()

    raw_path = "data/raw/ff49_daily_returns.csv"
    if not os.path.exists(raw_path):
        df_ff49 = download_fama_french_49_industry(start_year=1990)
        df_ff49.to_csv(raw_path)
    else:
        df_ff49 = pd.read_csv(raw_path, parse_dates=["Date"], index_col="Date")

    T, p = df_ff49.shape
    # Complete integer spectrum q in 1..48
    q_all = list(range(1, p))

    print(f"Running CEFI on FF49 across complete dimensional grid (q = 1..{p-1})...")

    # 1. Historical Slices
    window_slices = [df_ff49.iloc[t_end - 500 : t_end].values for t_end in range(500, T + 1, args.step)]
    slice_dates = [df_ff49.index[t_end - 1] for t_end in range(500, T + 1, args.step)]

    results = Parallel(n_jobs=args.n_jobs)(
        delayed(run_single_window_ce)(w, q_candidates=q_all, kappa_do=1.0, n_restarts=2, max_iter=25)
        for w in window_slices
    )

    out_df = pd.DataFrame(results, index=slice_dates)
    out_df.to_csv(args.output_csv)

    print("\n" + "=" * 90)
    print(f"REPLICACIÓN EN FF49 (GRID COMPLETO q = 1..{p-1}):")
    print("=" * 90)
    print(f"Mean CEFI (FF49):           {out_df['cefi'].mean():.4f}")
    print(f"Modal Causal Dimension q*:  {out_df['q_star'].mode()[0]} (out of 49 micro assets)")
    print(f"Frecuencia de q* <= 4:      {(out_df['q_star'] <= 4).mean() * 100:.2f}%")

    # 2. Matched Null Inference on FF49 (Calm 2005 vs COVID 2020)
    print("\n" + "-" * 90)
    print("INFERENCIA NULA EN FF49 (Calm 2005 vs 2020 COVID Shock, B=100):")
    print("-" * 90)
    test_dates = [
        ("Calm 2005", "2005-01-03", "2006-12-30"),
        ("2020 COVID Shock", "2019-03-01", "2021-02-28")
    ]
    for label, start, end in test_dates:
        mask = (df_ff49.index >= pd.to_datetime(start)) & (df_ff49.index <= pd.to_datetime(end))
        w_slice = df_ff49.loc[mask].iloc[:500].values
        if len(w_slice) < 500:
            continue
        for null_t in ["static", "diag_contemp"]:
            n_res = run_null_for_ff49_window(w_slice, null_t, B=100, q_candidates=q_all, n_jobs=args.n_jobs)
            print(f"  FF49 [{label}] vs [{null_t.upper()}]: CEFI_obs = {n_res['cefi_obs']:.4f} (q*={n_res['q_obs']}) | E[CEFI_0] = {n_res['mu_0']:.4f} (Q95={n_res['q95_0']:.4f}) | z_dev = {n_res['z_dev']:+.2f} | p_emp = {n_res['p_emp']:.4f}")


if __name__ == "__main__":
    main()
