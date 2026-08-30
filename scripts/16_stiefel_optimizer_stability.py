#!/usr/bin/env python3
"""
Script 16: Stiefel Manifold Optimizer Convergence & Multistart Stability Check
=============================================================================
Evaluates optimizer stability across iterations (35, 75, 150) and restarts (4, 10, 25)
on historical benchmark windows (Calm 2005, 2008 GFC Peak, 2020 COVID Shock).
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from causal_emergence.micro_var import fit_micro_var1
from causal_emergence.stiefel_optimizer import optimize_coarse_graining_stiefel
from causal_emergence.analytical_ei import compute_continuous_ei, compute_macro_ei


def evaluate_stability():
    df = pd.read_csv("data/raw/ff30_daily_returns.csv", parse_dates=["Date"], index_col="Date")
    regimes = [
        ("Calm 2005", "2005-01-01", "2006-12-31"),
        ("2008 GFC Peak", "2007-10-01", "2009-06-30"),
        ("2020 COVID Shock", "2020-02-01", "2020-05-31")
    ]

    p = df.shape[1]
    configs = [
        (35, 4),
        (75, 10),
        (150, 25)
    ]

    print("\n" + "="*80)
    print("STIEFEL OPTIMIZER CONVERGENCE & MULTISTART STABILITY DIAGNOSTICS")
    print("="*80)

    for r_name, start, end in regimes:
        sub_df = df.loc[start:end]
        if len(sub_df) < 250:
            idx = df.index.get_indexer([pd.to_datetime(end)], method="nearest")[0]
            sub_df = df.iloc[idx - 500 : idx]
        else:
            sub_df = sub_df.iloc[-500:] if len(sub_df) >= 500 else sub_df

        A, Sigma_eps = fit_micro_var1(sub_df.values)
        Sigma_x = np.cov(sub_df.values, rowvar=False)
        ei_micro = compute_continuous_ei(A, Sigma_eps, Sigma_x=Sigma_x)

        print(f"\n>>> Regime: {r_name} (T={len(sub_df)}, p={p}, EI_micro/p={ei_micro/p:.4f})")
        for n_iter, n_restarts in configs:
            best_cefi = -1e9
            best_q = -1
            for q in range(1, p):
                W_opt, obj_opt = optimize_coarse_graining_stiefel(
                    A, Sigma_eps, q=q, Sigma_x=Sigma_x,
                    max_iter=n_iter, n_restarts=n_restarts
                )

                cefi_q = (obj_opt / q) - (ei_micro / p)
                if cefi_q > best_cefi:
                    best_cefi = cefi_q
                    best_q = q

            print(f"    [Iter={n_iter:3d}, Restarts={n_restarts:2d}]: Optimal q* = {best_q:2d} | Max CEFI = {best_cefi:.5f}")


if __name__ == "__main__":
    evaluate_stability()
