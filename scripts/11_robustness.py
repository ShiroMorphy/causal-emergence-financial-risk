#!/usr/bin/env python3
"""
Script 11: Multi-Window Robustness & Lagged Peak Timing Analysis (W = 500, 750, 1000)
==================================================================================
Evaluates CEFI_t dynamics, modal q_t^*, and cross-lag timing Corr(CEFI^{500}_t, CEFI^{1000}_{t+l}).
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
from causal_emergence.dynamic_pipeline import run_single_window_ce


def main():
    parser = argparse.ArgumentParser(description="Multi-Window Robustness.")
    parser.add_argument("--input-file", type=str, default="data/raw/ff30_daily_returns.csv")
    parser.add_argument("--output-file", type=str, default="reports/tables/table_window_robustness.tex")
    parser.add_argument("--step", type=int, default=10)
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()

    df = pd.read_csv(args.input_file, parse_dates=["Date"], index_col="Date")
    T, p = df.shape
    q_all = list(range(1, p))

    windows = [500, 750, 1000]
    results_by_w = {}

    for W in windows:
        slices = [df.iloc[t_end - W : t_end].values for t_end in range(1000, T + 1, args.step)]
        res_list = Parallel(n_jobs=args.n_jobs)(
            delayed(run_single_window_ce)(w, q_candidates=q_all, kappa_do=1.0, n_restarts=2, max_iter=25)
            for w in slices
        )
        cefi_series = np.array([r["cefi"] for r in res_list])
        q_series = np.array([r["q_star"] for r in res_list])
        results_by_w[W] = {"cefi": cefi_series, "q_star": q_series}

    ref_cefi = results_by_w[500]["cefi"]
    ref_q = results_by_w[500]["q_star"]

    # Compute Lagged Cross-Correlation between W=500 and W=1000
    c_1000 = results_by_w[1000]["cefi"]
    max_lag_corr = 0.0
    best_lag = 0
    for lag in range(-25, 26):  # steps of 10 days => -250 to +250 days
        if lag < 0:
            corr_l = float(np.corrcoef(ref_cefi[:lag], c_1000[-lag:])[0, 1])
        elif lag > 0:
            corr_l = float(np.corrcoef(ref_cefi[lag:], c_1000[:-lag])[0, 1])
        else:
            corr_l = float(np.corrcoef(ref_cefi, c_1000)[0, 1])
        if corr_l > max_lag_corr:
            max_lag_corr = corr_l
            best_lag = lag * args.step

    summary_rows = []
    for W in windows:
        c_w = results_by_w[W]["cefi"]
        q_w = results_by_w[W]["q_star"]

        sp_corr, _ = spearmanr(ref_cefi, c_w)
        p_corr = float(np.corrcoef(ref_cefi, c_w)[0, 1])
        q_agree = float(np.mean(ref_q == q_w) * 100.0)

        summary_rows.append({
            "Window Length (W)": f"{W} trading days (~{W//250}y)",
            "Pearson Corr vs W=500": f"{p_corr:.3f}",
            "Spearman Corr vs W=500": f"{sp_corr:.3f}",
            "Modal q*": f"{int(pd.Series(q_w).mode()[0])}",
            "Mean CEFI": f"{np.mean(c_w):.3f}",
            "Std CEFI": f"{np.std(c_w):.3f}"
        })

    summary_df = pd.DataFrame(summary_rows).set_index("Window Length (W)")
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        f.write("% Multi-Window Estimation Robustness (1992-2026)\n")
        f.write(summary_df.to_latex(caption="Estimation Window Robustness ($W = 500, 750, 1000$ days)", label="tab:window_robustness"))

    print("\n" + "=" * 90)
    print("ROBUSTEZ A DIFERENTES VENTANAS TEMPORALES:")
    print("=" * 90)
    print(summary_df.to_string())
    print(f"\nPeak Cross-Correlation W=500 vs W=1000: rho = {max_lag_corr:.3f} (Lag: {best_lag:+d} trading days due to rolling filter phase shift).")


if __name__ == "__main__":
    main()
