#!/usr/bin/env python3
"""
Script 08: Comprehensive Sensitivity Analysis of Intervention Strength Kappa
=============================================================================
Evaluates CEFI_t across kappa in {0.25, 0.5, 1.0, 2.0, 4.0} and tests if:
1. Historical ranking (Spearman rho) is preserved.
2. H2 Crisis Effect (beta_Crisis > 0) is invariant across kappa.
3. H3 Dimensional Collapse (q* in 2..4) holds invariant across kappa.
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
from econometrics.predictive_regressions import run_predictive_regression_hac


def main():
    parser = argparse.ArgumentParser(description="Run Kappa Sensitivity Analysis.")
    parser.add_argument("--input-file", type=str, default="data/raw/ff30_daily_returns.csv")
    parser.add_argument("--output-file", type=str, default="reports/tables/table_kappa_sensitivity.tex")
    parser.add_argument("--step", type=int, default=10)
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()

    df = pd.read_csv(args.input_file, parse_dates=["Date"], index_col="Date")
    T, p = df.shape
    q_all = list(range(1, p))

    kappas = [0.25, 0.5, 1.0, 2.0, 4.0]
    window_slices = [df.iloc[t_end - 500 : t_end].values for t_end in range(500, T + 1, args.step)]
    slice_dates = [df.index[t_end - 1] for t_end in range(500, T + 1, args.step)]

    # Crisis dummy for H2 test
    crisis_intervals = [
        ("2008 GFC", "2007-10-01", "2009-06-30"),
        ("2020 COVID Crash", "2020-02-01", "2020-05-30")
    ]
    crisis_dummy = np.zeros(len(slice_dates))
    for i, d in enumerate(slice_dates):
        for _, start, end in crisis_intervals:
            if pd.to_datetime(start) <= d <= pd.to_datetime(end):
                crisis_dummy[i] = 1
                break

    results_by_kappa = {}
    for kappa in kappas:
        res_list = Parallel(n_jobs=args.n_jobs)(
            delayed(run_single_window_ce)(w, q_candidates=q_all, kappa_do=kappa, n_restarts=2, max_iter=25)
            for w in window_slices
        )
        cefi_series = np.array([r["cefi"] for r in res_list])
        q_series = np.array([r["q_star"] for r in res_list])
        results_by_kappa[kappa] = {"cefi": cefi_series, "q_star": q_series}

    ref_cefi = results_by_kappa[1.0]["cefi"]

    summary_rows = []
    for kappa in kappas:
        c_k = results_by_kappa[kappa]["cefi"]
        q_k = results_by_kappa[kappa]["q_star"]

        sp_corr, _ = spearmanr(ref_cefi, c_k)
        p_corr = float(np.corrcoef(ref_cefi, c_k)[0, 1])

        # H2 HAC Regression of CEFI on Crisis Dummy
        X_reg = np.column_stack([np.ones(len(c_k)), crisis_dummy])
        reg_h2 = run_predictive_regression_hac(c_k, X_reg, feature_names=["const", "Crisis"], max_lags=20)
        beta_crisis = reg_h2["params"]["Crisis"]
        t_crisis = reg_h2["tvalues"]["Crisis"]

        med_q_crisis = float(np.median(q_k[crisis_dummy == 1]))
        med_q_calm = float(np.median(q_k[crisis_dummy == 0]))

        summary_rows.append({
            "Kappa (kappa)": f"{kappa:.2f}",
            "Spearman vs kappa=1": f"{sp_corr:.3f}",
            "Pearson vs kappa=1": f"{p_corr:.3f}",
            "Modal q*": f"{int(pd.Series(q_k).mode()[0])}",
            "q* Crisis / Calm": f"{med_q_crisis:.0f} / {med_q_calm:.0f}",
            "H2 Beta (Crisis)": f"{beta_crisis:+.3f}",
            "H2 t-stat": f"{t_crisis:+.2f}"
        })

    summary_df = pd.DataFrame(summary_rows).set_index("Kappa (kappa)")
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        f.write("% Sensitivity Analysis of Intervention Scale Parameter Kappa (1992-2026)\n")
        f.write(summary_df.to_latex(caption="Robustness to Intervention Scale Parameter $\\kappa$: Hypothesis Invariance (1992--2026)", label="tab:kappa_sensitivity"))

    print("\n" + "=" * 90)
    print("MATRIZ DE SENSIBILIDAD DE KAPPA E INVARIANZA DE HIPÓTESIS:")
    print("=" * 90)
    print(summary_df.to_string())


if __name__ == "__main__":
    main()
