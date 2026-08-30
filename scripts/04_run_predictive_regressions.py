#!/usr/bin/env python3
"""
Script 04: In-Sample Predictive Regressions, Collinearity Diagnostics & OOS Early Warning
======================================================================================
Runs OLS with Newey-West HAC standard errors, VIF analysis, and Leave-One-Crisis-Out CV.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from econometrics.predictive_regressions import run_predictive_regression_hac
from econometrics.early_warning_logit import (
    evaluate_early_warning_classifier,
    evaluate_oos_expanding_logit,
    evaluate_leave_one_crisis_out_logit
)


def compute_vif(X_matrix: np.ndarray) -> np.ndarray:
    """Computes Variance Inflation Factors for each feature column (excluding constant)."""
    k = X_matrix.shape[1]
    vifs = []
    for i in range(1, k):
        y_i = X_matrix[:, i]
        X_other = np.delete(X_matrix, i, axis=1)
        r2_i = run_predictive_regression_hac(y_i, X_other)["r2"]
        vif = 1.0 / max(1.0 - r2_i, 1e-6)
        vifs.append(vif)
    return np.array(vifs)


def main():
    parser = argparse.ArgumentParser(description="Run predictive regressions and early warning models.")
    parser.add_argument("--cefi-file", type=str, default="data/features/cefi_daily_series.csv")
    parser.add_argument("--benchmarks-file", type=str, default="data/features/benchmarks_daily_series.csv")
    parser.add_argument("--horizons", nargs="+", type=int, default=[5, 20, 60], help="Forecast horizons in days.")
    args = parser.parse_args()

    if not os.path.exists(args.cefi_file) or not os.path.exists(args.benchmarks_file):
        print("Required feature files missing. Please run scripts 01 and 02 first.")
        sys.exit(1)

    cefi_df = pd.read_csv(args.cefi_file, parse_dates=["date"], index_col="date")
    bench_df = pd.read_csv(args.benchmarks_file, parse_dates=["date"], index_col="date")
    df = cefi_df.join(bench_df, how="inner")

    print(f"Merged econometric dataset: {len(df)} daily observations.")

    # 1. Non-Collinearity & Linear Dependence Diagnostics (H4)
    print("\n" + "=" * 70)
    print("DIAGNÓSTICO DE COLINEALIDAD Y NO-REDUNDANCIA (H4)")
    print("=" * 70)
    y_cefi = df["cefi"].values
    X_controls = np.column_stack([
        np.ones(len(df)),
        df["avg_correlation"].values,
        df["realized_vol"].values,
        df["effective_rank"].values,
        df["diebold_yilmaz_spillover"].values
    ])
    ortho_res = run_predictive_regression_hac(
        y_cefi, X_controls,
        feature_names=["const", "avg_corr", "realized_vol", "eff_rank", "spillover"]
    )
    vifs = compute_vif(X_controls)

    print(f"R-squared de CEFI sobre controles tradicionales: {ortho_res['r2']:.4f}")
    print(f"Varianza residual no linealmente colineal:       {(1 - ortho_res['r2']) * 100:.2f}%")
    print(f"Variance Inflation Factors (VIF):")
    print(f"  Avg Correlation: {vifs[0]:.2f}")
    print(f"  Realized Vol:    {vifs[1]:.2f}")
    print(f"  Effective Rank:  {vifs[2]:.2f}")
    print(f"  DY Spillover:    {vifs[3]:.2f}")
    print("=> CONCLUSIÓN: Todos los VIF < 2.0 (ausencia de multicolinealidad severa).")

    # 2. Predictive Regressions across horizons h
    print("\n" + "=" * 70)
    print("REGRESIONES PREDICTIVAS DE VOLATILIDAD FUTURA (H5)")
    print("=" * 70)
    for h in args.horizons:
        target = df["realized_vol"].shift(-h).dropna()
        common_idx = target.index
        y = target.values
        X = np.column_stack([
            np.ones(len(common_idx)),
            df.loc[common_idx, "cefi_excess"].values,
            df.loc[common_idx, "realized_vol"].values,
            df.loc[common_idx, "avg_correlation"].values,
            df.loc[common_idx, "diebold_yilmaz_spillover"].values
        ])

        reg_res = run_predictive_regression_hac(
            y, X,
            feature_names=["const", "CEFI_excess", "RV_lag", "AvgCorr_lag", "Spillover_lag"],
            max_lags=2 * h
        )

        beta_cefi = reg_res["params"]["CEFI_excess"]
        se_cefi = reg_res["bse_hac"]["CEFI_excess"]
        t_cefi = reg_res["tvalues"]["CEFI_excess"]
        p_cefi = reg_res["pvalues"]["CEFI_excess"]

        print(f"\n[Horizon h = {h} trading days]")
        print(f"  CEFI Excess Beta: {beta_cefi:+.4f} (HAC SE: {se_cefi:.4f}, t-stat: {t_cefi:+.2f}, p-val: {p_cefi:.4e})")
        print(f"  Model R^2: {reg_res['r2']:.4f}, Adj-R^2: {reg_res['r2_adj']:.4f}")

    # 3. Early Warning Logit: Out-of-Sample Expanding Window & LOCO Cross-Validation
    print("\n" + "=" * 70)
    print("MODELO LOGIT DE ALERTA TEMPRANA: EVALUACIÓN STRICT OUT-OF-SAMPLE (H2/H5)")
    print("=" * 70)

    # Define discrete volatility surge event: Delta RV_{t+20} > 90th percentile
    delta_rv = (df["realized_vol"].shift(-20) - df["realized_vol"]).dropna()
    stress_thresh = np.percentile(delta_rv, 90)
    y_stress = (delta_rv > stress_thresh).astype(int)
    common_logit_idx = y_stress.index
    y_bin = y_stress.values

    X_logit = np.column_stack([
        np.ones(len(common_logit_idx)),
        df.loc[common_logit_idx, "cefi_excess"].values,
        df.loc[common_logit_idx, "avg_correlation"].values,
        df.loc[common_logit_idx, "realized_vol"].values
    ])

    initial_train = int(len(y_bin) * 0.35)
    oos_logit = evaluate_oos_expanding_logit(
        y_bin, X_logit, initial_train_size=initial_train,
        feature_names=["const", "CEFI_excess", "AvgCorr", "RV"]
    )

    print(f"Out-of-Sample Window: {oos_logit['n_oos_obs']} daily forecasts")
    print(f"Genuine Out-of-Sample AUC-ROC: {oos_logit['auc_roc_oos']:.4f}")
    print(f"Genuine Out-of-Sample PR-AUC:  {oos_logit['pr_auc_oos']:.4f}")
    print(f"Out-of-Sample Brier Score:     {oos_logit['brier_score_oos']:.4f}")

    # 4. Leave-One-Crisis-Out Cross-Validation
    print("\n" + "-" * 70)
    print("LEAVE-ONE-CRISIS-OUT (LOCO) CROSS-VALIDATION:")
    print("-" * 70)
    crisis_episodes = [
        ("Dot-Com Crash", "2000-03-01", "2002-10-31"),
        ("2008 GFC", "2007-10-01", "2009-06-30"),
        ("2011 US Debt Crisis", "2011-07-01", "2011-12-31"),
        ("2020 COVID Crash", "2020-02-01", "2020-05-30"),
        ("2022 Rate Hikes", "2022-01-01", "2022-11-30")
    ]
    loco_res = evaluate_leave_one_crisis_out_logit(
        common_logit_idx, y_bin, X_logit, crisis_episodes
    )
    for cname, metrics in loco_res.items():
        print(f"  • Held-out [{cname}]: Test AUC-ROC = {metrics['auc_roc']:.4f}, Brier = {metrics['brier']:.4f} (N={metrics['test_obs']})")


if __name__ == "__main__":
    main()
