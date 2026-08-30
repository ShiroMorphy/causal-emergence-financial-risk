#!/usr/bin/env python3
"""
Script 05: Out-of-Sample Pseudo-Real-Time Forecasting Evaluation
===============================================================
Performs expanding-window out-of-sample forecasts across multiple horizons (h=5, 20, 60)
and runs Clark-West & Diebold-Mariano tests.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from econometrics.oos_forecasting import run_expanding_window_oos


def main():
    parser = argparse.ArgumentParser(description="Run Out-of-Sample forecasting tests.")
    parser.add_argument("--cefi-file", type=str, default="data/features/cefi_daily_series.csv")
    parser.add_argument("--benchmarks-file", type=str, default="data/features/benchmarks_daily_series.csv")
    parser.add_argument("--train-ratio", type=float, default=0.35, help="Proportion of sample for initial training.")
    parser.add_argument("--horizons", nargs="+", type=int, default=[5, 20, 60], help="Forecast horizons in days.")
    args = parser.parse_args()

    if not os.path.exists(args.cefi_file) or not os.path.exists(args.benchmarks_file):
        print("Required feature files missing.")
        sys.exit(1)

    cefi_df = pd.read_csv(args.cefi_file, parse_dates=["date"], index_col="date")
    bench_df = pd.read_csv(args.benchmarks_file, parse_dates=["date"], index_col="date")
    df = cefi_df.join(bench_df, how="inner")

    print("\n" + "=" * 80)
    print("EVALUACIÓN PREDICTIVA OUT-OF-SAMPLE (OOS) MULTI-HORIZONTE")
    print("=" * 80)

    summary_rows = []

    for h in args.horizons:
        target = df["realized_vol"].shift(-h).dropna()
        common_idx = target.index
        y = target.values

        # Base features: const, RV_lag, AvgCorr_lag, Spillover_lag
        X_base = np.column_stack([
            np.ones(len(common_idx)),
            df.loc[common_idx, "realized_vol"].values,
            df.loc[common_idx, "avg_correlation"].values,
            df.loc[common_idx, "diebold_yilmaz_spillover"].values
        ])

        # Extended features: Base + CEFI_excess
        X_ext = np.column_stack([
            X_base,
            df.loc[common_idx, "cefi_excess"].values
        ])

        initial_train_size = int(len(y) * args.train_ratio)
        oos_res = run_expanding_window_oos(y, X_base, X_ext, initial_train_size=initial_train_size)

        pct_improvement = (1.0 - oos_res["rmse_ratio"]) * 100.0

        print(f"\n[Horizonte h = {h} días | OOS Obs: {oos_res['n_oos_obs']}]")
        print(f"  RMSE Base:     {oos_res['rmse_base']:.5f}")
        print(f"  RMSE +CEFI:    {oos_res['rmse_ext']:.5f} (Ratio: {oos_res['rmse_ratio']:.4f}, Mejora: {pct_improvement:+.2f}%)")
        print(f"  MAE Base:      {oos_res['mae_base']:.5f}")
        print(f"  MAE +CEFI:     {oos_res['mae_ext']:.5f}")
        print(f"  Clark-West:    CW = {oos_res['cw_stat']:+.4f} (p-value: {oos_res['cw_pvalue']:.4e})")
        print(f"  Diebold-Mar:   DM = {oos_res['dm_stat']:+.4f} (p-value: {oos_res['dm_pvalue']:.4e})")

        summary_rows.append({
            "Horizon": f"{h}d",
            "RMSE_Base": oos_res["rmse_base"],
            "RMSE_Ext": oos_res["rmse_ext"],
            "Ratio": oos_res["rmse_ratio"],
            "Pct_Improvement": pct_improvement,
            "CW_stat": oos_res["cw_stat"],
            "CW_pval": oos_res["cw_pvalue"],
            "DM_stat": oos_res["dm_stat"],
            "DM_pval": oos_res["dm_pvalue"]
        })

    summary_df = pd.DataFrame(summary_rows)
    print("\n" + "-" * 80)
    print("RESUMEN OOS FINAL:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
