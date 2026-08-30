#!/usr/bin/env python3
"""
Script 03: Non-Parametric Stationary Block Bootstrap Inference
==============================================================
Calculates 95% bootstrap confidence bands for CEFI_t and conducts surrogate tests.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from econometrics.bootstrap_inference import stationary_block_bootstrap_cefi


def main():
    parser = argparse.ArgumentParser(description="Run stationary block bootstrap for CEFI.")
    parser.add_argument("--input-file", type=str, default="data/raw/ff30_daily_returns.csv")
    parser.add_argument("--output-file", type=str, default="data/features/cefi_bootstrap_ci.csv")
    parser.add_argument("--window", type=int, default=500)
    parser.add_argument("--n-bootstraps", type=int, default=100)
    parser.add_argument("--sample-step", type=int, default=50, help="Evaluate bootstrap every N days.")
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: Input file {args.input_file} not found.")
        sys.exit(1)

    returns_df = pd.read_csv(args.input_file, parse_dates=["Date"], index_col="Date")
    T_total, p = returns_df.shape
    dates = returns_df.index

    print(f"Computing Stationary Block Bootstrap ({args.n_bootstraps} reps) every {args.sample_step} days...")
    bootstrap_rows = []

    for t_end in range(args.window, T_total + 1, args.sample_step):
        date_t = dates[t_end - 1]
        window_slice = returns_df.iloc[t_end - args.window : t_end].values

        point_est, ci_low, ci_high = stationary_block_bootstrap_cefi(
            window_slice,
            n_bootstraps=args.n_bootstraps,
            mean_block_length=20
        )

        bootstrap_rows.append({
            "date": date_t,
            "cefi_point": point_est,
            "ci_lower_95": ci_low,
            "ci_upper_95": ci_high
        })
        print(f"  [{date_t.date()}] CEFI: {point_est:+.4f} | 95% CI: [{ci_low:+.4f}, {ci_high:+.4f}]")

    df_out = pd.DataFrame(bootstrap_rows).set_index("date")
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    df_out.to_csv(args.output_file)
    print(f"\nSuccessfully saved bootstrap results to: {args.output_file}")


if __name__ == "__main__":
    main()
