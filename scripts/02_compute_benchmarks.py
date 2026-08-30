#!/usr/bin/env python3
"""
Script 02: Compute Comparative Financial Benchmarks
===================================================
Extracts rolling Realized Volatility, Average Correlation, First-PC ratio,
Effective Rank, and Network Spillover indices.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from benchmarks.volatility_measures import (
    compute_realized_volatility,
    compute_average_correlation,
    compute_first_pc_variance_ratio
)
from benchmarks.spectral_measures import (
    compute_effective_rank,
    compute_spectral_entropy
)
from benchmarks.network_measures import (
    compute_diebold_yilmaz_index,
    compute_granger_network_density
)


def main():
    parser = argparse.ArgumentParser(description="Compute benchmark indicators.")
    parser.add_argument("--input-file", type=str, default="data/raw/ff30_daily_returns.csv", help="Input returns CSV.")
    parser.add_argument("--output-file", type=str, default="data/features/benchmarks_daily_series.csv", help="Output benchmarks CSV.")
    parser.add_argument("--window", type=int, default=500, help="Rolling window size in trading days.")
    parser.add_argument("--step", type=int, default=1, help="Step size for rolling window.")
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: Input file {args.input_file} not found.")
        sys.exit(1)

    print(f"Loading data from {args.input_file}...")
    returns_df = pd.read_csv(args.input_file, parse_dates=["Date"], index_col="Date")
    T_total, p = returns_df.shape
    dates = returns_df.index

    print(f"Computing rolling benchmarks over {T_total} trading days (window={args.window})...")
    benchmark_rows = []

    for t_end in range(args.window, T_total + 1, args.step):
        date_t = dates[t_end - 1]
        window_slice = returns_df.iloc[t_end - args.window : t_end].values

        # Market-wide equal-weighted return for realized volatility benchmark
        market_return = np.mean(window_slice, axis=1)

        row = {
            "date": date_t,
            "realized_vol": compute_realized_volatility(market_return),
            "avg_correlation": compute_average_correlation(window_slice),
            "first_pc_ratio": compute_first_pc_variance_ratio(window_slice),
            "effective_rank": compute_effective_rank(window_slice),
            "spectral_entropy": compute_spectral_entropy(window_slice),
            "diebold_yilmaz_spillover": compute_diebold_yilmaz_index(window_slice),
            "granger_density": compute_granger_network_density(window_slice)
        }
        benchmark_rows.append(row)

    df_out = pd.DataFrame(benchmark_rows).set_index("date")
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    df_out.to_csv(args.output_file)
    print(f"Successfully computed benchmarks! Saved to: {args.output_file}")


if __name__ == "__main__":
    main()
