#!/usr/bin/env python3
"""
Script 01: Compute Historical Dynamic Causal Emergence Financial Index (CEFI)
=============================================================================
Runs rolling-window Stiefel manifold optimization with exact conditional projection,
scale-invariant interventions, and excess emergence over finite-sample null baseline.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from causal_emergence.dynamic_pipeline import compute_dynamic_cefi_series


def main():
    parser = argparse.ArgumentParser(description="Compute CEFI_t, Excess CEFI_t, and q_t^* series.")
    parser.add_argument("--input-file", type=str, default="data/raw/ff30_daily_returns.csv", help="Input returns CSV.")
    parser.add_argument("--output-file", type=str, default="data/features/cefi_daily_series.csv", help="Output features CSV.")
    parser.add_argument("--window", type=int, default=500, help="Rolling window size in trading days.")
    parser.add_argument("--step", type=int, default=2, help="Step size for rolling window (default=2 days).")
    parser.add_argument("--kappa-do", type=float, default=1.0, help="Dimensionless intervention strength.")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Number of CPU cores for parallel execution.")
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: Input file {args.input_file} not found. Please run scripts/00_download_data.py first.")
        sys.exit(1)

    print(f"Loading data from {args.input_file}...")
    returns_df = pd.read_csv(args.input_file, parse_dates=["Date"], index_col="Date")
    print(f"Dataset shape: {returns_df.shape} ({returns_df.index.min().date()} to {returns_df.index.max().date()})")

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    print(f"Computing scale-invariant CEFI series (window={args.window}, step={args.step}, kappa_do={args.kappa_do}, n_jobs={args.n_jobs})...")

    # Complete dimensional spectrum q in 1..29
    p = returns_df.shape[1]
    q_all = list(range(1, p))

    # Calculate CEFI across rolling windows in parallel
    cefi_df = compute_dynamic_cefi_series(
        returns_df,
        window_length=args.window,
        step_size=args.step,
        q_candidates=q_all,
        kappa_do=args.kappa_do,
        n_jobs=args.n_jobs
    )

    cefi_df.to_csv(args.output_file)
    print(f"\nSuccessfully generated CEFI series! Saved to: {args.output_file}")
    print(f"Total time steps: {len(cefi_df)}")
    print(f"Mean Raw CEFI Density:   {cefi_df['cefi'].mean():.4f}, Max: {cefi_df['cefi'].max():.4f}")
    print(f"Mean Excess CEFI (H0):   {cefi_df['cefi_excess'].mean():.4f}, Max: {cefi_df['cefi_excess'].max():.4f}")
    print(f"Positive Excess Freq:    {(cefi_df['cefi_excess'] > 0).mean() * 100:.2f}%")
    print(f"Modal Causal Dimension:  q* = {cefi_df['q_star'].mode()[0]}")


if __name__ == "__main__":
    main()
