#!/usr/bin/env python3
"""
Script 00: Download and Prepare Primary Financial Datasets
=========================================================
Downloads Kenneth French 30 Industry Portfolios and prepares data matrices.
"""

import os
import sys
import argparse

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from utils.data_fetcher import download_fama_french_30_industry


def main():
    parser = argparse.ArgumentParser(description="Download Fama-French 30 Industry Portfolios data.")
    parser.add_argument("--output-dir", type=str, default="data/raw", help="Directory to store raw data.")
    parser.add_argument("--start-year", type=int, default=1990, help="Start year for historical data.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "ff30_daily_returns.csv")

    print(f"Downloading Fama-French 30 Industry Portfolios (Daily, >= {args.start_year})...")
    df = download_fama_french_30_industry(start_year=args.start_year)
    df.to_csv(out_path)
    print(f"Successfully saved {len(df)} daily observations to: {out_path}")
    print(f"Date range: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")
    print(f"Industry assets (p={df.shape[1]}): {list(df.columns)}")

    # Also download macro controls (VIX, FRED spreads)
    from utils.data_fetcher import download_macro_controls
    try:
        macro_df = download_macro_controls(start_year=args.start_year)
        macro_path = os.path.join(args.output_dir, "macro_controls.csv")
        macro_df.to_csv(macro_path)
        print(f"Successfully saved macro controls ({len(macro_df)} obs) to: {macro_path}")
    except Exception as e:
        print(f"Note: Macro controls download encountered: {e}. Skipping or will retry later.")



if __name__ == "__main__":
    main()
