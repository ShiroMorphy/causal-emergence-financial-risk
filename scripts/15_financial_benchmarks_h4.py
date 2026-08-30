#!/usr/bin/env python3
"""
Script 15: Financial Benchmark Regressions and Orthogonality Tests (H4)
======================================================================
Tests the relationship and incremental information content of CEFI_t relative to:
1. Realized Volatility (RV_t)
2. Average Cross-Sectional Correlation (rho_bar_t)
3. Effective Rank (ER_t, Roy & Vetterli 2007)
4. Diebold-Yilmaz Connectedness Index (DY_t)
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from econometrics.predictive_regressions import run_predictive_regression_hac


def main():
    cefi_df = pd.read_csv("data/features/cefi_daily_series.csv", parse_dates=["date"], index_col="date")
    bench_df = pd.read_csv("data/features/benchmarks_daily_series.csv", parse_dates=["date"], index_col="date")
    df = cefi_df.join(bench_df, how="inner").dropna()

    y = df["cefi"].values
    X_vars = ["realized_vol", "avg_correlation", "effective_rank", "diebold_yilmaz_spillover"]
    feature_names = ["Realized Volatility (RV)", "Average Correlation (rho)", "Effective Rank (ER)", "Diebold-Yilmaz Index (DY)"]


    # Pairwise correlations
    corrs = {}
    for var, name in zip(X_vars, feature_names):
        corrs[name] = float(np.corrcoef(df[var], y)[0, 1])

    # Multivariate HAC Regression: CEFI_t on all standard systemic risk proxies
    X_mat = np.column_stack([np.ones(len(df))] + [df[v].values for v in X_vars])
    reg_res = run_predictive_regression_hac(y, X_mat, feature_names=["const"] + feature_names, max_lags=40)

    # Compute R2 and Orthogonal Residual Variance
    lr = LinearRegression().fit(df[X_vars], y)
    r2 = lr.score(df[X_vars], y)
    ortho_var_pct = (1.0 - r2) * 100.0

    print("\n" + "=" * 90)
    print("TEST FORMAL H4: CONTENIDO INCREMENTAL DE CEFI FRENTE A BENCHMARKS TRADICIONALES")
    print("=" * 90)
    print(f"R^2 con Benchmarks Tradicionales:       {r2 * 100:.2f}%")
    print(f"Varianza Ortogonal / No Redundante:     {ortho_var_pct:.2f}%\n")

    summary_rows = []
    for var, name in zip(X_vars, feature_names):
        summary_rows.append({
            "Benchmark Proxy": name,
            "Pairwise Corr vs CEFI": f"{corrs[name]:+.3f}",
            "Multivariate Beta": f"{reg_res['params'][name]:+.4f}",
            "HAC SE": f"({reg_res['bse_hac'][name]:.4f})",
            "t-statistic": f"{reg_res['tvalues'][name]:+.2f}",
            "p-value": f"{reg_res['pvalues'][name]:.4e}"
        })

    sum_df = pd.DataFrame(summary_rows).set_index("Benchmark Proxy")
    print(sum_df.to_string())

    tex_path = "reports/tables/table_financial_benchmarks_h4.tex"
    os.makedirs(os.path.dirname(tex_path), exist_ok=True)
    with open(tex_path, "w") as f_out:
        f_out.write("% Financial Benchmarks and Collinearity Test (H4)\n")
        f_out.write(sum_df.to_latex(caption="Relationship Between CEFI and Conventional Systemic Risk Benchmarks (HAC Newey-West $L=40$)", label="tab:benchmarks_h4"))

    print(f"\nResultados guardados en: {tex_path}")


if __name__ == "__main__":
    main()
