#!/usr/bin/env python3
"""
Script 06: Generate Publication-Ready Figures and LaTeX Tables
============================================================
Exports formatted LaTeX tables and PDF/PNG vector figures (300 DPI) for Q1 manuscript submission.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from econometrics.predictive_regressions import run_predictive_regression_hac
from econometrics.early_warning_logit import (
    evaluate_early_warning_classifier,
    evaluate_oos_expanding_logit,
    evaluate_leave_one_crisis_out_logit,
    sigmoid
)
from econometrics.oos_forecasting import run_expanding_window_oos


# Publication aesthetics
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "lines.linewidth": 1.5,
    "grid.alpha": 0.3,
    "grid.linestyle": "--"
})


def generate_all_latex_tables(df: pd.DataFrame, table_dir: str):
    """Generates Tables 1 to 5 formatted for Academic LaTeX."""
    print("Generating Academic LaTeX Tables...")

    # Table 1: Descriptive Statistics
    desc_cols = ["cefi_excess", "cefi", "q_star", "realized_vol", "avg_correlation", "effective_rank", "diebold_yilmaz_spillover"]
    col_names = ["CEFI Excess ($H_0$)", "CEFI Raw Density", "Causal Dim ($q^*$)", "Realized Vol", "Avg Correlation", "Effective Rank", "DY Spillover (%)"]
    sub_df = df[desc_cols].copy()
    sub_df.columns = col_names
    stats = sub_df.describe().T[["mean", "std", "min", "50%", "max"]]
    stats.columns = ["Mean", "Std Dev", "Min", "Median", "Max"]

    t1_path = os.path.join(table_dir, "table1_descriptive_stats.tex")
    with open(t1_path, "w") as f:
        f.write("% Table 1: Summary Statistics\n")
        f.write(stats.to_latex(float_format="%.4f", caption="Descriptive Statistics of Financial Market Dynamics and Scale-Invariant Causal Emergence Measures (1992--2026)", label="tab:desc_stats"))
    print(f"  -> Saved: {t1_path}")

    # Table 2: Linear Non-Collinearity Test (H4)
    y_cefi = df["cefi_excess"].values
    X_ortho = np.column_stack([
        np.ones(len(df)),
        df["avg_correlation"].values,
        df["realized_vol"].values,
        df["effective_rank"].values,
        df["diebold_yilmaz_spillover"].values
    ])
    ortho_res = run_predictive_regression_hac(
        y_cefi, X_ortho,
        feature_names=["Intercept", "Avg Correlation ($\\bar{\\rho}$)", "Realized Volatility ($RV$)", "Effective Rank", "Spillover Index ($DY$)"]
    )
    t2_rows = []
    for var, coef in ortho_res["params"].items():
        se = ortho_res["bse_hac"][var]
        t = ortho_res["tvalues"][var]
        p = ortho_res["pvalues"][var]
        t2_rows.append({
            "Variable": var,
            "Coefficient": f"{coef:+.4f}",
            "HAC Std Error": f"({se:.4f})",
            "t-statistic": f"{t:+.2f}",
            "p-value": f"{p:.4e}"
        })
    t2_df = pd.DataFrame(t2_rows).set_index("Variable")
    t2_path = os.path.join(table_dir, "table2_collinearity_test.tex")
    with open(t2_path, "w") as f:
        f.write(f"% Table 2: Non-Collinearity Diagnostics (R^2 = {ortho_res['r2']:.4f}, Adj-R^2 = {ortho_res['r2_adj']:.4f})\n")
        f.write(t2_df.to_latex(caption="Linear Non-Collinearity Diagnostics: Regressing Excess CEFI on Standard Market Benchmarks", label="tab:collinearity"))
    print(f"  -> Saved: {t2_path}")

    # Table 3: In-Sample Predictive Regressions (H5)
    t3_rows = []
    for h in [5, 20, 60]:
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
            y, X, feature_names=["const", "CEFI_excess", "RV_lag", "AvgCorr_lag", "Spillover_lag"], max_lags=2 * h
        )
        t3_rows.append({
            "Horizon (h)": f"{h} days",
            "CEFI Excess Beta": f"{reg_res['params']['CEFI_excess']:+.4f}",
            "HAC SE": f"({reg_res['bse_hac']['CEFI_excess']:.4f})",
            "t-stat": f"{reg_res['tvalues']['CEFI_excess']:+.2f}",
            "p-val": f"{reg_res['pvalues']['CEFI_excess']:.4e}",
            "R-squared": f"{reg_res['r2']:.4f}",
            "Adj R-squared": f"{reg_res['r2_adj']:.4f}"
        })
    t3_df = pd.DataFrame(t3_rows).set_index("Horizon (h)")
    t3_path = os.path.join(table_dir, "table3_predictive_regressions.tex")
    with open(t3_path, "w") as f:
        f.write(t3_df.to_latex(caption="In-Sample Predictive Regressions of Future Realized Volatility on Excess CEFI and Market Controls", label="tab:predictive_reg"))
    print(f"  -> Saved: {t3_path}")

    # Table 4: Out-of-Sample Multi-Horizon Forecasting Evaluation
    t4_rows = []
    for h in [5, 20, 60]:
        target = df["realized_vol"].shift(-h).dropna()
        common_idx = target.index
        y = target.values
        X_base = np.column_stack([
            np.ones(len(common_idx)),
            df.loc[common_idx, "realized_vol"].values,
            df.loc[common_idx, "avg_correlation"].values,
            df.loc[common_idx, "diebold_yilmaz_spillover"].values
        ])
        X_ext = np.column_stack([X_base, df.loc[common_idx, "cefi_excess"].values])
        oos_res = run_expanding_window_oos(y, X_base, X_ext, initial_train_size=int(len(y)*0.35))
        t4_rows.append({
            "Horizon": f"{h}d",
            "RMSE Base": f"{oos_res['rmse_base']:.5f}",
            "RMSE +CEFI": f"{oos_res['rmse_ext']:.5f}",
            "Ratio": f"{oos_res['rmse_ratio']:.4f}",
            "Clark-West Stat": f"{oos_res['cw_stat']:+.3f}",
            "CW p-val": f"{oos_res['cw_pvalue']:.4f}",
            "Diebold-Mariano Stat": f"{oos_res['dm_stat']:+.3f}",
            "DM p-val": f"{oos_res['dm_pvalue']:.4f}"
        })
    t4_df = pd.DataFrame(t4_rows).set_index("Horizon")
    t4_path = os.path.join(table_dir, "table4_oos_forecasting.tex")
    with open(t4_path, "w") as f:
        f.write(t4_df.to_latex(caption="Out-of-Sample Pseudo-Real-Time Forecasting Accuracy across Multiple Horizons", label="tab:oos_eval"))
    print(f"  -> Saved: {t4_path}")


def generate_all_figures(df: pd.DataFrame, fig_dir: str):
    """Generates Figure 2 to 5 as PDF and 300 DPI PNG."""
    print("Generating Publication Figures...")

    # Figure 2: Historical Dynamics of Excess CEFI_t and Financial Crises
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})

    ax1.plot(df.index, df["cefi_excess"], color="#1f77b4", label=r"$\mathrm{CEFI}_t^{\mathrm{excess}}$ (Over $H_0$ Null Bias)", lw=1.6)
    ax1.axhline(0, color="black", linestyle=":", alpha=0.6)
    ax1.set_ylabel(r"$\mathrm{CEFI}_t^{\mathrm{excess}}$ (nats / dim)", fontweight="bold")
    ax1.set_title("Historical Dynamics of Scale-Invariant Causal Emergence in Financial Markets (1992–2026)", fontweight="bold")
    ax1.grid(True)

    crisis_periods = [
        ("Dot-Com Bubble", "2000-03-01", "2002-10-01", "#ffebee"),
        ("2008 GFC", "2007-10-01", "2009-06-30", "#ffcdd2"),
        ("2011 US Debt Downgrade", "2011-07-01", "2011-12-31", "#ffebee"),
        ("2020 COVID Crash", "2020-02-01", "2020-05-30", "#ef9a9a"),
        ("2022 Rate Hikes", "2022-01-01", "2022-11-30", "#ffebee")
    ]
    for label, start, end, col in crisis_periods:
        ax1.axvspan(pd.to_datetime(start), pd.to_datetime(end), color=col, alpha=0.6, label=f"Crisis: {label}" if "GFC" in label or "COVID" in label else "")
        ax2.axvspan(pd.to_datetime(start), pd.to_datetime(end), color=col, alpha=0.6)

    ax1.legend(loc="upper left", frameon=True)

    # Bottom panel: Causal Effective Dimension q_t*
    ax2.plot(df.index, df["q_star"], color="#d62728", lw=1.4, label=r"Causal Effective Dimension ($q_t^*$)")
    ax2.set_ylabel(r"Dimension $q_t^*$", fontweight="bold")
    ax2.set_xlabel("Year", fontweight="bold")
    ax2.set_yticks([1, 2, 3, 4, 5, 10, 15, 20, 25, 29])
    ax2.grid(True)
    ax2.legend(loc="upper left", frameon=True)

    fig.tight_layout()
    f2_pdf = os.path.join(fig_dir, "figure2_cefi_dynamics_crises.pdf")
    f2_png = os.path.join(fig_dir, "figure2_cefi_dynamics_crises.png")
    fig.savefig(f2_pdf, bbox_inches="tight")
    fig.savefig(f2_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Saved: {f2_pdf} & {f2_png}")

    # Figure 3: Scatter of CEFI vs Average Correlation
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(df["avg_correlation"], df["cefi_excess"], c=df["realized_vol"] * 100, cmap="viridis", alpha=0.6, s=15)
    cb = fig.colorbar(scatter, ax=ax)
    cb.set_label("Realized Volatility (%)", fontweight="bold")
    ax.set_xlabel(r"Average Cross-Sectional Correlation ($\bar{\rho}_t$)", fontweight="bold")
    ax.set_ylabel(r"Excess Causal Emergence Index ($\mathrm{CEFI}_t^{\mathrm{excess}}$)", fontweight="bold")
    ax.set_title(r"Non-Collinearity Analysis: $\mathrm{CEFI}_t^{\mathrm{excess}}$ vs. Average Pairwise Correlation", fontweight="bold")
    ax.grid(True)
    fig.tight_layout()
    f3_pdf = os.path.join(fig_dir, "figure3_orthogonality_scatter.pdf")
    f3_png = os.path.join(fig_dir, "figure3_orthogonality_scatter.png")
    fig.savefig(f3_pdf, bbox_inches="tight")
    fig.savefig(f3_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Saved: {f3_pdf} & {f3_png}")

    # Figure 4: Genuine Out-of-Sample ROC Curve
    delta_rv = (df["realized_vol"].shift(-20) - df["realized_vol"]).dropna()
    stress_thresh = np.percentile(delta_rv, 90)
    y_stress = (delta_rv > stress_thresh).astype(int)
    common_idx = y_stress.index

    X_logit = np.column_stack([
        np.ones(len(common_idx)),
        df.loc[common_idx, "cefi_excess"].values,
        df.loc[common_idx, "avg_correlation"].values,
        df.loc[common_idx, "realized_vol"].values
    ])
    oos_logit = evaluate_oos_expanding_logit(
        y_stress.values, X_logit, initial_train_size=int(len(y_stress)*0.35),
        feature_names=["const", "CEFI_excess", "AvgCorr", "RV"]
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    thresholds = np.linspace(0, 1, 150)
    tprs, fprs = [], []
    y_act = oos_logit["y_actual_oos"]
    p_hat = oos_logit["p_hat_oos"]

    for th in thresholds:
        tp = np.sum((p_hat >= th) & (y_act == 1))
        fp = np.sum((p_hat >= th) & (y_act == 0))
        fn = np.sum((p_hat < th) & (y_act == 1))
        tn = np.sum((p_hat < th) & (y_act == 0))
        tprs.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        fprs.append(fp / (fp + tn) if (fp + tn) > 0 else 0)

    ax.plot(fprs, tprs, color="#2ca02c", lw=2.2, label=f"Expanding Window OOS Model (AUC = {oos_logit['auc_roc_oos']:.4f})")
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", label="Random Classifier (AUC = 0.5000)")
    ax.set_xlabel("False Positive Rate", fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontweight="bold")
    ax.set_title("Genuine Out-of-Sample ROC Curve for Volatility Surge ($h=20$ days)", fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True)
    fig.tight_layout()
    f4_pdf = os.path.join(fig_dir, "figure4_early_warning_roc.pdf")
    f4_png = os.path.join(fig_dir, "figure4_early_warning_roc.png")
    fig.savefig(f4_pdf, bbox_inches="tight")
    fig.savefig(f4_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Saved: {f4_pdf} & {f4_png}")


def main():
    parser = argparse.ArgumentParser(description="Generate publication figures and tables.")
    parser.add_argument("--cefi-file", type=str, default="data/features/cefi_daily_series.csv")
    parser.add_argument("--benchmarks-file", type=str, default="data/features/benchmarks_daily_series.csv")
    parser.add_argument("--output-dir", type=str, default="reports")
    args = parser.parse_args()

    table_dir = os.path.join(args.output_dir, "tables")
    figures_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(table_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    print(f"Loading feature files from {args.cefi_file} and {args.benchmarks_file}...")
    cefi_df = pd.read_csv(args.cefi_file, parse_dates=["date"], index_col="date")
    bench_df = pd.read_csv(args.benchmarks_file, parse_dates=["date"], index_col="date")
    df = cefi_df.join(bench_df, how="inner")

    generate_all_latex_tables(df, table_dir)
    generate_all_figures(df, figures_dir)
    print("\nALL PUBLICATION ASSETS (TABLES & FIGURES) REGENERATED WITH Q1 RIGOR!")


if __name__ == "__main__":
    main()
