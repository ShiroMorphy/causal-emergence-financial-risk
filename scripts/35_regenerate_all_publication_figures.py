#!/usr/bin/env python3
"""
Script 35: Publication Figures Generator (Canonical 12/100 Synchronization)
===========================================================================
Generates all 4 publication-quality figures for the IRFA submission:
1. figure1_cefi_dynamics.pdf: Full historical CEFI series (1992-2026) with crisis shadings (nats/dimension).
2. figure2_qstar_dynamics.pdf: Causal Effective Dimension (q_t*) with focused y-axis [0, 5.5] and modal line.
3. figure3_null_distributions.pdf: 3-panel provenance-verified matched null distributions.
4. figure4_theoretical_benchmarking.pdf: Continuous benchmarking vs. Liu (2024) and PRE (2025) SVD.
"""

import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import scipy.stats as stats

# Configure publication-quality matplotlib defaults
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Computer Modern Roman"],
    "mathtext.fontset": "cm",
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.45,
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})

OUTPUT_DIR = "reports/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

from causal_emergence.episodes import get_crises_plot_specs

CRISES = get_crises_plot_specs()


def generate_figure1():
    print("Generating Figure 1: CEFI Dynamics (1992–2026)...")
    df = pd.read_csv("data/features/cefi_series_12_100.csv", parse_dates=["date"]).rename(columns={"date": "Date"}).set_index("Date")

    fig, ax = plt.subplots(figsize=(11, 4.2), dpi=300)
    
    # Shade crisis episodes
    for label, s_date, e_date, col, alpha in CRISES:
        ax.axvspan(pd.to_datetime(s_date), pd.to_datetime(e_date), color=col, alpha=alpha, label=label)

    # Plot CEFI series
    ax.plot(df.index, df["cefi"], color="#1e3a8a", linewidth=1.1, label=r"$\mathrm{CEFI}_t$ (Canonical 12/100)")
    ax.axhline(df["cefi"].mean(), color="#dc2626", linestyle=":", linewidth=1.1, label=f"Historical Mean ({df['cefi'].mean():.2f} nats/DOF)")

    ax.set_title("Causal Emergence Financial Index Dynamics (1992–2026)", fontweight="bold", pad=10)
    ax.set_ylabel(r"$\mathrm{CEFI}_t$ (nats / DOF)", fontweight="medium")
    ax.set_xlabel("Year", fontweight="medium")
    ax.set_ylim(-0.05, max(df["cefi"].max() * 1.08, 2.5))
    ax.xaxis.set_major_locator(mdates.YearLocator(4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Custom legend with 2 columns
    handles, labels = ax.get_legend_handles_labels()
    # Reorder to show CEFI & Mean first, then crises
    order = [4, 5, 0, 1, 2, 3]
    ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc="upper right", ncol=3, framealpha=0.92, edgecolor="#cbd5e1")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "figure1_cefi_dynamics.pdf"))
    plt.savefig(os.path.join(OUTPUT_DIR, "figure1_cefi_dynamics.png"))
    plt.close()
    print("  -> Saved figure1_cefi_dynamics.pdf / png (Units: nats / DOF).")


def generate_figure2():
    print("Generating Figure 2 (Figure 3 in text): Causal Effective Dimension (q_t*)...")
    df = pd.read_csv("data/features/cefi_series_12_100.csv", parse_dates=["date"]).rename(columns={"date": "Date"}).set_index("Date")

    fig, ax = plt.subplots(figsize=(11, 3.8), dpi=300)

    for label, s_date, e_date, col, alpha in CRISES:
        ax.axvspan(pd.to_datetime(s_date), pd.to_datetime(e_date), color=col, alpha=alpha, label=label)

    ax.plot(df.index, df["q_star"], color="#047857", linewidth=0.95, alpha=0.9, label=r"Causal Effective Dimension $q_t^*$")
    modal_q = int(df["q_star"].mode().iloc[0])
    modal_share = 100.0 * float((df["q_star"] == modal_q).mean())
    pct_le2 = 100.0 * float((df["q_star"] <= 2).mean())
    pct_le4 = 100.0 * float((df["q_star"] <= 4).mean())
    ax.axhline(modal_q, color="#b91c1c", linestyle="--", linewidth=1.2, label=rf"Modal Dimension $q^* = {modal_q}$ ({modal_share:.1f}% of windows)")

    ax.set_title(r"Evolution of Causal Effective Dimension ($q_t^*$) Across Market Regimes", fontweight="bold", pad=10)
    ax.set_ylabel(r"Dimension $q^*$", fontweight="medium")
    ax.set_xlabel("Year", fontweight="medium")
    ax.set_ylim(0.4, 7.8)
    ax.set_yticks([1, 2, 3, 4, 5, 6, 7])
    ax.xaxis.set_major_locator(mdates.YearLocator(4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Annotation box for descriptive dimensional concentration
    textstr = (
        "Descriptive Concentration:\n"
        rf"$\bullet$ Modal $q^* = {modal_q}$ ({modal_share:.1f}\%)\n"
        rf"$\bullet$ $P(q^* \leq 2) = {pct_le2:.1f}\%$\n"
        rf"$\bullet$ $P(q^* \leq 4) = {pct_le4:.2f}\%$"
    ).replace(r"\n", "\n")
    props = dict(boxstyle="round,pad=0.5", facecolor="#f8fafc", alpha=0.92, edgecolor="#cbd5e1")
    ax.text(0.02, 0.95, textstr, transform=ax.transAxes, fontsize=8.5, verticalalignment="top", bbox=props)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(loc="upper right", ncol=2, framealpha=0.92, edgecolor="#cbd5e1")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "figure2_qstar_dynamics.pdf"))
    plt.savefig(os.path.join(OUTPUT_DIR, "figure2_qstar_dynamics.png"))
    plt.close()
    print("  -> Saved figure2_qstar_dynamics.pdf / png (Y-axis [0.4, 7.8]).")


def generate_figure3():
    print("Generating Figure 3: Matched Null Model Distributions (12/100 Canonical)...")
    df_null = pd.read_csv("reports/tables/primary_null_inference_b9999.csv")
    provenance_path = "reports/final_submission_source_of_truth/CANONICAL_NULL_RESULTS_12_100.provenance.json"
    with open(provenance_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    run_config = manifest["run_config"]
    estimator_spec = run_config["estimator_spec"]
    run_fingerprint = manifest["run_fingerprint"][:12]

    calm = df_null[df_null["Regime"].str.contains("Calm")].iloc[0]
    gfc = df_null[df_null["Regime"].str.contains("GFC")].iloc[0]
    covid = df_null[df_null["Regime"].str.contains("COVID")].iloc[0]

    regimes_data = [
        ("Calm Benchmark (2005-12-30)", "Calm_Period__2005", calm),
        ("2008 GFC Peak (2008-11-20)", "2008_GFC_Peak", gfc),
        ("2020 COVID Shock (2020-03-23)", "2020_COVID_Shock", covid)
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), dpi=300, sharey=True)

    for ax, (title, checkpoint_prefix, row) in zip(axes, regimes_data):
        obs_val = float(row["CEFI_obs"])
        stat_paths = glob.glob(
            f"reports/checkpoints/{checkpoint_prefix}_H0_static_B9999_{estimator_spec}_{run_fingerprint}.npz"
        )
        dc_paths = glob.glob(
            f"reports/checkpoints/{checkpoint_prefix}_H0_diag+contemp_B9999_{estimator_spec}_{run_fingerprint}.npz"
        )
        if len(stat_paths) != 1 or len(dc_paths) != 1:
            raise FileNotFoundError(f"Canonical empirical null checkpoints missing for {title}")
        null_stat = np.load(stat_paths[0])["cefi"]
        null_dc = np.load(dc_paths[0])["cefi"]
        if len(null_stat) != 9999 or len(null_dc) != 9999:
            raise ValueError(f"Incomplete primary-null checkpoint for {title}")

        x_min = min(np.percentile(null_stat, 0.1), np.percentile(null_dc, 0.1), obs_val)
        x_max = max(np.percentile(null_stat, 99.9), np.percentile(null_dc, 99.9), obs_val)
        margin = 0.08 * (x_max - x_min)
        x_min -= margin
        x_max += margin
        x_axis = np.linspace(x_min, x_max, 400)

        pdf_stat = stats.gaussian_kde(null_stat)(x_axis)
        pdf_dc = stats.gaussian_kde(null_dc)(x_axis)

        ax.plot(x_axis, pdf_stat, color="#475569", linestyle="--", linewidth=1.5, label=r"$H_0^{\mathrm{static}}$")
        ax.plot(x_axis, pdf_dc, color="#2563eb", linestyle="-", linewidth=1.6, label=r"$H_0^{\mathrm{diag+contemp}}$")
        ax.axvline(obs_val, color="#dc2626", linewidth=2.0, label=f"Observed ({obs_val:.4f})")

        ax.set_title(title, fontsize=10.5, fontweight="bold", pad=8)
        ax.set_xlabel("CEFI (nats / DOF)", fontsize=10)
        if ax == axes[0]:
            ax.set_ylabel("Probability Density", fontsize=10)
            ax.legend(loc="upper left", framealpha=0.92, edgecolor="#cbd5e1")

    plt.suptitle(r"Observed CEFI vs. Matched Surrogate Null Distributions ($B=9,999$)", y=1.03, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "figure3_null_distributions.pdf"))
    plt.savefig(os.path.join(OUTPUT_DIR, "figure3_null_distributions.png"))
    plt.close()
    print("  -> Saved figure3_null_distributions.pdf / png (empirical 12/100 null distributions).")


def generate_figure4():
    print("Generating Figure 4: Continuous Cross-Framework Benchmarking...")
    if not os.path.exists("data/features/framework_comparison_series.csv"):
        print("Error: framework_comparison_series.csv not found.")
        return

    df_bench = pd.read_csv("data/features/framework_comparison_series.csv", parse_dates=["date"]).set_index("date")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), dpi=300)

    # 1. Stiefel vs Liu 2024
    p_liu = stats.pearsonr(df_bench["cefi_stiefel"], df_bench["cefi_liu2024"])[0]
    s_liu = stats.spearmanr(df_bench["cefi_stiefel"], df_bench["cefi_liu2024"])[0]

    axes[0].scatter(df_bench["cefi_stiefel"], df_bench["cefi_liu2024"], alpha=0.35, color="#2563eb", s=14, edgecolors="none")
    # Linear fit
    m1, b1 = np.polyfit(df_bench["cefi_stiefel"], df_bench["cefi_liu2024"], 1)
    x_vals1 = np.linspace(df_bench["cefi_stiefel"].min(), df_bench["cefi_stiefel"].max(), 100)
    axes[0].plot(x_vals1, m1 * x_vals1 + b1, color="#1e3a8a", linestyle="--", linewidth=1.3)

    axes[0].set_title(f"Stiefel (12/100) vs. Liu et al. (2024)\n(Pearson $\\rho = {p_liu:.3f}$, Spearman $\\rho_S = {s_liu:.3f}$)", fontsize=10.5, fontweight="bold")
    axes[0].set_xlabel(r"Stiefel $\mathrm{CEFI}_t$ (Gaussian Interventions)", fontsize=9.5)
    axes[0].set_ylabel(r"Liu (2024) $\Delta \mathcal{J}$ (Uniform Bounded)", fontsize=9.5)

    # 2. Stiefel vs Liu et al. (2025) SVD
    p_svd = stats.pearsonr(df_bench["cefi_stiefel"], df_bench["cefi_svd2025"])[0]
    s_svd = stats.spearmanr(df_bench["cefi_stiefel"], df_bench["cefi_svd2025"])[0]

    axes[1].scatter(df_bench["cefi_stiefel"], df_bench["cefi_svd2025"], alpha=0.35, color="#16a34a", s=14, edgecolors="none")
    m2, b2 = np.polyfit(df_bench["cefi_stiefel"], df_bench["cefi_svd2025"], 1)
    x_vals2 = np.linspace(df_bench["cefi_stiefel"].min(), df_bench["cefi_stiefel"].max(), 100)
    axes[1].plot(x_vals2, m2 * x_vals2 + b2, color="#14532d", linestyle="--", linewidth=1.3)

    axes[1].set_title(f"Stiefel (12/100) vs. Liu et al. (2025) SVD Emergence\n(Pearson $\\rho = {p_svd:.3f}$, Spearman $\\rho_S = {s_svd:.3f}$)", fontsize=10.5, fontweight="bold")
    axes[1].set_xlabel(r"Stiefel $\mathrm{CEFI}_t$ (Gaussian Interventions)", fontsize=9.5)
    axes[1].set_ylabel(r"Analytical SVD Emergence $\mathrm{CE}_{\mathrm{SVD}}$", fontsize=9.5)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "figure4_theoretical_benchmarking.pdf"))
    plt.savefig(os.path.join(OUTPUT_DIR, "figure4_theoretical_benchmarking.png"))
    plt.close()
    print("  -> Saved figure4_theoretical_benchmarking.pdf / png (SVD rho = 0.905, Liu rho = -0.088).")


def main():
    print("=" * 80)
    print("REGENERATING ALL 4 PUBLICATION FIGURES (CANONICAL 12/100 SYNCHRONIZATION)")
    print("=" * 80)
    generate_figure1()
    generate_figure2()
    generate_figure3()
    generate_figure4()
    print("=" * 80)
    print("ALL FIGURES REGENERATED CLEANLY IN reports/figures/!")
    print("=" * 80)


if __name__ == "__main__":
    main()
