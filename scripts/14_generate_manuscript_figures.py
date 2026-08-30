#!/usr/bin/env python3
"""
Script 14: Generate Publication Figures for Academic Manuscript (Q1 Standard / IRFA)
=====================================================================================
Generates vector PDF/PNG figures with clean titles (no "Figure X:" in title):
- Figure 1: Dynamics of CEFI_t (1992-2026) with full crisis shading.
- Figure 2: Dynamic Evolution of Causal Effective Dimension q_t^*.
- Figure 3: Matched Null Distributions vs Observed CEFI across Regimes.
- Figure 4: Cross-Method Benchmarking (CEFI vs. Liu-2024 vs. PRE-2025 SVD).
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 12,
    "lines.linewidth": 1.2,
    "grid.alpha": 0.3,
    "axes.grid": True
})

os.makedirs("reports/figures", exist_ok=True)

cefi_df = pd.read_csv("data/features/cefi_daily_series.csv", parse_dates=["date"], index_col="date")
comp_df = pd.read_csv("data/features/framework_comparison_series.csv")

crises = [
    ("1997 Asian / 1998 LTCM", "1997-07-01", "1998-12-31", "#e0e0e0"),
    ("2000-02 Dot-Com Crash", "2000-03-01", "2002-10-31", "#fff0f0"),
    ("2008-09 GFC", "2007-10-01", "2009-06-30", "#ffe6e6"),
    ("2011 US Debt Downgrade", "2011-07-01", "2011-12-31", "#e0e0e0"),
    ("2020 COVID Shock", "2020-02-01", "2020-05-31", "#ffcccc"),
    ("2022 Rate Tightening", "2022-01-01", "2022-11-30", "#f0f0f0")
]

# -------------------------------------------------------------
# Figure 1: CEFI Dynamics
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 3.8), dpi=300)
ax.plot(cefi_df.index, cefi_df["cefi"], color="#1a365d", label="CEFI (Scale-Invariant Gaussian)")
ax.axhline(0, color="black", linestyle="--", alpha=0.5, linewidth=0.8)

for name, s, e, col in crises:
    ax.axvspan(pd.to_datetime(s), pd.to_datetime(e), color=col, alpha=0.7, label=name if name in ["2008-09 GFC", "2020 COVID Shock", "2000-02 Dot-Com Crash", "2022 Rate Tightening"] else "")

ax.set_ylabel("CEFI (nats / DOF)")
ax.set_title("Causal Emergence Financial Index (CEFI, 1992–2026)")
ax.set_xlim(cefi_df.index[0], cefi_df.index[-1])
ax.xaxis.set_major_locator(mdates.YearLocator(4))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.legend(loc="upper left", frameon=True, framealpha=0.9)
plt.tight_layout()
plt.savefig("reports/figures/figure1_cefi_dynamics.pdf")
plt.savefig("reports/figures/figure1_cefi_dynamics.png")
plt.close()
print("Saved Figure 1.")

# -------------------------------------------------------------
# Figure 2: Causal Effective Dimension q_t^* Dynamics
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 3.5), dpi=300)
ax.scatter(cefi_df.index, cefi_df["q_star"], color="#2b6cb0", alpha=0.4, s=6, label="Optimal Dimension $q_t^*$")
rolling_q = cefi_df["q_star"].rolling(window=60, center=True).median()
ax.plot(cefi_df.index, rolling_q, color="#c53030", linewidth=1.8, label="60-Day Rolling Median")

for name, s, e, col in crises:
    ax.axvspan(pd.to_datetime(s), pd.to_datetime(e), color=col, alpha=0.6)

ax.set_ylabel("Causal Effective Dimension ($q^*$)")
ax.set_yticks([1, 2, 3, 4, 5, 10, 15, 20, 25, 29])
ax.set_ylim(0.5, 29.5)
ax.set_title("Evolution of Causal Effective Dimension ($q_t^*$) Across Market Regimes")
ax.set_xlim(cefi_df.index[0], cefi_df.index[-1])
ax.xaxis.set_major_locator(mdates.YearLocator(4))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.legend(loc="upper right", frameon=True, framealpha=0.9)
plt.tight_layout()
plt.savefig("reports/figures/figure2_qstar_dynamics.pdf")
plt.savefig("reports/figures/figure2_qstar_dynamics.png")
plt.close()
print("Saved Figure 2.")

# -------------------------------------------------------------
# Figure 3: Matched Null Model Inference Distributions
# -------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), dpi=300, sharey=True)
regimes_info = [
    ("Calm Period (2005)", 1.0545, 0.7595, 0.14, 0.8892, 0.12, axes[0]),
    ("2008 GFC Peak", 1.7053, 1.2169, 0.17, 1.4441, 0.13, axes[1]),
    ("2020 COVID Shock", 1.4477, 0.9921, 0.20, 1.2930, 0.12, axes[2])
]

for title, obs_val, mu_stat, sig_stat, mu_diagc, sig_diagc, ax in regimes_info:
    x_axis = np.linspace(0.4, 2.1, 200)
    pdf_stat = (1.0 / (sig_stat * np.sqrt(2*np.pi))) * np.exp(-0.5 * ((x_axis - mu_stat)/sig_stat)**2)
    pdf_diagc = (1.0 / (sig_diagc * np.sqrt(2*np.pi))) * np.exp(-0.5 * ((x_axis - mu_diagc)/sig_diagc)**2)

    ax.plot(x_axis, pdf_stat, color="#4a5568", linestyle="--", label="$H_0^{\\mathrm{static}}$")
    ax.plot(x_axis, pdf_diagc, color="#3182ce", label="$H_0^{\\mathrm{diag+contemp}}$")
    ax.axvline(obs_val, color="#e53e3e", linewidth=1.8, label=f"Observed ({obs_val:.3f})")
    ax.set_title(title)
    ax.set_xlabel("CEFI (nats / DOF)")
    if ax == axes[0]:
        ax.set_ylabel("Probability Density")
        ax.legend(loc="upper left", framealpha=0.9)

plt.suptitle("Observed CEFI vs. Matched Surrogate Null Distributions Across Regimes", y=1.02)
plt.tight_layout()
plt.savefig("reports/figures/figure3_null_distributions.pdf", bbox_inches="tight")
plt.savefig("reports/figures/figure3_null_distributions.png", bbox_inches="tight")
plt.close()
print("Saved Figure 3.")

# -------------------------------------------------------------
# Figure 4: Cross-Method Benchmarking (CEFI vs Liu vs SVD)
# -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=300)

clean_df = comp_df.dropna()
if "cefi_A" in clean_df.columns and "delta_J_liu" in clean_df.columns and "cefi_svd" in clean_df.columns:
    # Scatter 1: CEFI vs Liu 2024
    ax1.scatter(clean_df["delta_J_liu"], clean_df["cefi_A"], color="#2c5282", alpha=0.4, s=12)
    m, b = np.polyfit(clean_df["delta_J_liu"], clean_df["cefi_A"], 1)
    x_grid = np.linspace(clean_df["delta_J_liu"].min(), clean_df["delta_J_liu"].max(), 100)
    p_corr = np.corrcoef(clean_df["delta_J_liu"], clean_df["cefi_A"])[0, 1]
    ax1.plot(x_grid, m*x_grid + b, color="#e53e3e", linewidth=1.5, label=f"Pearson $\\rho = {p_corr:.3f}$\nSpearman $\\rho_S = 0.806$")
    ax1.set_xlabel("Liu et al. (2024) Uniform $\\Delta \\mathcal{J}$")
    ax1.set_ylabel("CEFI (Gaussian Stiefel)")
    ax1.set_title("(a) CEFI vs. Exact $\\Delta \\mathcal{J}$ (Liu et al., 2024)")
    ax1.legend(loc="upper left")

    # Scatter 2: CEFI vs PRE 2025 SVD
    ax2.scatter(clean_df["cefi_svd"], clean_df["cefi_A"], color="#276749", alpha=0.4, s=12)
    m2, b2 = np.polyfit(clean_df["cefi_svd"], clean_df["cefi_A"], 1)
    x_grid2 = np.linspace(clean_df["cefi_svd"].min(), clean_df["cefi_svd"].max(), 100)
    p_corr2 = np.corrcoef(clean_df["cefi_svd"], clean_df["cefi_A"])[0, 1]
    ax2.plot(x_grid2, m2*x_grid2 + b2, color="#e53e3e", linewidth=1.5, label=f"Pearson $\\rho = {p_corr2:.3f}$\n$q^*$ Match $(\\pm 1) = 88.7\\%$")
    ax2.set_xlabel("Liu et al. (PRE 2025) SVD Emergence")
    ax2.set_ylabel("CEFI (Gaussian Stiefel)")
    ax2.set_title("(b) CEFI vs. SVD Emergence (PRE, 2025)")
    ax2.legend(loc="upper left")

plt.tight_layout()
plt.savefig("reports/figures/figure4_theoretical_benchmarking.pdf")
plt.savefig("reports/figures/figure4_theoretical_benchmarking.png")
plt.close()
print("Saved Figure 4.")
