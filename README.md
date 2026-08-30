# Causal Emergence in Financial Markets: Dynamic Organization and Effective Dimensionality During Systemic Stress

**Author:** Felipe Mora  
**Affiliation:** Departamento de Industrias, Universidad Técnica Federico Santa María, Valparaíso, Chile  
**Email:** [felipe.morar@usm.cl](mailto:felipe.morar@usm.cl)  

---

## Overview

Welcome to the replication repository for the research codebase of **"Causal Emergence in Financial Markets: Dynamic Organization and Effective Dimensionality During Systemic Stress"**.

In this project, I explore whether financial markets exhibit *causal emergence*—a phenomenon where an optimal macroscopic coarse-graining of a multi-asset network reveals stronger intertemporal causal transitions and lower noise than the underlying disaggregated microscopic assets.

Using continuous-state information theory, Riemannian optimization on Stiefel manifolds, and Vector Autoregressive models over 35 years of daily U.S. industry portfolio returns (1990–2026), I introduce:
1. **$\mathrm{CEFI}_t$ (Causal Emergence Financial Index):** A scale-invariant metric that measures the maximum effective information rate per degree of freedom gained by projecting the system into an optimal lower-dimensional subspace.
2. **$q_t^*$ (Causal Effective Dimension):** The macro-dimension that concentrates maximum causal power.

```
                  ┌──────────────────────────────────────────────┐
                  │    Microscopic System (p Industries / VAR)   │
                  │   x_{t+1} = A x_t + eps_{t+1}, eps ~ N(0, Σ) │
                  └──────────────────────┬───────────────────────┘
                                         │  W ∈ V_q(R^p) (Stiefel Manifold)
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │      Optimal Macro Coarse-Graining (q < p)   │
                  │      y_{t+1} = (WAW^T) y_t + eps_M           │
                  │      Maximized Macro Effective Information   │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │  CEFI_t = max_q [ EI_q / q  -  EI_p / p ]    │
                  │  q_t^*  = argmax_q [ EI_q / q - EI_p / p ]   │
                  └──────────────────────┴───────────────────────┘
```

---

## Key Empirical Findings

- **State-Dependent Emergence:** In calm market conditions (e.g., 2005), observed emergence is consistent with univariate persistence and contemporaneous covariance ($p = 0.1000$ under a cross-lag isolation null). In contrast, during systemic crises (2008 GFC peak and 2020 COVID shock), $\mathrm{CEFI}_t$ significantly exceeds the static correlation benchmark ($p < 0.05$).
- **Distinct Crisis Mechanics:** The 2008 GFC peak exhibits significant off-diagonal cross-lag dynamical organization ($p = 0.0240$), whereas the 2020 COVID shock is dominated by simultaneous global co-movements without requiring additional cross-lag coupling ($p = 0.1150$).
- **Low-Dimensional Concentration:** Systemic liquidity/contagion crises concentrate causal dynamics into low-dimensional subspaces ($q^* \in \{2, 3\}$, modal $q^* = 2$, with 80.8% of trading days having $q^* \le 4$), whereas valuation repricing episodes (such as the 2000 Dot-Com crash and 2022 rate tightening) disperse across higher dimensions ($q^* \approx 5$, modal $q^* = 4$).
- **Theoretical Concordance:** The Riemannian Stiefel estimates display strong concordance with the exact continuous formulation of Liu et al. (2024) (Pearson $\rho = 0.837$) and the analytical SVD singular spectrum of Liu et al. (PRE 2025) ($\rho = 0.832$, with 88.7% dimensional agreement within $\pm 1$).

---

## Repository Structure

```text
.
├── config/                   # Model and dataset configuration files
├── data/
│   ├── raw/                  # Daily returns (FF30 and FF49 from French Data Library)
│   └── features/             # Processed daily CEFI time series and benchmark indices
├── src/
│   ├── causal_emergence/     # Core analytical EI, Stiefel optimizer, micro-VAR, null models
│   ├── econometrics/         # Predictive regressions, HAC covariance, bootstrap routines
│   └── utils/                # Data fetchers and helper routines
├── scripts/                  # Numbered empirical pipeline scripts (00 to 16)
├── reports/
│   ├── figures/              # Generated publication vector figures (Figures 1-4)
│   └── tables/               # Formatted LaTeX and CSV tables (Tables 1-4, sensitivity, FF49)
├── tests/                    # Unit tests verifying scale invariance, Stiefel QR, etc.
├── requirements.txt          # Python dependencies
├── pyproject.toml            # Project packaging specification
├── reproduce_all.sh          # Master reproduction script
└── README.md
```

---

## Getting Started

### 1. Clone the repository and set up your environment

```bash
git clone https://github.com/ShiroMorphy/causal-emergence-financial-risk.git
cd causal-emergence-financial-risk

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### 2. Run the test suite

```bash
PYTHONPATH=src pytest tests/ -v
```

### 3. Reproduce all results, tables, and figures

You can run the entire empirical pipeline with a single command:

```bash
./reproduce_all.sh
```

Or execute the individual scripts step by step:

```bash
# 1. Estimate rolling CEFI series across 1992-2026 (FF30)
PYTHONPATH=src python scripts/06_run_rolling_analysis.py --step 2

# 2. Run matched surrogate null inference (B=999 replications)
PYTHONPATH=src python scripts/07_run_null_inference.py --B 999

# 3. Evaluate external theoretical validation (Liu 2024 & PRE 2025 SVD)
PYTHONPATH=src python scripts/09_framework_robustness.py --step 10

# 4. Generate publication vector figures
PYTHONPATH=src python scripts/14_generate_manuscript_figures.py

# 5. Run financial benchmark regressions (H4)
PYTHONPATH=src python scripts/15_financial_benchmarks_h4.py
```

---

## Citation & Contact

If you use this code or data in your research, please cite:

```bibtex
@article{mora2026causal,
  title={Causal Emergence in Financial Markets: Dynamic Organization and Effective Dimensionality During Systemic Stress},
  author={Mora, Felipe},
  journal={Working Paper, Departamento de Industrias, Universidad T{\'e}cnica Federico Santa Mar{\'i}a},
  year={2026}
}
```

For questions, comments, or collaborations, feel free to contact me at [felipe.morar@usm.cl](mailto:felipe.morar@usm.cl).

---

## License

This repository is licensed under the MIT License. See `LICENSE` for details.
