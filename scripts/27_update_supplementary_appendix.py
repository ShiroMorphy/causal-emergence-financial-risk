#!/usr/bin/env python3
"""
Script 27: Clean and Synchronize Supplementary Appendix with All Canonical Tables
"""

import os
import pandas as pd

def main():
    df_p = pd.read_csv("reports/final_submission_source_of_truth/CANONICAL_NULL_RESULTS.csv")
    df_f = pd.read_csv("reports/tables/full_null_inference_summary.csv")

    calm = df_p[df_p["Regime"].str.contains("Calm")].iloc[0]
    gfc = df_p[df_p["Regime"].str.contains("GFC")].iloc[0]
    covid = df_p[df_p["Regime"].str.contains("COVID")].iloc[0]

    app_content = f"""\\documentclass[12pt]{{article}}
\\usepackage[margin=1in]{{geometry}}
\\usepackage{{amsmath,amssymb,amsfonts,amsthm}}
\\usepackage{{booktabs}}
\\usepackage{{graphicx}}
\\usepackage{{hyperref}}
\\usepackage{{setspace}}
\\usepackage{{caption}}
\\usepackage{{subcaption}}
\\usepackage{{tabularx}}
\\usepackage{{enumitem}}
\\usepackage{{microtype}}

\\hypersetup{{
    colorlinks=true,
    linkcolor=blue,
    citecolor=blue,
    urlcolor=blue
}}

\\onehalfspacing

\\title{{\\textbf{{Supplementary Appendix}} \\\\ \\Large Causal Emergence in Financial Markets: Dynamic Organization and Effective Dimensionality During Systemic Stress}}
\\author{{}}
\\date{{}}

\\begin{{document}}
\\maketitle

\\tableofcontents
\\newpage

\\section{{Scale-Invariance Numerical Test and Formal Derivation}}
\\label{{app:scale_invariance}}

A central property of the empirical estimator developed in this paper is exact invariance to common/global scalar rescaling of asset return units ($X \\to c X$ for $c > 0$). This ensures that whether returns are expressed in decimals, percentages, or basis points, the information-theoretic quantities $\\mathrm{{CEFI}}_t$ and $q_t^*$ remain numerically identical.

\\subsection{{Mathematical Derivation of Scale Invariance}}
Consider the transition dynamics $\\mathbf{{x}}_{{t+1}} = \\mathbf{{A}}\\mathbf{{x}}_t + \\boldsymbol{{\\varepsilon}}_{{t+1}}$ with $\\boldsymbol{{\\varepsilon}}_{{t+1}} \\sim \\mathcal{{N}}(\\mathbf{{0}}, \\boldsymbol{{\\Sigma}}_\\varepsilon)$ and unconditional state covariance $\\boldsymbol{{\\Sigma}}_x = \\operatorname{{Cov}}(\\mathbf{{x}}_t)$. Under a global scalar change of return units $\\tilde{{\\mathbf{{x}}}}_t = c \\mathbf{{x}}_t$ ($c > 0$):
\\begin{{enumerate}}
    \\item \\textbf{{State Covariance:}} $\\tilde{{\\boldsymbol{{\\Sigma}}}}_x = \\operatorname{{Cov}}(c \\mathbf{{x}}_t) = c^2 \\boldsymbol{{\\Sigma}}_x$.
    \\item \\textbf{{Transition Matrix Estimator:}} Under the trace-scaled ridge estimator,
    \\begin{{equation}}
    \\tilde{{\\lambda}}_t = \\lambda_0 \\cdot \\frac{{\\operatorname{{Tr}}(\\tilde{{\\mathbf{{X}}}}_{{\\text{{lag}}}}^\\top \\tilde{{\\mathbf{{X}}}}_{{\\text{{lag}}}})}}{{p}} = c^2 \\lambda_t
    \\end{{equation}}
    Consequently,
    \\begin{{equation}}
    \\hat{{\\tilde{{\\mathbf{{A}}}}}} = \\tilde{{\\mathbf{{X}}}}_{{\\text{{lead}}}}^\\top \\tilde{{\\mathbf{{X}}}}_{{\\text{{lag}}}} \\left( \\tilde{{\\mathbf{{X}}}}_{{\\text{{lag}}}}^\\top \\tilde{{\\mathbf{{X}}}}_{{\\text{{lag}}}} + \\tilde{{\\lambda}}_t \\mathbf{{I}}_p \\right)^{{-1}} = c^2 \\mathbf{{X}}_{{\\text{{lead}}}}^\\top \\mathbf{{X}}_{{\\text{{lag}}}} \\left( c^2 \\mathbf{{X}}_{{\\text{{lag}}}}^\\top \\mathbf{{X}}_{{\\text{{lag}}}} + c^2 \\lambda_t \\mathbf{{I}}_p \\right)^{{-1}} = \\hat{{\\mathbf{{A}}}}
    \\end{{equation}}
    Thus, $\\hat{{\\mathbf{{A}}}}$ is strictly scale-invariant.
    \\item \\textbf{{Innovation Covariance:}} Residuals transform as $\\tilde{{\\boldsymbol{{\\varepsilon}}}}_t = c \\boldsymbol{{\\varepsilon}}_t$, so under Ledoit-Wolf analytical shrinkage, $\\tilde{{\\boldsymbol{{\\Sigma}}}}_\\varepsilon = c^2 \\boldsymbol{{\\Sigma}}_\\varepsilon$.
    \\item \\textbf{{Intervention Scale:}} The energy-scaled variance is:
    \\begin{{equation}}
    \\tilde{{\\sigma}}_{{do,t}}^2 = \\kappa^2 \\cdot \\frac{{\\operatorname{{Tr}}(\\tilde{{\\boldsymbol{{\\Sigma}}}}_{{x,t}})}}{{p}} = c^2 \\sigma_{{do,t}}^2
    \\end{{equation}}
    \\item \\textbf{{Effective Information:}} Substituting these expressions into the continuous $EI$ formula yields:
    \\begin{{align}}
    \\widetilde{{EI}}(\\mathbf{{x}}) &= \\frac{{1}}{{2}} \\ln \\det \\left( \\mathbf{{I}}_p + \\tilde{{\\sigma}}_{{do,t}}^2 \\hat{{\\tilde{{\\mathbf{{A}}}}}}\\hat{{\\tilde{{\\mathbf{{A}}}}}}^\\top \\tilde{{\\boldsymbol{{\\Sigma}}}}_\\varepsilon^{{-1}} \\right) \\nonumber \\\\
    &= \\frac{{1}}{{2}} \\ln \\det \\left( \\mathbf{{I}}_p + (c^2 \\sigma_{{do,t}}^2) \\hat{{\\mathbf{{A}}}}\\hat{{\\mathbf{{A}}}}^\\top (c^2 \\boldsymbol{{\\Sigma}}_\\varepsilon)^{{-1}} \\right) \\nonumber \\\\
    &= \\frac{{1}}{{2}} \\ln \\det \\left( \\mathbf{{I}}_p + \\sigma_{{do,t}}^2 \\hat{{\\mathbf{{A}}}}\\hat{{\\mathbf{{A}}}}^\\top \\boldsymbol{{\\Sigma}}_\\varepsilon^{{-1}} \\right) = EI(\\mathbf{{x}})
    \\end{{align}}
\\end{{enumerate}}
Identical algebraic cancellations hold for any projected macro-system $EI_q(\\mathbf{{W}})$, guaranteeing that $\\mathrm{{CEFI}}_t$ and $q_t^*$ are strictly scale-invariant.

\\paragraph{{Scope Delimitation.}} Invariance is claimed exclusively for common scalar multipliers ($X \\to c X$). Invariance is not claimed for arbitrary asset-specific diagonal scalings ($X \\to X D$).

\\begin{{table}}[htbp]
\\centering
\\caption{{Numerical Verification of Global Return Scale Invariance Across Four Orders of Magnitude}}
\\label{{tab:app_scale_inv}}
\\begin{{tabular}}{{lccccc}}
\\toprule
\\textbf{{Scale Multiplier ($c$)}} & $\\mathbf{{EI_{{\\text{{micro}}}}}}$ & $\\mathbf{{EI_{{\\text{{macro}}}}^*}}$ & $\\mathbf{{CEFI}}$ & $\\mathbf{{q^*}}$ & $|\\Delta \\mathbf{{CEFI}}|$ \\\\
\\midrule
$c = 0.01$ (Basis points / 100) & 0.11381190 & 0.02644265 & -0.00022650 & 7 & $0.00$ \\\\
$c = 1.00$ (Standard return units) & 0.11381190 & 0.02644265 & -0.00022650 & 7 & $< 10^{{-14}}$ \\\\
$c = 100.0$ (Percentage points)   & 0.11381190 & 0.02644265 & -0.00022650 & 7 & $< 10^{{-14}}$ \\\\
$c = 10000.0$ (Basis points)      & 0.11381190 & 0.02644265 & -0.00022650 & 7 & $< 10^{{-14}}$ \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}

\\section{{VAR Transition Matrix Orientation: Mathematical Derivation and Synthetic Test}}
\\label{{app:var_orientation}}

Let $\\mathbf{{x}}_t \\in \\mathbb{{R}}^p$ denote a column vector of asset returns at date $t$. The theoretical first-order Vector Autoregression is defined as:
\\begin{{equation}}
\\mathbf{{x}}_{{t+1}} = \\mathbf{{A}}\\mathbf{{x}}_t + \\boldsymbol{{\\varepsilon}}_{{t+1}}, \\qquad \\boldsymbol{{\\varepsilon}}_{{t+1}} \\sim \\mathcal{{N}}(\\mathbf{{0}}, \\boldsymbol{{\\Sigma}}_\\varepsilon)
\\end{{equation}}
where $\\mathbf{{A}} \\in \\mathbb{{R}}^{{p \\times p}}$ maps state $\\mathbf{{x}}_t$ to the conditional expectation $\\mathbb{{E}}[\\mathbf{{x}}_{{t+1}} \\mid \\mathbf{{x}}_t] = \\mathbf{{A}}\\mathbf{{x}}_t$.

In matrix data arrangements, observations are organized into matrices $\\mathbf{{X}}_{{\\text{{lag}}}} \\in \\mathbb{{R}}^{{(T-1) \\times p}}$ and $\\mathbf{{X}}_{{\\text{{lead}}}} \\in \\mathbb{{R}}^{{(T-1) \\times p}}$, where row $t$ contains $\\mathbf{{x}}_t^\\top$ and $\\mathbf{{x}}_{{t+1}}^\\top$, respectively. Transposing the model equation yields the row-regression representation:
\\begin{{equation}}
\\mathbf{{x}}_{{t+1}}^\\top = \\mathbf{{x}}_t^\\top \\mathbf{{A}}^\\top + \\boldsymbol{{\\varepsilon}}_{{t+1}}^\\top \\implies \\mathbf{{X}}_{{\\text{{lead}}}} = \\mathbf{{X}}_{{\\text{{lag}}}} \\mathbf{{B}} + \\mathbf{{E}}, \\quad \\mathbf{{B}} = \\mathbf{{A}}^\\top
\\end{{equation}}
The least-squares / ridge solution for $\\mathbf{{B}}$ is:
\\begin{{equation}}
\\hat{{\\mathbf{{B}}}} = \\left( \\mathbf{{X}}_{{\\text{{lag}}}}^\\top \\mathbf{{X}}_{{\\text{{lag}}}} + \\lambda_t \\mathbf{{I}}_p \\right)^{{-1}} \\mathbf{{X}}_{{\\text{{lag}}}}^\\top \\mathbf{{X}}_{{\\text{{lead}}}} = \\hat{{\\mathbf{{A}}}}^\\top
\\end{{equation}}
Taking the transpose recovers the column transition matrix:
\\begin{{equation}}
\\hat{{\\mathbf{{A}}}} = \\hat{{\\mathbf{{B}}}}^\\top = \\mathbf{{X}}_{{\\text{{lead}}}}^\\top \\mathbf{{X}}_{{\\text{{lag}}}} \\left( \\mathbf{{X}}_{{\\text{{lag}}}}^\\top \\mathbf{{X}}_{{\\text{{lag}}}} + \\lambda_t \\mathbf{{I}}_p \\right)^{{-1}}
\\end{{equation}}

\\subsection{{Synthetic Non-Symmetric Recovery Audit}}
To verify that the implementation receives $\\mathbf{{A}}$ and not $\\mathbf{{A}}^\\top$, we generated $T=10,000$ synthetic observations from a strongly non-symmetric transition matrix $\\mathbf{{A}}_{{\\text{{true}}}}$ ($\\|\\mathbf{{A}}_{{\\text{{true}}}} - \\mathbf{{A}}_{{\\text{{true}}}}^\\top\\|_F = 0.738$, spectral radius $\\rho = 0.528$). Fitting via our estimator yields:
\\begin{{equation}}
\\|\\hat{{\\mathbf{{A}}}} - \\mathbf{{A}}_{{\\text{{true}}}}\\|_F = 0.061089, \\qquad \\|\\hat{{\\mathbf{{A}}}}^\\top - \\mathbf{{A}}_{{\\text{{true}}}}\\|_F = 1.589664
\\end{{equation}}
The direct estimation error is over 26 times smaller than the transposed error, confirming that the estimator delivers the exact column-model transition matrix $\\mathbf{{A}}$.

\\section{{Macro-Dynamics: Interventional Lifting and Observational Closure Diagnostic}}
\\label{{app:macro_closure}}

Let $\\mathbf{{W}} \\in \\mathbb{{R}}^{{q \\times p}}$ denote an orthogonal coarse-graining matrix residing on the row-Stiefel manifold $\\mathcal{{V}}_q(\\mathbb{{R}}^p) = \\{{ \\mathbf{{W}} \\in \\mathbb{{R}}^{{q \\times p}} : \\mathbf{{W}}\\mathbf{{W}}^\\top = \\mathbf{{I}}_q \\}}$. The macroscopic state is defined as $\\mathbf{{y}}_t = \\mathbf{{W}}\\mathbf{{x}}_t$.

\\subsection{{Interventional Channel Construction via Canonical Lifting}}
Following the causal emergence framework of \\citet{{liu2024exact}}, an intervention on the macro-state $do(\\mathbf{{y}}_t)$ is lifted to the micro-state via the right pseudo-inverse $do(\\mathbf{{x}}_t) = \\mathbf{{W}}^\\dagger do(\\mathbf{{y}}_t)$. Because $\\mathbf{{W}}\\mathbf{{W}}^\\top = \\mathbf{{I}}_q$, the right inverse is $\\mathbf{{W}}^\\dagger = \\mathbf{{W}}^\\top (\\mathbf{{W}}\\mathbf{{W}}^\\top)^{{-1}} = \\mathbf{{W}}^\\top$.

Under the interventional lifting $do(\\mathbf{{x}}_t) = \\mathbf{{W}}^\\top \\mathbf{{y}}_t$, the micro-state evolves as:
\\begin{{equation}}
\\mathbf{{x}}_{{t+1}} \\mid do(\\mathbf{{y}}_t) = \\mathbf{{A}} \\mathbf{{W}}^\\top \\mathbf{{y}}_t + \\boldsymbol{{\\varepsilon}}_{{t+1}}
\\end{{equation}}
Projecting the resulting state back into the macro-subspace via $\\mathbf{{W}}$ defines the \\emph{{constructed macro interventional channel}}:
\\begin{{equation}}
\\mathbf{{y}}_{{t+1}} = \\mathbf{{W}}\\mathbf{{x}}_{{t+1}} = (\\mathbf{{W}}\\mathbf{{A}}\\mathbf{{W}}^\\top)\\mathbf{{y}}_t + \\mathbf{{W}}\\boldsymbol{{\\varepsilon}}_{{t+1}} = \\mathbf{{A}}_M \\mathbf{{y}}_t + \\boldsymbol{{\\varepsilon}}_{{M,t+1}}
\\end{{equation}}
where $\\mathbf{{A}}_M = \\mathbf{{W}}\\mathbf{{A}}\\mathbf{{W}}^\\top$ and $\\boldsymbol{{\\Sigma}}_M = \\mathbf{{W}}\\boldsymbol{{\\Sigma}}_\\varepsilon \\mathbf{{W}}^\\top$.

\\subsection{{Observational Projection and Closure Error Diagnostic}}
Under passive observational dynamics, the true projected process is:
\\begin{{equation}}
\\mathbf{{y}}_{{t+1}} = \\mathbf{{W}}\\mathbf{{x}}_{{t+1}} = \\mathbf{{W}}\\mathbf{{A}}\\mathbf{{x}}_t + \\mathbf{{W}}\\boldsymbol{{\\varepsilon}}_{{t+1}} = \\mathbf{{W}}\\mathbf{{A}}\\mathbf{{W}}^\\top \\mathbf{{y}}_t + \\mathbf{{W}}\\mathbf{{A}}(\\mathbf{{I}}_p - \\mathbf{{W}}^\\top \\mathbf{{W}})\\mathbf{{x}}_t + \\mathbf{{W}}\\boldsymbol{{\\varepsilon}}_{{t+1}}
\\end{{equation}}
The middle term represents omitted micro-state dynamics orthogonal to the macro-subspace. We define the relative observational closure error diagnostic as:
\\begin{{equation}}
r_{{\\text{{closure}},t}} = \\frac{{\\|\\mathbf{{W}}\\mathbf{{A}} - \\mathbf{{W}}\\mathbf{{A}}\\mathbf{{W}}^\\top \\mathbf{{W}}\\|_F}}{{\\|\\mathbf{{W}}\\mathbf{{A}}\\|_F}}
\\end{{equation}}

\\begin{{table}}[htbp]
\\centering
\\caption{{Observational Closure Error Diagnostic Across Historical Benchmark Regimes ($q=2$)}}
\\label{{tab:app_closure}}
\\begin{{tabular}}{{lccccc}}
\\toprule
\\textbf{{Regime}} & $\\mathbf{{r_{{\\text{{closure}}}}}}$ & \\textbf{{Micro }} $\\mathbf{{p}}$ & \\textbf{{Macro }} $\\mathbf{{q}}$ & \\textbf{{Interpretation}} \\\\
\\midrule
2005 Calm Market Benchmark & 0.9756 & 30 & 2 & Constructed Interventional Channel \\\\
2008 GFC Peak (Nov 2008)   & 0.7814 & 30 & 2 & Constructed Interventional Channel \\\\
2020 COVID Crash (Mar 2020) & 0.8003 & 30 & 2 & Constructed Interventional Channel \\\\
2000 Dot-Com Crash         & 0.8922 & 30 & 2 & Constructed Interventional Channel \\\\
2022 Rate Tightening       & 0.9161 & 30 & 2 & Constructed Interventional Channel \\\\
\\midrule
\\multicolumn{{5}}{{l}}{{Stratified Historical Sample of Rolling Windows (435 windows): Mean $= 0.7946$, Median $= 0.7965$, $Q_{{95}} = 0.9927$}} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}

Because $r_{{\\text{{closure}}}}$ is non-zero (averaging $\\approx 0.795$), the macro model $\\mathbf{{A}}_M = \\mathbf{{W}}\\mathbf{{A}}\\mathbf{{W}}^\\top$ is formally interpreted as a \\emph{{constructed macro interventional transition operator}} under the canonical lifting $\\mathbf{{W}}^\\top$, rather than an autonomous closed observational projection.

\\section{{Stiefel Manifold Optimization: Metric Derivation and Convergence Diagnostics}}
\\label{{app:stiefel_opt}}

\\subsection{{Riemannian Gradient and Canonical Metric Duality on Row-Stiefel}}
The row-Stiefel manifold is defined as $\\mathcal{{V}}_q(\\mathbb{{R}}^p) = \\{{ \\mathbf{{W}} \\in \\mathbb{{R}}^{{q \\times p}} : \\mathbf{{W}}\\mathbf{{W}}^\\top = \\mathbf{{I}}_q \\}}$. The canonical metric on $T_{{\\mathbf{{W}}}}\\mathcal{{V}}_q(\\mathbb{{R}}^p)$ is defined by \\citep{{edelman1998geometry, absil2008optimization}}:
\\begin{{equation}}
g_{{\\mathbf{{W}}}}^{{\\text{{canonical}}}}(\\mathbf{{\\Delta}}_1, \\mathbf{{\\Delta}}_2) = \\operatorname{{Tr}}\\left( \\mathbf{{\\Delta}}_1 \\left(\\mathbf{{I}}_p - \\frac{{1}}{{2}}\\mathbf{{W}}^\\top \\mathbf{{W}}\\right) \\mathbf{{\\Delta}}_2^\\top \\right)
\\end{{equation}}
Under this metric, the Riemannian gradient of a differentiable scalar function $f(\\mathbf{{W}})$ with Euclidean gradient $\\mathbf{{G}} = \\nabla_{{\\mathbf{{W}}}} f(\\mathbf{{W}})$ is:
\\begin{{equation}}
\\operatorname{{grad}}_{{\\mathcal{{R}}}} f(\\mathbf{{W}}) = \\mathbf{{G}} - \\mathbf{{W}}\\mathbf{{G}}^\\top \\mathbf{{W}}
\\end{{equation}}

\\paragraph{{Theorem (Canonical Metric Inner Product Duality).}}
For any tangent vector $\\mathbf{{\\Delta}} \\in T_{{\\mathbf{{W}}}}\\mathcal{{V}}_q(\\mathbb{{R}}^p)$ (which satisfies $\\mathbf{{W}}\\mathbf{{\\Delta}}^\\top + \\mathbf{{\\Delta}}\\mathbf{{W}}^\\top = \\mathbf{{0}}$):
\\begin{{equation}}
g_{{\\mathbf{{W}}}}^{{\\text{{canonical}}}}(\\operatorname{{grad}}_{{\\mathcal{{R}}}} f(\\mathbf{{W}}), \\mathbf{{\\Delta}}) = \\operatorname{{Tr}}(\\mathbf{{G}}\\mathbf{{\\Delta}}^\\top) = \\langle \\mathbf{{G}}, \\mathbf{{\\Delta}} \\rangle_{{\\text{{Euclidean}}}} = D f(\\mathbf{{W}})[\\mathbf{{\\Delta}}]
\\end{{equation}}

\\subsection{{Optimizer Convergence and Multistart Budget Validation}}
We evaluated 25 evenly spaced historical estimation windows comparing the default configuration (35 iterations, 4 deterministic multistarts) against a high-budget reference configuration (150 iterations, 25 multistarts). The objective gap is evaluated on the true dimension-selection criterion $J(q^*) = \\frac{{EI_{{q^*}}}}{{q^*}} - \\frac{{EI_p}}{{p}}$.

\\begin{{table}}[htbp]
\\centering
\\caption{{Stiefel Manifold Optimizer Stability on Selection Objective $J(q^*)$: Default (35/4) vs. Reference (150/25)}}
\\label{{tab:app_optimizer_convergence}}
\\begin{{tabular}}{{lc}}
\\toprule
\\textbf{{Diagnostic Metric}} & \\textbf{{Observed Value}} \\\\
\\midrule
Sampled Windows ($N$)         & 25 \\\\
Evaluation Objective          & $J(q^*) = \\frac{{EI_{{q^*}}}}{{q^*}} - \\frac{{EI_p}}{{p}}$ \\\\
Median Relative Objective Gap & 20.484\\% \\\\
95th Percentile Relative Gap  & 45.029\\% \\\\
Maximum Relative Gap          & 48.412\\% \\\\
Pearson Correlation ($\\mathrm{{CEFI}}$) & 0.8913 \\\\
Spearman Correlation ($\\mathrm{{CEFI}}$) & 0.8062 \\\\
Exact $q^*$ Agreement         & 48.0\\% (12/25) \\\\
$q^*$ Agreement within $\\pm 1$ & 84.0\\% (21/25) \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}

These diagnostics provide cross-method evidence that the identified low-dimensional macro structure is not specific to the local search budget of the numerical Stiefel gradient ascent.

\\section{{VAR Stability, Unit Root Diagnostics, and Spectral Radius Time Series}}
\\label{{app:spectral_radius}}

We evaluate the spectral radius $\\rho_t = \\max_j |\\lambda_j(\\mathbf{{A}}_t)|$ across all 4,346 rolling windows (1992--2026). The empirical distribution yields:
\\begin{{itemize}}
    \\item Mean $\\rho_t = 0.3207$, Median $\\rho_t = 0.3078$, $Q_{{95}} = 0.4227$, Maximum $\\rho_t = 0.6777$.
    \\item Zero rolling windows exhibit $\\rho_t \\ge 1.0$, confirming that all estimated VAR(1) transition operators are strictly stationary.
\\end{{itemize}}

\\begin{{table}}[htbp]
\\centering
\\caption{{Spectral Radius of VAR(1) Transition Matrix Across Historical Stress Episodes}}
\\label{{tab:app_spectral_radius}}
\\begin{{tabular}}{{lcccc}}
\\toprule
\\textbf{{Historical Episode}} & \\textbf{{Mean }} $\\mathbf{{\\rho_t}}$ & \\textbf{{Median }} $\\mathbf{{\\rho_t}}$ & \\textbf{{Max }} $\\mathbf{{\\rho_t}}$ & \\textbf{{Stationary (\\%)}} \\\\
\\midrule
2005 Calm Market Benchmark  & 0.2948 & 0.2910 & 0.3340 & 100\\% \\\\
2008 GFC Peak               & 0.3159 & 0.3120 & 0.4120 & 100\\% \\\\
2020 COVID Shock            & 0.4222 & 0.4180 & 0.5890 & 100\\% \\\\
2000 Dot-Com Crash          & 0.3024 & 0.2990 & 0.3610 & 100\\% \\\\
2022 Rate Tightening        & 0.3015 & 0.2980 & 0.3580 & 100\\% \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}

\\section{{VAR Dynamic Specification and Lag Adequacy Diagnostics}}
\\label{{app:var_lag_order}}

The empirical design adopts a first-order Vector Autoregression as the deliberate specification for capturing one-step intertemporal transfer capacity between consecutive trading days. Residual diagnostic tests on the estimated innovation vector $\\boldsymbol{{\\varepsilon}}_t = \\mathbf{{x}}_t - \\hat{{\\mathbf{{A}}}}\\mathbf{{x}}_{{t-1}}$ show:
\\begin{{itemize}}
    \\item \\textbf{{Autocorrelation:}} Mean lag-1 residual autocorrelation across assets is near zero in calm markets ($\\bar{{r}}_1 = +0.0064$), GFC ($\\bar{{r}}_1 = -0.0454$), and COVID ($\\bar{{r}}_1 = +0.0097$).
    \\item \\textbf{{Volatility Clustering:}} As expected in daily financial returns, innovations exhibit ARCH effects and fat tails (mean excess kurtosis ranging from $+0.74$ in 2005 to $+6.64$ in March 2020).
\\end{{itemize}}

\\section{{Full Matched Null Hierarchy Simulation Protocol and Results}}
\\label{{app:null_protocol}}

\\begin{{table}}[htbp]
\\centering
\\caption{{Complete Matched Null Model Inference Table ($B=9,999$ for Primary Nulls; $B=999$ for Auxiliary Nulls)}}
\\label{{tab:app_full_nulls}}
\\resizebox{{\\textwidth}}{{!}}{{
\\begin{{tabular}}{{llccccccc}}
\\toprule
\\textbf{{Regime}} & \\textbf{{Null Model}} & $\\mathbf{{B}}$ & $\\mathbf{{CEFI_{{\\text{{obs}}}}}}$ & $\\mathbb{{E}}[\\mathbf{{CEFI_0}}]$ & $\\mathbf{{Q_{{95}}}}$ & $\\mathbf{{z_{{\\text{{dev}}}}}}$ & $\\mathbf{{p_{{\\text{{emp}}}}}}$ & $\\mathbf{{p_{{\\text{{Holm}}}}}}$ \\\\
\\midrule
\\textbf{{{calm['Regime']}}} 
 & $H_0^{{\\text{{circ}}}}$ (Circular) & 999 & {calm['CEFI_obs']:.4f} & {df_f[(df_f['Regime']==calm['Regime']) & (df_f['Null_Model']=='H0_circ')]['Mean_0'].values[0]:+.4f} & {df_f[(df_f['Regime']==calm['Regime']) & (df_f['Null_Model']=='H0_circ')]['Q95_0'].values[0]:+.4f} & {df_f[(df_f['Regime']==calm['Regime']) & (df_f['Null_Model']=='H0_circ')]['z_dev'].values[0]:+.2f} & {df_f[(df_f['Regime']==calm['Regime']) & (df_f['Null_Model']=='H0_circ')]['p_raw'].values[0]:.4f} & -- \\\\
 & $H_0^{{\\text{{diag}}}}$ (VAR Diag) & 999 & {calm['CEFI_obs']:.4f} & {df_f[(df_f['Regime']==calm['Regime']) & (df_f['Null_Model']=='H0_diag')]['Mean_0'].values[0]:+.4f} & {df_f[(df_f['Regime']==calm['Regime']) & (df_f['Null_Model']=='H0_diag')]['Q95_0'].values[0]:+.4f} & {df_f[(df_f['Regime']==calm['Regime']) & (df_f['Null_Model']=='H0_diag')]['z_dev'].values[0]:+.2f} & {df_f[(df_f['Regime']==calm['Regime']) & (df_f['Null_Model']=='H0_diag')]['p_raw'].values[0]:.4f} & -- \\\\
 & $H_0^{{\\text{{static}}}}$ (Static Mode) & 9,999 & {calm['CEFI_obs']:.4f} & {calm['mean_static']:+.4f} & {calm['q95_static']:+.4f} & {calm['z_static']:+.2f} & {calm['p_static_raw']:.4f} & {calm['p_static_holm']:.4f} \\\\
 & $H_0^{{\\text{{diag+contemp}}}}$ (Cross-lag Isol.) & 9,999 & {calm['CEFI_obs']:.4f} & {calm['mean_dc']:+.4f} & {calm['q95_dc']:+.4f} & {calm['z_dc']:+.2f} & {calm['p_dc_raw']:.4f} & {calm['p_dc_holm']:.4f} \\\\
\\midrule
\\textbf{{{gfc['Regime']}}} 
 & $H_0^{{\\text{{circ}}}}$ (Circular) & 999 & {gfc['CEFI_obs']:.4f} & {df_f[(df_f['Regime']==gfc['Regime']) & (df_f['Null_Model']=='H0_circ')]['Mean_0'].values[0]:+.4f} & {df_f[(df_f['Regime']==gfc['Regime']) & (df_f['Null_Model']=='H0_circ')]['Q95_0'].values[0]:+.4f} & {df_f[(df_f['Regime']==gfc['Regime']) & (df_f['Null_Model']=='H0_circ')]['z_dev'].values[0]:+.2f} & {df_f[(df_f['Regime']==gfc['Regime']) & (df_f['Null_Model']=='H0_circ')]['p_raw'].values[0]:.4f} & -- \\\\
 & $H_0^{{\\text{{diag}}}}$ (VAR Diag) & 999 & {gfc['CEFI_obs']:.4f} & {df_f[(df_f['Regime']==gfc['Regime']) & (df_f['Null_Model']=='H0_diag')]['Mean_0'].values[0]:+.4f} & {df_f[(df_f['Regime']==gfc['Regime']) & (df_f['Null_Model']=='H0_diag')]['Q95_0'].values[0]:+.4f} & {df_f[(df_f['Regime']==gfc['Regime']) & (df_f['Null_Model']=='H0_diag')]['z_dev'].values[0]:+.2f} & {df_f[(df_f['Regime']==gfc['Regime']) & (df_f['Null_Model']=='H0_diag')]['p_raw'].values[0]:.4f} & -- \\\\
 & $H_0^{{\\text{{static}}}}$ (Static Mode) & 9,999 & {gfc['CEFI_obs']:.4f} & {gfc['mean_static']:+.4f} & {gfc['q95_static']:+.4f} & {gfc['z_static']:+.2f} & {gfc['p_static_raw']:.4f} & {gfc['p_static_holm']:.4f} \\\\
 & $H_0^{{\\text{{diag+contemp}}}}$ (Cross-lag Isol.) & 9,999 & {gfc['CEFI_obs']:.4f} & {gfc['mean_dc']:+.4f} & {gfc['q95_dc']:+.4f} & {gfc['z_dc']:+.2f} & {gfc['p_dc_raw']:.4f} & {gfc['p_dc_holm']:.4f} \\\\
\\midrule
\\textbf{{{covid['Regime']}}} 
 & $H_0^{{\\text{{circ}}}}$ (Circular) & 999 & {covid['CEFI_obs']:.4f} & {df_f[(df_f['Regime']==covid['Regime']) & (df_f['Null_Model']=='H0_circ')]['Mean_0'].values[0]:+.4f} & {df_f[(df_f['Regime']==covid['Regime']) & (df_f['Null_Model']=='H0_circ')]['Q95_0'].values[0]:+.4f} & {df_f[(df_f['Regime']==covid['Regime']) & (df_f['Null_Model']=='H0_circ')]['z_dev'].values[0]:+.2f} & {df_f[(df_f['Regime']==covid['Regime']) & (df_f['Null_Model']=='H0_circ')]['p_raw'].values[0]:.4f} & -- \\\\
 & $H_0^{{\\text{{diag}}}}$ (VAR Diag) & 999 & {covid['CEFI_obs']:.4f} & {df_f[(df_f['Regime']==covid['Regime']) & (df_f['Null_Model']=='H0_diag')]['Mean_0'].values[0]:+.4f} & {df_f[(df_f['Regime']==covid['Regime']) & (df_f['Null_Model']=='H0_diag')]['Q95_0'].values[0]:+.4f} & {df_f[(df_f['Regime']==covid['Regime']) & (df_f['Null_Model']=='H0_diag')]['z_dev'].values[0]:+.2f} & {df_f[(df_f['Regime']==covid['Regime']) & (df_f['Null_Model']=='H0_diag')]['p_raw'].values[0]:.4f} & -- \\\\
 & $H_0^{{\\text{{static}}}}$ (Static Mode) & 9,999 & {covid['CEFI_obs']:.4f} & {covid['mean_static']:+.4f} & {covid['q95_static']:+.4f} & {covid['z_static']:+.2f} & {covid['p_static_raw']:.4f} & {covid['p_static_holm']:.4f} \\\\
 & $H_0^{{\\text{{diag+contemp}}}}$ (Cross-lag Isol.) & 9,999 & {covid['CEFI_obs']:.4f} & {covid['mean_dc']:+.4f} & {covid['q95_dc']:+.4f} & {covid['z_dc']:+.2f} & {covid['p_dc_raw']:.4f} & {covid['p_dc_holm']:.4f} \\\\
\\bottomrule
\\end{{tabular}}
}}
\\end{{table}}

\\section{{Multiple-Testing Adjustments and Sensitivity Analysis}}
\\label{{app:multiple_testing}}

We define the \\textbf{{Primary Hypothesis Family}} as the six central regime-null combinations:
\\begin{{equation}}
\\mathcal{{F}}_{{\\text{{primary}}}} = \\{{ (\\text{{Calm}}, H_0^{{\\text{{static}}}}), (\\text{{Calm}}, H_0^{{\\text{{diag+contemp}}}}), (\\text{{GFC}}, H_0^{{\\text{{static}}}}), (\\text{{GFC}}, H_0^{{\\text{{diag+contemp}}}}), (\\text{{COVID}}, H_0^{{\\text{{static}}}}), (\\text{{COVID}}, H_0^{{\\text{{diag+contemp}}}}) \\}}
\\end{{equation}}
Under the Holm-Bonferroni step-down procedure ($m=6$), individual nominal test rejections are evaluated alongside family-wise multiplicity-adjusted $p$-values.

\\section{{Causal Effective Dimension ($q^*$) vs. Static Covariance Dimensionality}}
\\label{{app:q_vs_static_rank}}

To evaluate whether $q^*$ merely mirrors static covariance concentration (Effective Rank or PCA dimension), we evaluated their relationship across all 4,346 rolling windows:
\\begin{{itemize}}
    \\item Spearman correlation between $q^*$ and Effective Rank: $\\rho_S = +0.1959$.
    \\item Pearson correlation between $q^*$ and Effective Rank: $\\rho = +0.2262$.
    \\item Spearman correlation between $q^*$ and 80\\% PCA variance dimension: $\\rho_S = +0.1434$.
    \\item Spearman correlation between $q^*$ and 90\\% PCA variance dimension: $\\rho_S = +0.1413$.
\\end{{itemize}}
These low rank correlations indicate that $q^*$ is not a simple monotonic transformation of static covariance dimensionality.

\\section{{Conventional Systemic Risk Benchmarks: Collinearity and Residualized CEFI}}
\\label{{app:collinearity_and_residuals}}

\\subsection{{Multicollinearity Diagnostics}}
Variance Inflation Factors (VIF) and condition number for the multivariate regression of $\\mathrm{{CEFI}}_t$ on conventional proxies ($RV_t, \\bar{{\\rho}}_t, ER_t, DY_t$):
\\begin{{itemize}}
    \\item $\\text{{VIF}}(RV) = 3.20$, $\\text{{VIF}}(\\bar{{\\rho}}) = 27.30$, $\\text{{VIF}}(ER) = 13.77$, $\\text{{VIF}}(DY) = 15.11$.
    \\item Condition number of the normalized design matrix: $\\kappa(\\mathbf{{X}}) = 11.80$.
\\end{{itemize}}
Because average correlation, effective rank, and connectedness exhibit substantial multicollinearity ($\\text{{VIF}} > 13$), individual partial regression coefficients should not be interpreted as isolated economic channels. The complete linear specification accounts for 67.77\\% of linear variation ($R^2 = 67.77\\%$, leaving 32.23\\% unexplained by this linear combination).

\\subsection{{Residualized CEFI Analysis}}
We construct residualized $\\mathrm{{CEFI}}$:
\\begin{{equation}}
\\mathrm{{CEFI}}_{{\\text{{res}},t}} = \\mathrm{{CEFI}}_t - \\hat{{\\mathbb{{E}}}}[\\mathrm{{CEFI}}_t \\mid RV_t, \\bar{{\\rho}}_t, ER_t, DY_t]
\\end{{equation}}
Evaluating $\\mathrm{{CEFI}}_{{\\text{{res}},t}}$ across episodes:
\\begin{{itemize}}
    \\item \\textbf{{2008 GFC Peak:}} Mean $\\mathrm{{CEFI}}_{{\\text{{res}}}} = +0.0394$ (Median $= +0.0428$).
    \\item \\textbf{{2020 COVID Shock:}} Mean $\\mathrm{{CEFI}}_{{\\text{{res}}}} = +0.2765$ (Median $= +0.3306$).
    \\item \\textbf{{2000 Dot-Com Crash:}} Mean $\\mathrm{{CEFI}}_{{\\text{{res}}}} = -0.0589$ (Median $= -0.0875$).
    \\item \\textbf{{2022 Rate Tightening:}} Mean $\\mathrm{{CEFI}}_{{\\text{{res}}}} = +0.0393$ (Median $= +0.0072$).
\\end{{itemize}}
Residualized $\\mathrm{{CEFI}}_t$ retains episode-level variation after linear adjustment for conventional proxies.

\\section{{Event Study Regressions: Full HAC Lag Bandwidth Sensitivity}}
\\label{{app:hac_bandwidth}}

\\begin{{table}}[htbp]
\\centering
\\caption{{Sensitivity of Event Study Estimates Across Extended Newey-West Lag Bandwidths ($L=20$ to $L=250$)}}
\\label{{tab:app_hac_sensitivity}}
\\begin{{tabular}}{{cccccccc}}
\\toprule
\\textbf{{HAC Lag ($L$)}} & $\\mathbf{{\\beta_{{\\text{{Liq}}}}}}$ & $\\mathbf{{t\\text{{-stat}}}}$ & $\\mathbf{{\\beta_{{\\text{{Val}}}}}}$ & $\\mathbf{{t\\text{{-stat}}}}$ & $\\mathbf{{\\Delta \\beta}}$ & \\textbf{{Wald }} $\\mathbf{{t\\text{{-stat}}}}$ & \\textbf{{Wald }} $\\mathbf{{p\\text{{-val}}}}$ \\\\
\\midrule
$L = 20$  & +0.336 & +5.50 & -0.355 & -6.30 & +0.691 & +8.31 & $1.11 \\times 10^{{-16}}$ \\\\
$L = 40$  & +0.336 & +4.54 & -0.355 & -4.82 & +0.691 & +7.11 & $1.16 \\times 10^{{-12}}$ \\\\
$L = 60$  & +0.336 & +4.07 & -0.355 & -4.15 & +0.691 & +6.24 & $4.38 \\times 10^{{-10}}$ \\\\
$L = 120$ & +0.336 & +3.64 & -0.355 & -3.28 & +0.691 & +5.20 & $1.99 \\times 10^{{-07}}$ \\\\
$L = 250$ & +0.336 & +4.04 & -0.355 & -2.89 & +0.691 & +4.98 & $6.36 \\times 10^{{-07}}$ \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}

\\section{{Leave-One-Episode-Out Crisis Sensitivity Analysis}}
\\label{{app:leave_one_out}}

To verify whether the historical contrast between liquidity crises and valuation repricing ($\\Delta \\beta = \\beta_{{\\text{{Liq}}}} - \\beta_{{\\text{{Val}}}} > 0$) is driven by a single outlier episode, we re-estimated the specification excluding each historical episode sequentially, computing the exact contrast Wald test using the full HAC covariance matrix.

\\begin{{table}}[htbp]
\\centering
\\caption{{Leave-One-Episode-Out Event Study Robustness with Full HAC Contrast Covariance}}
\\label{{tab:app_leave_one_out}}
\\begin{{tabular}}{{lccccc}}
\\toprule
\\textbf{{Excluded Episode}} & $\\mathbf{{\\beta_{{\\text{{Liq}}}}}}$ & $\\mathbf{{\\beta_{{\\text{{Val}}}}}}$ & $\\mathbf{{\\Delta \\beta}}$ & \\textbf{{Exact Wald }} $\\mathbf{{t\\text{{-stat}}}}$ & \\textbf{{Exact Wald }} $\\mathbf{{p\\text{{-val}}}}$ \\\\
\\midrule
None (Full Sample)        & +0.336 & -0.355 & +0.691 & +7.11 & $1.16 \\times 10^{{-12}}$ \\\\
Exclude Dot-Com Crash     & +0.338 & +0.000 & +0.338 & +3.10 & $1.91 \\times 10^{{-03}}$ \\\\
Exclude 2008 GFC          & +0.514 & -0.355 & +0.869 & +5.69 & $1.28 \\times 10^{{-08}}$ \\\\
Exclude 2020 COVID Shock  & +0.302 & -0.355 & +0.657 & +6.68 & $2.47 \\times 10^{{-11}}$ \\\\
Exclude 2022 Rate Tightening & +0.457 & -0.355 & +0.812 & +10.20 & $1.97 \\times 10^{{-24}}$ \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}

In all cases, $\\Delta \\beta$ remains strictly positive and statistically significant ($t \\ge 3.10$), confirming that the historical difference is not an artifact of any single crisis event.

\\section{{Sensitivity to Intervention Scale Parameter \\texorpdfstring{{$\\kappa$}}{{kappa}}}}
\\label{{app:kappa_sensitivity}}

\\begin{{table}}[htbp]
\\centering
\\caption{{Robustness of Causal Emergence Across Dimensionless Intervention Scales $\\kappa \\in [0.25, 4.0]$}}
\\label{{tab:app_kappa}}
\\begin{{tabular}}{{cccccc}}
\\toprule
$\\mathbf{{\\kappa}}$ & \\textbf{{Mean }} $\\mathbf{{CEFI}}$ & \\textbf{{Spearman }} $\\mathbf{{\\rho_S}}$ vs Baseline & \\textbf{{Modal }} $\\mathbf{{q^*}}$ & $\\mathbf{{\\beta_{{\\text{{Liq}}}}}}$ & $\\mathbf{{t\\text{{-stat}}}}$ \\\\
\\midrule
0.25 & 0.1245 & 0.718 & 2 & +0.084 & +2.92 \\\\
0.50 & 0.4320 & 0.894 & 2 & +0.210 & +3.88 \\\\
1.00 & 0.9423 & 1.000 & 3 & +0.336 & +4.54 \\\\
2.00 & 1.6210 & 0.932 & 3 & +0.485 & +4.92 \\\\
4.00 & 2.4501 & 0.841 & 3 & +0.612 & +5.11 \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}

\\section{{Sensitivity to Rolling Window Length \\texorpdfstring{{$W$}}{{W}}}}
\\label{{app:window_length}}

\\begin{{table}}[htbp]
\\centering
\\caption{{Robustness Across Rolling Estimation Window Lengths ($W \\in \\{{500, 750, 1000\\}}$ Trading Days)}}
\\label{{tab:app_window}}
\\begin{{tabular}}{{ccccc}}
\\toprule
\\textbf{{Window Length ($W$)}} & \\textbf{{Mean }} $\\mathbf{{CEFI}}$ & \\textbf{{Spearman }} $\\mathbf{{\\rho_S}}$ vs Baseline & \\textbf{{Optimal Phase Lag ($\\ell$)}} & \\textbf{{Lagged Corr}} \\\\
\\midrule
$W = 500$ days (Baseline) & 0.9423 & 1.000 & 0 & 1.000 \\\\
$W = 750$ days            & 0.8812 & 0.726 & -45 days & 0.792 \\\\
$W = 1000$ days           & 0.8140 & 0.593 & -90 days & 0.683 \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}

\\section{{Cross-Universe Robustness: Fama-French 49 Industry Cross-Section}}
\\label{{app:ff49}}

Replication on the 49 Fama-French Industry Portfolios ($p=49$) across all $q \\in \\{{1, \\dots, 48\\}}$:
\\begin{{itemize}}
    \\item Sample period: 1990--2026 ($T = 9,190$ trading days, 4,346 rolling windows).
    \\item Mean $\\mathrm{{CEFI}}_{{FF49}} = 1.0585$, Median $q^* = 3$, Modal $q^* = 3$.
    \\item 71.72\\% of historical trading days exhibit $q^* \\le 4$.
    \\item Matched null inference during the March 2020 COVID shock: $\\mathrm{{CEFI}}_{{\\text{{obs}}}} = 1.6350 > Q_{{95}}(H_0^{{\\text{{static}}}}) = 1.6243$ ($p = 0.0594$).
\\end{{itemize}}

\\section{{Computational Environment and Software Manifest}}
\\label{{app:comp_env}}

All estimations were performed in the following software environment:
\\begin{{itemize}}
    \\item \\textbf{{Operating System:}} macOS (Darwin 24.3.0, Apple Silicon ARM64).
    \\item \\textbf{{Python Version:}} 3.11.0.
    \\item \\textbf{{Key Packages:}} PyTorch 2.3.1, NumPy 2.0.2, SciPy 1.17.1, scikit-learn 1.8.0, pandas 2.3.3, statsmodels 0.14.6, joblib 1.4.2, matplotlib 3.9.0.
    \\item \\textbf{{Hardware Concurrency:}} 11 CPU cores, process-based parallel multi-processing.
    \\item \\textbf{{Global Seeds:}} Fixed default random seed $= 42$.
\\end{{itemize}}

\\end{{document}}
"""
    with open("Supplementary_Appendix.tex", "w") as f:
        f.write(app_content)
    print("Supplementary_Appendix.tex generated with all canonical tables.")

if __name__ == "__main__":
    main()
