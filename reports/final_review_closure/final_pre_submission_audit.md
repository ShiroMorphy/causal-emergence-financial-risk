# Final Pre-Submission Audit Report: IRFA Master Consensus Closure

**Target Journal:** *International Review of Financial Analysis* (IRFA)  
**Paper Title:** Causal Emergence in Financial Markets: Dynamic Organization and Effective Dimensionality During Systemic Stress  
**Author:** Felipe Mora, M.Sc., Ph.D.(c) (ORCID: 0009-0001-1034-5948)  
**Status:** **READY FOR SUBMISSION - ALL 20 ACCEPTANCE GATES PASSED (STRICT PARITY & SINGLE SOURCE OF TRUTH)**

---

## 1. Executive Summary & Verification Gates

| Gate ID | Area / Dimension | Verification Criteria | Status |
| :--- | :--- | :--- | :--- |
| **G01** | Global Scale Invariance | Numerical invariance under $X 	o cX$ ($c \in [0.01, 10000]$) | **PASSED** ($|\Delta| < 10^{-14}$) |
| **G02** | VAR Orientation Recovery | Synthetic recovery error direct $\ll$ transposed ($0.0611$ vs $1.5897$) | **PASSED** (26x error ratio) |
| **G03** | Trailing Benchmark Windows | Exact 500-day trailing windows terminating at crisis dates | **PASSED** (2005-12-30, 2008-11-20, 2020-03-23) |
| **G04** | Optimization Budget Parity | Identical 4/35 search budget in observed and all $B=9,999$ surrogates | **PASSED** (Strictly identical budget) |
| **G05** | Primary Matched Null Family | $B=9,999$ Monte Carlo runs for $H_0^{\text{static}}$ and $H_0^{\text{diag+contemp}}$ | **PASSED** ($6 \times 9,999$ simulations) |
| **G06** | Holm-Bonferroni FWER | Exact step-down correction on $m=6$ primary hypothesis family | **PASSED** (Exact step-down multiplier) |
| **G07** | Leave-One-Out Exact Wald | Full Newey-West HAC contrast covariance ($R \hat{V}_{\text{HAC}} R^\top$) | **PASSED** ($t \ge 3.10$, $p < 0.002$) |
| **G08** | Interventional Channel Framing | $\mathbf{A}_M = \mathbf{W}\mathbf{A}\mathbf{W}^\top$ defined via canonical lifting $do(\mathbf{x}) = \mathbf{W}^\top do(\mathbf{y})$ | **PASSED** ($r_{\text{closure}} \approx 0.795$ documented) |
| **G09** | VAR(1) Dynamic Justification | One-step transfer capacity; low residual autocorrelation ($|r| < 0.045$) | **PASSED** (Faux VAR(2) claim removed) |
| **G10** | Stiefel Canonical Duality | $\langle \operatorname{grad}_{\mathcal{R}} f, \mathbf{\Delta} \rangle_{\text{canonical}} = \langle \mathbf{G}, \mathbf{\Delta} \rangle_{\text{Euclidean}}$ verified | **PASSED** (15/15 unit tests passing) |
| **G11** | Multicollinearity Diagnostics | Documented high VIFs ($> 13$) among conventional systemic proxies | **PASSED** (Caution against partial coeffs) |
| **G12** | Residualized CEFI Analysis | Verified episode-level variation persists after orthogonalization | **PASSED** (COVID $+0.2765$, GFC $+0.0394$) |
| **G13** | Benchmark Concordance | Evaluated against Liu et al. (2024, 2025) across 870 windows | **PASSED** ($\rho = 0.837, 0.832$) |
| **G14** | Cross-Universe FF49 | 49 industry portfolios ($p=49$) yield modal $q^* = 3$ | **PASSED** (71.7\% $q^* \le 4$) |
| **G15** | Unit Root Diagnostic | Spectral radius $\rho_t \le 0.678$ across all 4,346 windows (0\% unit roots) | **PASSED** (Strict stationarity) |
| **G16** | Double-Blind Anonymity | `manuscript.tex` and `Supplementary_Appendix.tex` anonymized | **PASSED** (No author metadata in text) |
| **G17** | Title Page & Metadata | Felipe Mora, M.Sc., Ph.D.(c), ORCID 0009-0001-1034-5948 | **PASSED** (Correct title & ORCID) |
| **G18** | Cover Letter Tone | First-person singular ("I submit my manuscript"), M.Sc., Ph.D.(c) | **PASSED** (Aligned with IRFA guidelines) |
| **G19** | Highlights Length | 5 bullet points, each $\le 85$ characters including spaces | **PASSED** (All bullets within limit) |
| **G20** | Single Source of Truth | Zero hardcoded stale numbers across LaTeX, CSV, and Markdown | **PASSED** (100\% programmatic sync) |

---

## 2. Canonical Statistical Estimates ($B=9,999$ Matched Null Family)

- **Calm Period (2005-12-30):** $\mathrm{CEFI}_{\text{obs}} = 0.9848$ ($q^* = 3$)
  - $H_0^{\text{static}}$: $p_{\text{raw}} = 0.1940$ (SE $= 0.0040$), $p_{\text{Holm}} = 0.3880$, $z = +0.85$, $\mathbb{E}[\mathrm{CEFI}_0] = 0.8587$
  - $H_0^{\text{diag+contemp}}$: $p_{\text{raw}} = 0.6215$ (SE $= 0.0049$), $p_{\text{Holm}} = 0.6215$, $z = -0.35$, $\mathbb{E}[\mathrm{CEFI}_0] = 1.0255$
- **2008 GFC Peak (2008-11-20):** $\mathrm{CEFI}_{\text{obs}} = 1.8948$ ($q^* = 2$)
  - $H_0^{\text{static}}$: $p_{\text{raw}} = 0.0014$ (SE $= 0.0004$), $p_{\text{Holm}} = 0.0042$, $z = +3.16$, $\mathbb{E}[\mathrm{CEFI}_0] = 1.3712$
  - $H_0^{\text{diag+contemp}}$: $p_{\text{raw}} = 0.0001$ (SE $= 0.0001$), $p_{\text{Holm}} = 0.0006$, $z = +4.57$, $\mathbb{E}[\mathrm{CEFI}_0] = 1.4832$
- **2020 COVID Shock Trough (2020-03-23):** $\mathrm{CEFI}_{\text{obs}} = 1.8787$ ($q^* = 2$)
  - $H_0^{\text{static}}$: $p_{\text{raw}} = 0.0001$ (SE $= 0.0001$), $p_{\text{Holm}} = 0.0006$, $z = +2.95$, $\mathbb{E}[\mathrm{CEFI}_0] = 1.3154$
  - $H_0^{\text{diag+contemp}}$: $p_{\text{raw}} = 0.0001$ (SE $= 0.0001$), $p_{\text{Holm}} = 0.0006$, $z = +6.22$, $\mathbb{E}[\mathrm{CEFI}_0] = 1.3977$

---

## 3. Submission Documents Generated

1. `manuscript.pdf`: Complete, compiled, double-blind anonymized main text (Table 1, Table 2, all figures embedded).
2. `Supplementary_Appendix.pdf`: 13-page technical appendix with formal proofs, diagnostics, and robustness checks.
3. `Title_Page.pdf`: Title page with author Felipe Mora, M.Sc., Ph.D.(c), Universidad Técnica Federico Santa María, and ORCID `0009-0001-1034-5948`.
4. `Cover_Letter.pdf`: Submission cover letter for *International Review of Financial Analysis*.
5. `Highlights.txt`: 5 concise bullet points ($\le 85$ characters).
