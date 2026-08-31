#!/usr/bin/env python3
"""
Script 18: Strict Matched Null Inference with Exact Trailing Windows and Identical Optimizer Budget
==================================================================================================
Guarantees:
1. Exact trailing 500-day windows ending on benchmark dates:
   - Calm 2005/2006: 2005-12-30 (or 2006-12-29)
   - 2008 GFC Peak: 2008-11-20
   - 2020 COVID Crash: 2020-03-23
2. Strictly identical optimizer budget for observed CEFI and all surrogates:
   n_restarts = 4, max_iter = 35
3. B = 9,999 for primary nulls (H0_static and H0_diag+contemp)
   B = 999 for auxiliary nulls (H0_circ and H0_diag)
4. Primary test family (m=6) with Holm-Bonferroni correction and Monte Carlo SE
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from causal_emergence.micro_var import fit_micro_var1
from causal_emergence.analytical_ei import compute_continuous_ei, compute_emergence_spectrum
from causal_emergence.stiefel_optimizer import optimize_coarse_graining_stiefel
from causal_emergence.null_models import (
    generate_circular_null_data,
    generate_diagonal_var_null_data,
    generate_static_correlation_null_data,
    generate_diag_plus_contemp_null_data,
    evaluate_single_null_realization
)


def compute_observed_cefi_strict(window: np.ndarray, q_candidates: list, seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    T, p = window.shape
    A_obs, S_eps_obs = fit_micro_var1(window)
    S_x_obs = np.cov(window, rowvar=False)

    ei_m_obs = compute_continuous_ei(A_obs, S_eps_obs, Sigma_x=S_x_obs, kappa_do=1.0)
    macro_eis_obs = {}
    for q in q_candidates:
        _, ei_q = optimize_coarse_graining_stiefel(
            A_obs, S_eps_obs, q=q, Sigma_x=S_x_obs, kappa_do=1.0, n_restarts=4, max_iter=35
        )
        macro_eis_obs[q] = ei_q

    cefi_obs, q_obs, _, _ = compute_emergence_spectrum(ei_m_obs, macro_eis_obs, p_micro=p)
    return cefi_obs, q_obs, A_obs, S_eps_obs


def main():
    print("=" * 90)
    print("STARTING STRICT MATCHED NULL INFERENCE (B=9,999, 4 RESTARTS / 35 ITERS IN OBS AND NULL)")
    print("=" * 90)

    df = pd.read_csv("data/raw/ff30_daily_returns.csv", parse_dates=["Date"], index_col="Date")
    p = df.shape[1]
    q_all = list(range(1, p))

    # Exact benchmark end dates
    benchmarks = [
        ("Calm Period (2005)", "2005-12-30"),
        ("2008 GFC Peak", "2008-11-20"),
        ("2020 COVID Shock", "2020-03-23")
    ]

    B_primary = 9999
    B_aux = 999

    primary_records = []
    full_records = []

    for label, end_date in benchmarks:
        end_loc = df.index.get_indexer([pd.to_datetime(end_date)], method="nearest")[0]
        actual_date = df.index[end_loc].strftime("%Y-%m-%d")
        window = df.iloc[end_loc - 500 + 1 : end_loc + 1].values
        assert len(window) == 500, f"Window length is {len(window)}, expected 500"

        cefi_obs, q_obs, A_obs, S_eps_obs = compute_observed_cefi_strict(window, q_all, seed=42)
        print(f"\n>>> Regime [{label}] (End Date: {actual_date}): Observed CEFI = {cefi_obs:.4f}, q* = {q_obs}")

        # 1. H0_static (B=9,999) with 4 restarts / 35 iters
        print(f"  Evaluating H0_static (B={B_primary}, restarts=4, iters=35)...")
        t0 = time.time()
        def _eval_static(s):
            np.random.seed(s)
            X_s = generate_static_correlation_null_data(window)
            return evaluate_single_null_realization(X_s, q_candidates=q_all, kappa_do=1.0, n_restarts=4, max_iter=35)

        seeds_static = np.random.randint(1000, 9999999, size=B_primary)
        res_static = Parallel(n_jobs=-1)(delayed(_eval_static)(s) for s in seeds_static)
        null_static = np.array([r[0] for r in res_static])
        q_static = np.array([r[1] for r in res_static])

        p_static = float((1.0 + np.sum(null_static >= cefi_obs)) / (B_primary + 1.0))
        se_static = float(np.sqrt(p_static * (1.0 - p_static) / B_primary))
        z_static = float((cefi_obs - np.mean(null_static)) / np.std(null_static))
        q95_static = float(np.percentile(null_static, 95))
        print(f"    Done in {time.time()-t0:.1f}s | p_emp = {p_static:.4f} (SE={se_static:.4f}), z = {z_static:+.2f}, E[CEFI_0] = {np.mean(null_static):.4f}")

        # 2. H0_diag+contemp (B=9,999) with 4 restarts / 35 iters
        print(f"  Evaluating H0_diag+contemp (B={B_primary}, restarts=4, iters=35)...")
        t0 = time.time()
        def _eval_dc(s):
            np.random.seed(s)
            X_dc = generate_diag_plus_contemp_null_data(window, A_obs, S_eps_obs)
            return evaluate_single_null_realization(X_dc, q_candidates=q_all, kappa_do=1.0, n_restarts=4, max_iter=35)

        seeds_dc = np.random.randint(1000, 9999999, size=B_primary)
        res_dc = Parallel(n_jobs=-1)(delayed(_eval_dc)(s) for s in seeds_dc)
        null_dc = np.array([r[0] for r in res_dc])
        q_dc = np.array([r[1] for r in res_dc])

        p_dc = float((1.0 + np.sum(null_dc >= cefi_obs)) / (B_primary + 1.0))
        se_dc = float(np.sqrt(p_dc * (1.0 - p_dc) / B_primary))
        z_dc = float((cefi_obs - np.mean(null_dc)) / np.std(null_dc))
        q95_dc = float(np.percentile(null_dc, 95))
        print(f"    Done in {time.time()-t0:.1f}s | p_emp = {p_dc:.4f} (SE={se_dc:.4f}), z = {z_dc:+.2f}, E[CEFI_0] = {np.mean(null_dc):.4f}")

        # 3. Auxiliary Nulls (B=999) with 4 restarts / 35 iters
        print(f"  Evaluating Auxiliary Nulls H0_circ & H0_diag (B={B_aux}, restarts=4, iters=35)...")
        def _eval_circ(s):
            np.random.seed(s)
            X_c = generate_circular_null_data(window)
            return evaluate_single_null_realization(X_c, q_candidates=q_all, kappa_do=1.0, n_restarts=4, max_iter=35)

        def _eval_diag(s):
            np.random.seed(s)
            X_d = generate_diagonal_var_null_data(window, A_obs, S_eps_obs)
            return evaluate_single_null_realization(X_d, q_candidates=q_all, kappa_do=1.0, n_restarts=4, max_iter=35)

        seeds_circ = np.random.randint(1000, 9999999, size=B_aux)
        seeds_diag = np.random.randint(1000, 9999999, size=B_aux)
        res_circ = Parallel(n_jobs=-1)(delayed(_eval_circ)(s) for s in seeds_circ)
        res_diag = Parallel(n_jobs=-1)(delayed(_eval_diag)(s) for s in seeds_diag)

        null_circ = np.array([r[0] for r in res_circ])
        null_diag = np.array([r[0] for r in res_diag])

        p_circ = float((1.0 + np.sum(null_circ >= cefi_obs)) / (B_aux + 1.0))
        z_circ = float((cefi_obs - np.mean(null_circ)) / np.std(null_circ))
        q95_circ = float(np.percentile(null_circ, 95))

        p_diag = float((1.0 + np.sum(null_diag >= cefi_obs)) / (B_aux + 1.0))
        z_diag = float((cefi_obs - np.mean(null_diag)) / np.std(null_diag))
        q95_diag = float(np.percentile(null_diag, 95))

        # Store results
        primary_records.append({
            "Regime": label,
            "Benchmark_End_Date": actual_date,
            "CEFI_obs": cefi_obs,
            "q_obs": q_obs,
            "p_static_raw": p_static,
            "mc_se_static": se_static,
            "z_static": z_static,
            "mean_static": float(np.mean(null_static)),
            "q95_static": q95_static,
            "modal_q_static": int(pd.Series(q_static).mode()[0]),
            "p_dc_raw": p_dc,
            "mc_se_dc": se_dc,
            "z_dc": z_dc,
            "mean_dc": float(np.mean(null_dc)),
            "q95_dc": q95_dc,
            "modal_q_dc": int(pd.Series(q_dc).mode()[0])
        })

        full_records.extend([
            {"Regime": label, "Null_Model": "H0_circ", "B": B_aux, "CEFI_obs": cefi_obs, "q_obs": q_obs, "Mean_0": float(np.mean(null_circ)), "Q95_0": q95_circ, "z_dev": z_circ, "p_raw": p_circ, "modal_q_0": int(pd.Series([r[1] for r in res_circ]).mode()[0])},
            {"Regime": label, "Null_Model": "H0_diag", "B": B_aux, "CEFI_obs": cefi_obs, "q_obs": q_obs, "Mean_0": float(np.mean(null_diag)), "Q95_0": q95_diag, "z_dev": z_diag, "p_raw": p_diag, "modal_q_0": int(pd.Series([r[1] for r in res_diag]).mode()[0])},
            {"Regime": label, "Null_Model": "H0_static", "B": B_primary, "CEFI_obs": cefi_obs, "q_obs": q_obs, "Mean_0": float(np.mean(null_static)), "Q95_0": q95_static, "z_dev": z_static, "p_raw": p_static, "modal_q_0": int(pd.Series(q_static).mode()[0])},
            {"Regime": label, "Null_Model": "H0_diag+contemp", "B": B_primary, "CEFI_obs": cefi_obs, "q_obs": q_obs, "Mean_0": float(np.mean(null_dc)), "Q95_0": q95_dc, "z_dev": z_dc, "p_raw": p_dc, "modal_q_0": int(pd.Series(q_dc).mode()[0])}
        ])

    # 4. Primary Family Holm-Bonferroni Correction (m=6)
    p_values_primary = []
    for r in primary_records:
        p_values_primary.append((r["Regime"], "H0_static", r["p_static_raw"]))
        p_values_primary.append((r["Regime"], "H0_diag+contemp", r["p_dc_raw"]))

    # Sort p-values ascending for Holm step-down
    p_values_sorted = sorted(p_values_primary, key=lambda x: x[2])
    m = len(p_values_sorted)  # m = 6
    holm_adjusted = {}
    running_max = 0.0
    for rank, (regime, null_name, p_val) in enumerate(p_values_sorted):
        multiplier = m - rank
        adj = min(1.0, multiplier * p_val)
        running_max = max(running_max, adj)
        holm_adjusted[(regime, null_name)] = running_max

    for r in primary_records:
        r["p_static_holm"] = holm_adjusted[(r["Regime"], "H0_static")]
        r["p_dc_holm"] = holm_adjusted[(r["Regime"], "H0_diag+contemp")]

    for row in full_records:
        if (row["Regime"], row["Null_Model"]) in holm_adjusted:
            row["p_Holm"] = holm_adjusted[(row["Regime"], row["Null_Model"])]
        else:
            row["p_Holm"] = None

    # Save to CSV
    os.makedirs("reports/tables", exist_ok=True)
    df_primary = pd.DataFrame(primary_records)
    df_full = pd.DataFrame(full_records)

    df_primary.to_csv("reports/tables/primary_null_inference_b9999.csv", index=False)
    df_full.to_csv("reports/tables/full_null_inference_summary.csv", index=False)

    print("\n" + "=" * 90)
    print("FINAL STRICT MATCHED NULL INFERENCE RESULTS TABLE")
    print("=" * 90)
    for r in primary_records:
        print(f"\nRegime: [{r['Regime']}] (End Date: {r['Benchmark_End_Date']}) | Observed CEFI = {r['CEFI_obs']:.4f} (q* = {r['q_obs']})")
        print(f"  H0_static:       p_raw = {r['p_static_raw']:.4f} (SE={r['mc_se_static']:.4f}) | Holm p = {r['p_static_holm']:.4f} | z = {r['z_static']:+.2f} | E[CEFI_0] = {r['mean_static']:.4f}")
        print(f"  H0_diag+contemp: p_raw = {r['p_dc_raw']:.4f} (SE={r['mc_se_dc']:.4f}) | Holm p = {r['p_dc_holm']:.4f} | z = {r['z_dc']:+.2f} | E[CEFI_0] = {r['mean_dc']:.4f}")

    print("\n>>> Results successfully saved to reports/tables/primary_null_inference_b9999.csv and full_null_inference_summary.csv")


if __name__ == "__main__":
    main()
