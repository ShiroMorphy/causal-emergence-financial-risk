#!/usr/bin/env python3
"""
Script 34: Canonical 12/100 Robustness Rerun & Exact Synchronization Engine
==========================================================================
Executes all secondary robustness pipelines under canonical 12 restarts / 100 iterations:
1. Kappa Sensitivity across kappa in {0.25, 0.50, 1.00, 2.00, 4.00} (4,346 windows).
2. Window Length Sensitivity across W in {500, 750, 1000} (full rolling windows).
3. q* vs. Effective Rank and PCA Dimensions (exact empirical correlations).
4. Observational Closure Diagnostic r_closure at optimal q*.
5. FF49 Cross-Universe Replication under 12/100.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import torch

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from causal_emergence.micro_var import fit_micro_var1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_RESTARTS = 12
MAX_ITER = 100

def evaluate_batch_cefi_cuda(A_batch, Sigma_eps_batch, Sigma_x_batch, p, q_candidates, n_restarts=12, max_iter=100, kappa=1.0):
    B = len(A_batch)
    A_t = torch.tensor(A_batch, dtype=torch.float64, device=DEVICE)
    S_eps_t = torch.tensor(Sigma_eps_batch, dtype=torch.float64, device=DEVICE)
    S_x_t = torch.tensor(Sigma_x_batch, dtype=torch.float64, device=DEVICE)

    I_p = torch.eye(p, dtype=torch.float64, device=DEVICE).unsqueeze(0).expand(B, -1, -1)
    S_clean = 0.5 * (S_eps_t + S_eps_t.transpose(-2, -1)) + 1e-10 * (torch.diagonal(S_eps_t, dim1=-2, dim2=-1).sum(-1, keepdim=True).unsqueeze(-1) / float(p)) * I_p
    S_eff_micro = (kappa ** 2) * (A_t @ S_x_t @ A_t.transpose(-2, -1)) + S_clean
    ei_micro = 0.5 * (torch.linalg.slogdet(S_eff_micro)[1] - torch.linalg.slogdet(S_clean)[1])
    micro_density = ei_micro / float(p)

    best_cefi = torch.full((B,), -1e9, dtype=torch.float64, device=DEVICE)
    best_q = torch.zeros(B, dtype=torch.long, device=DEVICE)

    for q in q_candidates:
        best_obj_q = torch.full((B,), -1e9, dtype=torch.float64, device=DEVICE)
        for start_idx in range(n_restarts):
            torch.manual_seed(42 + start_idx * 1000)
            W_raw = torch.randn(B, q, p, dtype=torch.float64, device=DEVICE)
            Q, _ = torch.linalg.qr(W_raw.transpose(-2, -1))
            W = Q.transpose(-2, -1)[:, :q, :].clone().requires_grad_(True)
            lr = 0.05
            for it in range(max_iter):
                A_M = W @ A_t @ W.transpose(-2, -1)
                S_eps_M = W @ S_eps_t @ W.transpose(-2, -1)
                S_x_M = W @ S_x_t @ W.transpose(-2, -1)
                S_eff_M = (kappa ** 2) * (A_M @ S_x_M @ A_M.transpose(-2, -1)) + S_eps_M
                obj = 0.5 * (torch.linalg.slogdet(S_eff_M)[1] - torch.linalg.slogdet(S_eps_M)[1])
                total_loss = -obj.sum()
                total_loss.backward()
                with torch.no_grad():
                    G = W.grad
                    W_step = W + lr * G
                    Q_ret, _ = torch.linalg.qr(W_step.transpose(-2, -1))
                    W_new = Q_ret.transpose(-2, -1)[:, :q, :]
                W = W_new.clone().requires_grad_(True)

            with torch.no_grad():
                A_M = W @ A_t @ W.transpose(-2, -1)
                S_eps_M = W @ S_eps_t @ W.transpose(-2, -1)
                S_x_M = W @ S_x_t @ W.transpose(-2, -1)
                S_eff_M = (kappa ** 2) * (A_M @ S_x_M @ A_M.transpose(-2, -1)) + S_eps_M
                final_obj = 0.5 * (torch.linalg.slogdet(S_eff_M)[1] - torch.linalg.slogdet(S_eps_M)[1])
                best_obj_q = torch.maximum(best_obj_q, final_obj)

        cefi_q = (best_obj_q / float(q)) - micro_density
        improved = cefi_q > best_cefi
        best_cefi = torch.where(improved, cefi_q, best_cefi)
        best_q = torch.where(improved, torch.full_like(best_q, q), best_q)

    return best_cefi.cpu().numpy(), best_q.cpu().numpy()


def run_kappa_sensitivity():
    print("=" * 80)
    print("1. RUNNING KAPPA SENSITIVITY UNDER 12/100 (4,346 WINDOWS)")
    print("=" * 80)
    df_returns = pd.read_csv("data/raw/ff30_daily_returns.csv", parse_dates=["Date"], index_col="Date")
    W = 500
    step = 1
    T_total, p = df_returns.shape
    dates = df_returns.index
    q_all = list(range(1, p))

    windows = [df_returns.iloc[t_end - W : t_end].values for t_end in range(W, T_total + 1, step)]
    task_dates = [dates[t_end - 1] for t_end in range(W, T_total + 1, step)]

    A_batch = []
    S_eps_batch = []
    S_x_batch = []
    for w in windows:
        A_m, S_eps_m = fit_micro_var1(w)
        S_x_m = np.cov(w, rowvar=False)
        A_batch.append(A_m)
        S_eps_batch.append(S_eps_m)
        S_x_batch.append(S_x_m)

    A_batch = np.stack(A_batch, axis=0)
    S_eps_batch = np.stack(S_eps_batch, axis=0)
    S_x_batch = np.stack(S_x_batch, axis=0)

    episodes = {
        "Dot-Com Crash": ("2000-03-01", "2002-10-09", "Valuation Repricing"),
        "2008 GFC Peak": ("2007-10-01", "2009-03-31", "Systemic Liquidity"),
        "2020 COVID Shock": ("2020-02-01", "2020-04-30", "Systemic Liquidity"),
        "2022 Rate Tightening": ("2022-01-01", "2022-12-31", "Valuation Repricing")
    }

    # Baseline 12/100 series (kappa = 1.0)
    df_base = pd.read_csv("data/features/cefi_series_12_100.csv", parse_dates=["date"]).rename(columns={"date": "Date"}).set_index("Date")
    base_cefi = df_base["cefi"].values

    kappas = [0.25, 0.50, 1.00, 2.00, 4.00]
    records = []

    for k in kappas:
        t0 = time.time()
        print(f"Evaluating kappa = {k:.2f}...")
        chunk = 512
        cefi_chunks = []
        q_chunks = []
        for i in range(0, len(A_batch), chunk):
            c_s, q_s = evaluate_batch_cefi_cuda(
                A_batch[i:i+chunk], S_eps_batch[i:i+chunk], S_x_batch[i:i+chunk],
                p, q_all, n_restarts=N_RESTARTS, max_iter=MAX_ITER, kappa=k
            )
            cefi_chunks.append(c_s)
            q_chunks.append(q_s)
        cefi_k = np.concatenate(cefi_chunks)
        q_k = np.concatenate(q_chunks)

        s_corr = stats.spearmanr(base_cefi, cefi_k)[0]
        modal_q = int(pd.Series(q_k).mode()[0])

        df_k = pd.DataFrame({"cefi": cefi_k, "Date": task_dates}).set_index("Date")
        df_k["is_liquidity"] = 0
        df_k["is_valuation"] = 0
        for ep_name, (s_date, e_date, ep_type) in episodes.items():
            mask = (df_k.index >= s_date) & (df_k.index <= e_date)
            if ep_type == "Systemic Liquidity":
                df_k.loc[mask, "is_liquidity"] = 1
            elif ep_type == "Valuation Repricing":
                df_k.loc[mask, "is_valuation"] = 1

        X = sm.add_constant(df_k[["is_liquidity", "is_valuation"]].values)
        res_hac = sm.OLS(df_k["cefi"].values, X).fit(cov_type="HAC", cov_kwds={"maxlags": 40})
        b_liq = res_hac.params[1]
        t_liq = res_hac.tvalues[1]

        records.append({
            "kappa": k,
            "Mean_CEFI": float(np.mean(cefi_k)),
            "Spearman_rho_vs_Baseline": float(s_corr),
            "Modal_q": modal_q,
            "beta_Liq": float(b_liq),
            "t_stat_Liq": float(t_liq)
        })
        print(f"  kappa = {k:.2f} done in {time.time()-t0:.1f}s | Mean CEFI = {np.mean(cefi_k):.4f}, Modal q = {modal_q}, beta_Liq = {b_liq:.4f} (t={t_liq:.2f})")

    df_kappa = pd.DataFrame(records)
    df_kappa.to_csv("reports/tables/table_kappa_sensitivity.csv", index=False)
    print("\nTable A13: Kappa Sensitivity (12/100):")
    print(df_kappa.to_string(index=False))


def run_window_length_sensitivity():
    print("\n" + "=" * 80)
    print("2. RUNNING WINDOW LENGTH SENSITIVITY UNDER 12/100 (W in 500, 750, 1000)")
    print("=" * 80)
    df_returns = pd.read_csv("data/raw/ff30_daily_returns.csv", parse_dates=["Date"], index_col="Date")
    p = df_returns.shape[1]
    q_all = list(range(1, p))

    lengths = [500, 750, 1000]
    series_dict = {}

    for W in lengths:
        t0 = time.time()
        T_total = len(df_returns)
        dates = df_returns.index
        windows = [df_returns.iloc[t_end - W : t_end].values for t_end in range(W, T_total + 1, 1)]
        task_dates = [dates[t_end - 1] for t_end in range(W, T_total + 1, 1)]

        A_batch = []
        S_eps_batch = []
        S_x_batch = []
        for w in windows:
            A_m, S_eps_m = fit_micro_var1(w)
            S_x_m = np.cov(w, rowvar=False)
            A_batch.append(A_m)
            S_eps_batch.append(S_eps_m)
            S_x_batch.append(S_x_m)

        A_batch = np.stack(A_batch, axis=0)
        S_eps_batch = np.stack(S_eps_batch, axis=0)
        S_x_batch = np.stack(S_x_batch, axis=0)

        chunk = 512
        cefi_chunks = []
        for i in range(0, len(A_batch), chunk):
            c_s, _ = evaluate_batch_cefi_cuda(
                A_batch[i:i+chunk], S_eps_batch[i:i+chunk], S_x_batch[i:i+chunk],
                p, q_all, n_restarts=N_RESTARTS, max_iter=MAX_ITER, kappa=1.0
            )
            cefi_chunks.append(c_s)
        cefi_w = np.concatenate(cefi_chunks)
        df_w = pd.DataFrame({"cefi": cefi_w, "Date": task_dates}).set_index("Date")
        series_dict[W] = df_w
        print(f"  W = {W} done in {time.time()-t0:.1f}s | Mean CEFI = {cefi_w.mean():.4f}")

    base_df = series_dict[500]
    records = []
    for W in lengths:
        w_df = series_dict[W]
        merged = base_df.join(w_df, lsuffix="_base", rsuffix="_alt", how="inner").dropna()
        s_corr = stats.spearmanr(merged["cefi_base"], merged["cefi_alt"])[0]

        best_lag = 0
        best_corr = float(stats.pearsonr(merged["cefi_base"], merged["cefi_alt"])[0])
        for lag in range(-150, 151, 5):
            shifted = merged["cefi_alt"].shift(lag)
            valid = pd.concat([merged["cefi_base"], shifted], axis=1).dropna()
            r_val = float(stats.pearsonr(valid.iloc[:, 0], valid.iloc[:, 1])[0])
            if r_val > best_corr:
                best_corr = r_val
                best_lag = lag

        records.append({
            "Window_Length": f"W = {W} days" if W != 500 else "W = 500 days (Baseline)",
            "Mean_CEFI": float(w_df["cefi"].mean()),
            "Spearman_rho_vs_Baseline": float(s_corr),
            "Optimal_Phase_Lag": f"{best_lag} days" if best_lag != 0 else "0",
            "Lagged_Corr": float(best_corr)
        })

    df_window = pd.DataFrame(records)
    df_window.to_csv("reports/tables/table_window_length_sensitivity.csv", index=False)
    print("\nTable A14: Window Length Sensitivity (12/100):")
    print(df_window.to_string(index=False))


def run_q_star_diagnostics():
    print("\n" + "=" * 80)
    print("3. RUNNING EXACT Q* VS EFFECTIVE RANK AND PCA CORRELATIONS (12/100)")
    print("=" * 80)
    df_cefi = pd.read_csv("data/features/cefi_series_12_100.csv", parse_dates=["date"]).rename(columns={"date": "Date"}).set_index("Date")
    df_returns = pd.read_csv("data/raw/ff30_daily_returns.csv", parse_dates=["Date"], index_col="Date")
    W = 500
    T_total = len(df_returns)

    er_list = []
    pca80_list = []
    pca90_list = []
    for t_end in range(W, T_total + 1):
        window = df_returns.iloc[t_end - W : t_end].values
        cov_mat = np.cov(window, rowvar=False)
        vals, _ = np.linalg.eigh(cov_mat)
        vals = np.maximum(vals, 1e-12)
        p_dist = vals / vals.sum()
        er = float(np.exp(-np.sum(p_dist * np.log(p_dist))))
        er_list.append(er)

        vals_sorted = np.sort(vals)[::-1]
        cum_var = np.cumsum(vals_sorted) / vals_sorted.sum()
        pca80 = int(np.searchsorted(cum_var, 0.80) + 1)
        pca90 = int(np.searchsorted(cum_var, 0.90) + 1)
        pca80_list.append(pca80)
        pca90_list.append(pca90)

    q_star = df_cefi["q_star"].values
    er_arr = np.array(er_list)
    pca80_arr = np.array(pca80_list)
    pca90_arr = np.array(pca90_list)

    s_er = stats.spearmanr(q_star, er_arr)[0]
    p_er = stats.pearsonr(q_star, er_arr)[0]
    s_pca80 = stats.spearmanr(q_star, pca80_arr)[0]
    s_pca90 = stats.spearmanr(q_star, pca90_arr)[0]

    print(f"Spearman rho(q*, ER)    = {s_er:.4f}")
    print(f"Pearson rho(q*, ER)     = {p_er:.4f}")
    print(f"Spearman rho(q*, PCA80) = {s_pca80:.4f}")
    print(f"Spearman rho(q*, PCA90) = {s_pca90:.4f}")


def run_closure_diagnostics():
    print("\n" + "=" * 80)
    print("4. RUNNING OBSERVATIONAL CLOSURE DIAGNOSTIC AT OPTIMAL Q* (12/100)")
    print("=" * 80)
    df_returns = pd.read_csv("data/raw/ff30_daily_returns.csv", parse_dates=["Date"], index_col="Date")
    p = df_returns.shape[1]

    benchmarks = {
        "2005 Calm Market Benchmark": "2005-12-30",
        "2008 GFC Peak (Nov 2008)": "2008-11-20",
        "2020 COVID Crash (Mar 2020)": "2020-03-23",
        "2000 Dot-Com Crash": "2001-03-20",
        "2022 Rate Tightening": "2022-06-15"
    }

    closure_records = []
    for label, d_str in benchmarks.items():
        sub = df_returns.loc[:d_str]
        w = sub.iloc[-500:].values
        A_m, S_eps_m = fit_micro_var1(w)
        S_x_m = np.cov(w, rowvar=False)

        q_opt = 1
        A_t = torch.tensor(A_m[None, ...], dtype=torch.float64, device=DEVICE)
        S_eps_t = torch.tensor(S_eps_m[None, ...], dtype=torch.float64, device=DEVICE)
        S_x_t = torch.tensor(S_x_m[None, ...], dtype=torch.float64, device=DEVICE)

        best_W = None
        best_obj = -1e9
        for start_idx in range(N_RESTARTS):
            torch.manual_seed(42 + start_idx * 1000)
            W_raw = torch.randn(1, q_opt, p, dtype=torch.float64, device=DEVICE)
            Q, _ = torch.linalg.qr(W_raw.transpose(-2, -1))
            W = Q.transpose(-2, -1)[:, :q_opt, :].clone().requires_grad_(True)
            for it in range(MAX_ITER):
                A_M = W @ A_t @ W.transpose(-2, -1)
                S_eps_M = W @ S_eps_t @ W.transpose(-2, -1)
                S_x_M = W @ S_x_t @ W.transpose(-2, -1)
                S_eff_M = (A_M @ S_x_M @ A_M.transpose(-2, -1)) + S_eps_M
                obj = 0.5 * (torch.linalg.slogdet(S_eff_M)[1] - torch.linalg.slogdet(S_eps_M)[1])
                (-obj.sum()).backward()
                with torch.no_grad():
                    W_step = W + 0.05 * W.grad
                    Q_ret, _ = torch.linalg.qr(W_step.transpose(-2, -1))
                    W_new = Q_ret.transpose(-2, -1)[:, :q_opt, :]
                W = W_new.clone().requires_grad_(True)

            with torch.no_grad():
                A_M = W @ A_t @ W.transpose(-2, -1)
                S_eps_M = W @ S_eps_t @ W.transpose(-2, -1)
                S_x_M = W @ S_x_t @ W.transpose(-2, -1)
                S_eff_M = (A_M @ S_x_M @ A_M.transpose(-2, -1)) + S_eps_M
                final_obj = float(0.5 * (torch.linalg.slogdet(S_eff_M)[1] - torch.linalg.slogdet(S_eps_M)[1]).cpu().item())
                if final_obj > best_obj:
                    best_obj = final_obj
                    best_W = W[0].cpu().numpy()

        WA = best_W @ A_m
        WA_proj = WA @ best_W.T @ best_W
        r_closure = float(np.linalg.norm(WA - WA_proj, ord="fro") / np.linalg.norm(WA, ord="fro"))

        closure_records.append({
            "Regime": label,
            "r_closure": r_closure,
            "Micro_p": p,
            "Macro_q": q_opt
        })
        print(f"  {label} (q*={q_opt}): r_closure = {r_closure:.4f}")

    df_closure = pd.DataFrame(closure_records)
    df_closure.to_csv("reports/tables/table_closure_diagnostics.csv", index=False)


def run_ff49_rerun():
    print("\n" + "=" * 80)
    print("5. RUNNING FF49 CROSS-UNIVERSE REPLICATION UNDER 12/100")
    print("=" * 80)
    raw_path = "data/raw/ff49_daily_returns.csv"
    if not os.path.exists(raw_path):
        print(f"Error: {raw_path} not found.")
        return

    df_ff49 = pd.read_csv(raw_path, parse_dates=["Date"], index_col="Date")
    T_total, p = df_ff49.shape
    q_all = list(range(1, p))
    W = 500
    step = 10
    dates = df_ff49.index

    windows = [df_ff49.iloc[t_end - W : t_end].values for t_end in range(W, T_total + 1, step)]
    task_dates = [dates[t_end - 1] for t_end in range(W, T_total + 1, step)]

    print(f"Evaluating {len(windows)} slices of FF49 across q in 1..48 under 12/100...")
    A_batch = []
    S_eps_batch = []
    S_x_batch = []
    for w in windows:
        A_m, S_eps_m = fit_micro_var1(w)
        S_x_m = np.cov(w, rowvar=False)
        A_batch.append(A_m)
        S_eps_batch.append(S_eps_m)
        S_x_batch.append(S_x_m)

    A_batch = np.stack(A_batch, axis=0)
    S_eps_batch = np.stack(S_eps_batch, axis=0)
    S_x_batch = np.stack(S_x_batch, axis=0)

    chunk = 128
    cefi_chunks = []
    q_chunks = []
    t0 = time.time()
    for i in range(0, len(A_batch), chunk):
        c_s, q_s = evaluate_batch_cefi_cuda(
            A_batch[i:i+chunk], S_eps_batch[i:i+chunk], S_x_batch[i:i+chunk],
            p, q_all, n_restarts=N_RESTARTS, max_iter=MAX_ITER, kappa=1.0
        )
        cefi_chunks.append(c_s)
        q_chunks.append(q_s)

    cefi_ff49 = np.concatenate(cefi_chunks)
    q_ff49 = np.concatenate(q_chunks)

    df_out = pd.DataFrame({"cefi": cefi_ff49, "q_star": q_ff49, "Date": task_dates}).set_index("Date")
    df_out.to_csv("data/features/cefi_ff49_daily_series.csv")

    mean_c = float(cefi_ff49.mean())
    median_c = float(np.median(cefi_ff49))
    modal_q = int(pd.Series(q_ff49).mode()[0])
    pct_le4 = float((q_ff49 <= 4).mean() * 100.0)

    print(f"FF49 Replicated in {time.time()-t0:.1f}s:")
    print(f"  Mean CEFI: {mean_c:.4f}, Median CEFI: {median_c:.4f}, Modal q*: {modal_q}, P(q* <= 4): {pct_le4:.2f}%")


def main():
    print("=" * 80)
    print("STARTING CANONICAL 12/100 ROBUSTNESS RERUN SUITE")
    print("=" * 80)
    t_start = time.time()
    run_kappa_sensitivity()
    run_window_length_sensitivity()
    run_q_star_diagnostics()
    run_closure_diagnostics()
    run_ff49_rerun()
    print("\n" + "=" * 80)
    print(f"ALL ROBUSTNESS CALCULATIONS COMPLETED IN {time.time()-t_start:.1f}s!")
    print("=" * 80)

if __name__ == "__main__":
    main()
