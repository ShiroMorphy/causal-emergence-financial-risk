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
import hashlib
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import torch

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from causal_emergence.micro_var import fit_micro_var1
from causal_emergence.cuda_stiefel import ESTIMATOR_SPEC, estimator_fingerprint, evaluate_batch_cefi

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_RESTARTS = 12
MAX_ITER = 100


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_canonical_series():
    df = pd.read_csv("data/features/cefi_series_12_100.csv", parse_dates=["date"])
    required = {
        "date", "cefi", "q_star", "estimator_spec", "estimator_sha256",
        "micro_var_sha256", "data_sha256",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Canonical series lacks provenance columns: {sorted(missing)}")
    if set(df["estimator_spec"].dropna()) != {ESTIMATOR_SPEC}:
        raise ValueError("Canonical series was produced by a different estimator specification")
    if set(df["estimator_sha256"].dropna()) != {estimator_fingerprint()}:
        raise ValueError("Canonical series estimator hash does not match the running code")
    if set(df["micro_var_sha256"].dropna()) != {_sha256("src/causal_emergence/micro_var.py")}:
        raise ValueError("Canonical series micro-VAR hash does not match the running code")
    if set(df["data_sha256"].dropna()) != {_sha256("data/raw/ff30_daily_returns.csv")}:
        raise ValueError("Canonical series input-data hash does not match the local data")
    if df["date"].duplicated().any():
        raise ValueError("Canonical series contains duplicate dates")
    return df

def evaluate_batch_cefi_cuda(A_batch, Sigma_eps_batch, Sigma_x_batch, p, q_candidates, n_restarts=12, max_iter=100, kappa=1.0):
    del p  # inferred and validated by the shared canonical implementation
    cefi, q_star, _, _, _ = evaluate_batch_cefi(
        A_batch,
        Sigma_eps_batch,
        Sigma_x_batch,
        q_candidates,
        n_restarts=n_restarts,
        max_iter=max_iter,
        kappa=kappa,
        device=DEVICE,
        search_dtype=torch.float64,
    )
    return cefi, q_star


def run_kappa_sensitivity():
    print("=" * 80)
    print("1. RUNNING KAPPA SENSITIVITY UNDER 12/100 (4,346 WINDOWS)")
    print("=" * 80)
    df_returns = pd.read_csv("data/raw/ff30_daily_returns.csv", parse_dates=["Date"], index_col="Date")
    W = 500
    step = 2
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

    from causal_emergence.episodes import get_episodes_tuples
    episodes = get_episodes_tuples()

    # Baseline 12/100 series (kappa = 1.0)
    df_base = _load_canonical_series().rename(columns={"date": "Date"}).set_index("Date")
    base_cefi = df_base["cefi"].reindex(pd.DatetimeIndex(task_dates)).to_numpy()

    kappas = [0.25, 0.50, 1.00, 2.00, 4.00]
    records = []

    for k in kappas:
        t0 = time.time()
        print(f"Evaluating kappa = {k:.2f}...")
        checkpoint = (
            f"data/features/cefi_kappa_{k:.2f}_12_100_"
            f"{estimator_fingerprint()[:12]}.csv"
        )
        if k == 1.0:
            cefi_k = base_cefi.copy()
            q_k = df_base["q_star"].reindex(pd.DatetimeIndex(task_dates)).to_numpy(dtype=int)
        elif os.path.exists(checkpoint):
            saved = pd.read_csv(checkpoint, parse_dates=["Date"]).set_index("Date")
            saved = saved.reindex(pd.DatetimeIndex(task_dates))
            valid = (
                saved[["cefi", "q_star"]].notna().all().all()
                and set(saved["estimator_spec"].dropna()) == {ESTIMATOR_SPEC}
                and set(saved["estimator_sha256"].dropna()) == {estimator_fingerprint()}
            )
            if valid:
                cefi_k = saved["cefi"].to_numpy()
                q_k = saved["q_star"].to_numpy(dtype=int)
                print(f"  Resuming from {checkpoint}")
            else:
                raise ValueError(f"Incomplete checkpoint: {checkpoint}")
        else:
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
            pd.DataFrame({
                "Date": task_dates,
                "cefi": cefi_k,
                "q_star": q_k,
                "estimator_spec": ESTIMATOR_SPEC,
                "estimator_sha256": estimator_fingerprint(),
            }).to_csv(checkpoint, index=False)

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
        pd.DataFrame(records).to_csv("reports/tables/table_kappa_sensitivity.csv", index=False)
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
        windows = [df_returns.iloc[t_end - W : t_end].values for t_end in range(W, T_total + 1, 2)]
        task_dates = [dates[t_end - 1] for t_end in range(W, T_total + 1, 2)]

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

        checkpoint = (
            f"data/features/cefi_window_{W}_12_100_"
            f"{estimator_fingerprint()[:12]}.csv"
        )
        if W == 500:
            base = _load_canonical_series()
            df_w = base.rename(columns={"date": "Date"}).set_index("Date")[["cefi"]]
        elif os.path.exists(checkpoint):
            candidate = pd.read_csv(checkpoint, parse_dates=["Date"]).set_index("Date")
            if (
                set(candidate["estimator_spec"].dropna()) == {ESTIMATOR_SPEC}
                and set(candidate["estimator_sha256"].dropna()) == {estimator_fingerprint()}
                and candidate["cefi"].notna().all()
            ):
                df_w = candidate
                print(f"  Resuming from {checkpoint}")
            else:
                raise ValueError(f"Incompatible checkpoint: {checkpoint}")
        else:
            chunk = 512
            cefi_chunks = []
            for i in range(0, len(A_batch), chunk):
                c_s, _ = evaluate_batch_cefi_cuda(
                    A_batch[i:i+chunk], S_eps_batch[i:i+chunk], S_x_batch[i:i+chunk],
                    p, q_all, n_restarts=N_RESTARTS, max_iter=MAX_ITER, kappa=1.0
                )
                cefi_chunks.append(c_s)
            cefi_w = np.concatenate(cefi_chunks)
            df_w = pd.DataFrame({
                "cefi": cefi_w,
                "Date": task_dates,
                "estimator_spec": ESTIMATOR_SPEC,
                "estimator_sha256": estimator_fingerprint(),
            }).set_index("Date")
            df_w.to_csv(checkpoint)
        series_dict[W] = df_w
        print(f"  W = {W} done in {time.time()-t0:.1f}s | Mean CEFI = {df_w['cefi'].mean():.4f}")

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
            "Window_Length": f"W = {W} trading days" if W != 500 else "W = 500 trading days (Baseline)",
            "Mean_CEFI": float(w_df["cefi"].mean()),
            "Spearman_rho_vs_Baseline": float(s_corr),
            "Optimal_Phase_Lag_Steps": best_lag,
            "Optimal_Phase_Lag_Trading_Days": 2 * best_lag,
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
    df_cefi = _load_canonical_series().rename(columns={"date": "Date"}).set_index("Date")
    df_returns = pd.read_csv("data/raw/ff30_daily_returns.csv", parse_dates=["Date"], index_col="Date")
    W = 500
    er_list = []
    pca80_list = []
    pca90_list = []
    for date in df_cefi.index:
        if date not in df_returns.index:
            raise ValueError(f"CEFI date {date} is absent from the returns index")
        t_end = int(df_returns.index.get_loc(date)) + 1
        if t_end < W:
            raise ValueError(f"CEFI date {date} has fewer than {W} preceding observations")
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
    pd.DataFrame([
        {"Comparison": "q_star_vs_effective_rank", "Pearson_rho": p_er, "Spearman_rho": s_er},
        {"Comparison": "q_star_vs_pca80_dimension", "Pearson_rho": stats.pearsonr(q_star, pca80_arr)[0], "Spearman_rho": s_pca80},
        {"Comparison": "q_star_vs_pca90_dimension", "Pearson_rho": stats.pearsonr(q_star, pca90_arr)[0], "Spearman_rho": s_pca90},
    ]).to_csv("reports/tables/table_qstar_static_dimension_diagnostics.csv", index=False)


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
        target_date = pd.to_datetime(d_str)
        end_loc = df_returns.index.get_indexer([target_date], method="nearest")[0]
        actual_date = df_returns.index[end_loc]
        w = df_returns.iloc[end_loc - 500 + 1 : end_loc + 1].values
        if len(w) != 500:
            raise ValueError(f"Insufficient data for closure benchmark {d_str}")
        A_m, S_eps_m = fit_micro_var1(w)
        S_x_m = np.cov(w, rowvar=False)

        _, q_arr = evaluate_batch_cefi_cuda(
            A_m[None, ...], S_eps_m[None, ...], S_x_m[None, ...],
            p, range(1, p), n_restarts=N_RESTARTS, max_iter=MAX_ITER, kappa=1.0,
        )
        q_opt = int(q_arr[0])
        _, _, _, _, best_w = evaluate_batch_cefi(
            A_m[None, ...], S_eps_m[None, ...], S_x_m[None, ...], [q_opt],
            n_restarts=N_RESTARTS, max_iter=MAX_ITER, kappa=1.0,
            device=DEVICE,
            search_dtype=torch.float64,
            return_best_w=True,
        )
        best_W = best_w[0]

        WA = best_W @ A_m
        WA_proj = WA @ best_W.T @ best_W
        r_closure = float(np.linalg.norm(WA - WA_proj, ord="fro") / np.linalg.norm(WA, ord="fro"))

        closure_records.append({
            "Regime": label,
            "Benchmark_End_Date": actual_date.strftime("%Y-%m-%d"),
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
    out_path = "data/features/cefi_ff49_daily_series.csv"
    if os.path.exists(out_path):
        df_saved = pd.read_csv(out_path)
        if len(df_saved) >= 800 and set(df_saved["estimator_spec"].dropna()) == {ESTIMATOR_SPEC}:
            print(f"Verified existing FF49 series ({len(df_saved)} windows); skipping recomputation.")
            cefi_ff49 = df_saved["cefi"].to_numpy()
            q_ff49 = df_saved["q_star"].to_numpy()
            mean_c = float(cefi_ff49.mean())
            median_c = float(np.median(cefi_ff49))
            modal_q = int(pd.Series(q_ff49).mode()[0])
            pct_le4 = float((q_ff49 <= 4).mean() * 100.0)
            print(f"FF49 Replicated from saved series:")
            print(f"  Mean CEFI: {mean_c:.4f}, Median CEFI: {median_c:.4f}, Modal q*: {modal_q}, P(q* <= 4): {pct_le4:.2f}%")
            return

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

    df_out = pd.DataFrame({
        "cefi": cefi_ff49,
        "q_star": q_ff49,
        "optimizer_budget": "12/100",
        "estimator_spec": ESTIMATOR_SPEC,
        "estimator_sha256": estimator_fingerprint(),
        "Date": task_dates,
    }).set_index("Date")
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
