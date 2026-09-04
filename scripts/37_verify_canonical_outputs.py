#!/usr/bin/env python3
"""Fail-closed verification of every canonical production artifact."""

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from causal_emergence.cuda_stiefel import ESTIMATOR_SPEC, estimator_fingerprint


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def verify_series():
    path = ROOT / "data/features/cefi_series_12_100.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    require(len(df) == 4346, f"canonical CEFI has {len(df)} rows, expected 4346")
    require(not df["date"].duplicated().any(), "canonical CEFI dates are duplicated")
    require(df["date"].is_monotonic_increasing, "canonical CEFI dates are not sorted")
    require(np.isfinite(df[["cefi", "ei_micro", "macro_ei_max"]]).all().all(), "non-finite CEFI values")
    require(((df["q_star"] >= 1) & (df["q_star"] < 30)).all(), "q_star outside 1..29")
    require(set(df["estimator_spec"]) == {ESTIMATOR_SPEC}, "series estimator specification mismatch")
    require(set(df["estimator_sha256"]) == {estimator_fingerprint()}, "series estimator hash mismatch")
    require(set(df["micro_var_sha256"]) == {sha256(ROOT / "src/causal_emergence/micro_var.py")}, "series micro-VAR hash mismatch")
    require(set(df["data_sha256"]) == {sha256(ROOT / "data/raw/ff30_daily_returns.csv")}, "series data hash mismatch")

    returns = pd.read_csv(ROOT / "data/raw/ff30_daily_returns.csv", parse_dates=["Date"])
    expected_dates = returns["Date"].iloc[499::2].reset_index(drop=True)
    pd.testing.assert_series_equal(df["date"].reset_index(drop=True), expected_dates, check_names=False)
    return df


def verify_nulls():
    primary_path = ROOT / "reports/final_submission_source_of_truth/CANONICAL_NULL_RESULTS_12_100.csv"
    full_path = ROOT / "reports/tables/full_null_inference_summary.csv"
    manifest_path = ROOT / "reports/final_submission_source_of_truth/CANONICAL_NULL_RESULTS_12_100.provenance.json"
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    config = manifest["run_config"]
    require(config["estimator_spec"] == ESTIMATOR_SPEC, "null estimator specification mismatch")
    require(config["estimator_sha256"] == estimator_fingerprint(), "null estimator hash mismatch")
    require(config["B_primary"] == 9999 and config["B_aux"] == 999, "null replication budget mismatch")
    computed_run_fingerprint = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode("utf-8")
    ).hexdigest()
    require(manifest["run_fingerprint"] == computed_run_fingerprint, "null run fingerprint mismatch")
    for relative, expected_hash in manifest["outputs"].items():
        require(sha256(ROOT / relative) == expected_hash, f"output hash mismatch: {relative}")

    primary = pd.read_csv(primary_path)
    full = pd.read_csv(full_path)
    require(len(primary) == 3, "primary-null table must contain three regimes")
    require(len(full) == 12, "full-null table must contain twelve regime/model rows")
    require((full.loc[full["Null_Model"].isin(["H0_static", "H0_diag+contemp"]), "B"] == 9999).all(), "primary B mismatch")
    require((full.loc[full["Null_Model"].isin(["H0_circ", "H0_diag"]), "B"] == 999).all(), "auxiliary B mismatch")

    for _, row in primary.iterrows():
        prefix = "".join(ch if ch.isalnum() else "_" for ch in row["Regime"]).strip("_")
        for null_name, B in (("H0_static", 9999), ("H0_diag+contemp", 9999), ("H0_circ", 999), ("H0_diag", 999)):
            checkpoint = ROOT / "reports/checkpoints" / (
                f"{prefix}_{null_name}_B{B}_{ESTIMATOR_SPEC}_{manifest['run_fingerprint'][:12]}.npz"
            )
            require(checkpoint.exists(), f"missing checkpoint: {checkpoint.name}")
            with np.load(checkpoint) as saved:
                require(len(saved["cefi"]) == B and len(saved["q_star"]) == B, f"incomplete checkpoint: {checkpoint.name}")
    return primary


def verify_robustness(cefi):
    qdiag = pd.read_csv(ROOT / "reports/tables/table_qstar_static_dimension_diagnostics.csv")
    require(len(qdiag) == 3 and np.isfinite(qdiag.select_dtypes("number")).all().all(), "invalid q-star diagnostics")

    closure = pd.read_csv(ROOT / "reports/tables/table_closure_diagnostics.csv")
    require(len(closure) == 5, "closure table must contain five benchmarks")
    require("Benchmark_End_Date" in closure, "closure table lacks aligned benchmark dates")

    window = pd.read_csv(ROOT / "reports/tables/table_window_length_sensitivity.csv")
    require({"Optimal_Phase_Lag_Steps", "Optimal_Phase_Lag_Trading_Days"}.issubset(window.columns), "window lags lack explicit units")
    require((window["Optimal_Phase_Lag_Trading_Days"] == 2 * window["Optimal_Phase_Lag_Steps"]).all(), "window lag conversion mismatch")

    kappa = pd.read_csv(ROOT / "reports/tables/table_kappa_sensitivity.csv")
    require(set(np.round(kappa["kappa"], 2)) == {0.25, 0.5, 1.0, 2.0, 4.0}, "incomplete kappa sensitivity")

    ff49 = pd.read_csv(ROOT / "data/features/cefi_ff49_daily_series.csv")
    require(set(ff49["estimator_spec"]) == {ESTIMATOR_SPEC}, "FF49 estimator specification mismatch")
    require(set(ff49["estimator_sha256"]) == {estimator_fingerprint()}, "FF49 estimator hash mismatch")

    comparison = pd.read_csv(ROOT / "data/features/framework_comparison_series.csv")
    require(set(comparison["estimator_spec"]) == {ESTIMATOR_SPEC}, "framework comparison specification mismatch")
    require(len(comparison) == 870, "framework comparison must contain 870 windows")

    require(len(cefi) == 4346, "downstream verification received wrong CEFI series")


def verify_publication_package():
    for stem in (
        "figure1_cefi_dynamics",
        "figure2_qstar_dynamics",
        "figure3_null_distributions",
        "figure4_theoretical_benchmarking",
    ):
        for suffix in ("pdf", "png"):
            path = ROOT / f"reports/figures/{stem}.{suffix}"
            require(path.exists() and path.stat().st_size > 1000, f"missing/empty figure: {path.name}")

    forbidden = ("Bits / Dimension", "scripts/06_run_rolling_analysis.py")
    for relative in ("manuscript.tex", "Supplementary_Appendix.tex", "README.md"):
        text = (ROOT / relative).read_text(encoding="utf-8")
        for token in forbidden:
            require(token not in text, f"stale token {token!r} remains in {relative}")


def main():
    cefi = verify_series()
    verify_nulls()
    verify_robustness(cefi)
    verify_publication_package()
    print("PASS: canonical outputs are complete, aligned, provenance-matched, and publication-ready")


if __name__ == "__main__":
    main()
