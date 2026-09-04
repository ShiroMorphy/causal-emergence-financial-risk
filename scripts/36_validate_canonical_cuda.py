#!/usr/bin/env python3
"""Preflight validation for the canonical isotropic CUDA estimator."""

import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from causal_emergence.cuda_stiefel import ESTIMATOR_SPEC, estimator_fingerprint, evaluate_batch_cefi
from causal_emergence.micro_var import fit_micro_var1


def main():
    returns = pd.read_csv("data/raw/ff30_daily_returns.csv", parse_dates=["Date"], index_col="Date")
    dates = ["2005-12-30", "2008-11-20", "2020-03-23"]
    systems = []
    for date in dates:
        window = returns.loc[:date].iloc[-500:].to_numpy()
        A, S_eps = fit_micro_var1(window)
        systems.append((A, S_eps, np.cov(window, rowvar=False)))
    A, S_eps, S_x = (np.stack(values) for values in zip(*systems))

    started = time.time()
    cpu = evaluate_batch_cefi(
        A, S_eps, S_x, range(1, 30), n_restarts=12, max_iter=100,
        device=torch.device("cpu"), search_dtype=torch.float64,
    )
    target = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    together = evaluate_batch_cefi(
        A, S_eps, S_x, range(1, 30), n_restarts=12, max_iter=100,
        device=target, search_dtype=torch.float64,
    )
    elapsed = time.time() - started
    alone = evaluate_batch_cefi(
        A[:1], S_eps[:1], S_x[:1], range(1, 30), n_restarts=12, max_iter=100,
        device=target, search_dtype=torch.float64,
    )

    np.testing.assert_allclose(together[0][0], alone[0][0], rtol=0, atol=1e-8)
    np.testing.assert_equal(together[1][0], alone[1][0])
    np.testing.assert_allclose(together[0], cpu[0], rtol=1e-8, atol=1e-10)
    np.testing.assert_array_equal(together[1], cpu[1])
    print(f"Estimator: {ESTIMATOR_SPEC}", flush=True)
    print(f"Estimator SHA-256: {estimator_fingerprint()}", flush=True)
    print(f"Device preflight ({target}) completed in {elapsed:.2f} seconds", flush=True)
    for date, cefi, q_star in zip(dates, together[0], together[1]):
        print(f"{date}: CEFI={cefi:.8f} nats, q*={q_star}", flush=True)
    print("PASS: CPU/device parity, batch-partition invariant, isotropic, nats", flush=True)


if __name__ == "__main__":
    main()
