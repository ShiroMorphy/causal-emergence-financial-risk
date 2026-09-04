#!/usr/bin/env python3
"""
scripts/38_run_parallel_cuda_production.py

Master Parallel Orchestrator for Canonical 12/100 Production Run on RTX 5090.
Launches concurrent processes for Phase 2 (rolling series) and Phase 3 (3 benchmark regimes),
then finalizes downstream econometrics, robustness, and verification.
"""

import os
import sys
import time
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable

def main():
    os.chdir(ROOT)
    os.makedirs("reports/logs", exist_ok=True)
    os.makedirs("reports/checkpoints", exist_ok=True)

    print("=" * 90)
    print("MASTER PARALLEL CUDA ORCHESTRATOR (12 RESTARTS / 100 ITERATIONS)")
    print(f"Working directory: {ROOT}")
    print(f"Python interpreter: {PYTHON}")
    print("=" * 90)
    t_start = time.time()

    workers = [
        {
            "name": "Phase 2 (Rolling Series FF30, 4,346 windows)",
            "cmd": [PYTHON, "-u", "scripts/33_run_canonical_cuda_production.py", "--phase", "2"],
            "log": "reports/logs/worker_phase2_rolling_series.log"
        },
        {
            "name": "Phase 3 Regime 0: Calm Period (2005-12-30)",
            "cmd": [PYTHON, "-u", "scripts/33_run_canonical_cuda_production.py", "--phase", "3", "--regime", "0"],
            "log": "reports/logs/worker_phase3_regime0_calm.log"
        },
        {
            "name": "Phase 3 Regime 1: 2008 GFC Peak (2008-11-20)",
            "cmd": [PYTHON, "-u", "scripts/33_run_canonical_cuda_production.py", "--phase", "3", "--regime", "1"],
            "log": "reports/logs/worker_phase3_regime1_gfc.log"
        },
        {
            "name": "Phase 3 Regime 2: 2020 COVID Shock (2020-03-23)",
            "cmd": [PYTHON, "-u", "scripts/33_run_canonical_cuda_production.py", "--phase", "3", "--regime", "2"],
            "log": "reports/logs/worker_phase3_regime2_covid.log"
        },
    ]

    running = []
    file_handles = []

    for w in workers:
        print(f"Launching [{w['name']}] -> {w['log']}")
        fh = open(w["log"], "w", buffering=1)
        file_handles.append(fh)
        proc = subprocess.Popen(
            w["cmd"],
            stdout=fh,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONPATH": "src", "PYTHONUNBUFFERED": "1"}
        )
        running.append((w["name"], proc, fh, w["log"]))

    print(f"\nAll {len(running)} workers launched concurrently in the background!")
    print("Monitoring progress (press Ctrl+C to stop all workers)...\n")

    try:
        active = list(running)
        while active:
            time.sleep(15)
            still_active = []
            for name, proc, fh, log_path in active:
                ret = proc.poll()
                if ret is None:
                    still_active.append((name, proc, fh, log_path))
                else:
                    elapsed = time.time() - t_start
                    if ret == 0:
                        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] SUCCESS: [{name}] finished in {elapsed:.1f}s ({elapsed/60:.1f} min)")
                    else:
                        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ERROR: [{name}] failed with returncode {ret}! Check {log_path}")
                        raise RuntimeError(f"Worker {name} failed with exit code {ret}")
            active = still_active

    except KeyboardInterrupt:
        print("\nInterrupt received. Terminating all worker processes...")
        for name, proc, fh, _ in running:
            if proc.poll() is None:
                proc.terminate()
        time.sleep(2)
        for name, proc, fh, _ in running:
            if proc.poll() is None:
                proc.kill()
        sys.exit(1)
    finally:
        for fh in file_handles:
            fh.close()

    print("\n" + "=" * 90)
    print("ALL 4 PARALLEL WORKERS COMPLETED SUCCESSFULLY!")
    print("Executing Finalization: Holm-Bonferroni Combination, Econometrics, and Reports...")
    print("=" * 90)

    fin_cmd = [PYTHON, "scripts/33_run_canonical_cuda_production.py", "--phase", "finalize"]
    subprocess.run(fin_cmd, check=True, env={**os.environ, "PYTHONPATH": "src"})

    print("\nExecuting Robustness Suite (Script 34)...")
    rob_cmd = [PYTHON, "scripts/34_rerun_all_robustness_12_100.py"]
    subprocess.run(rob_cmd, check=True, env={**os.environ, "PYTHONPATH": "src"})

    print("\nRegenerating All Publication Figures (Script 35)...")
    fig_cmd = [PYTHON, "scripts/35_regenerate_all_publication_figures.py"]
    subprocess.run(fig_cmd, check=True, env={**os.environ, "PYTHONPATH": "src"})

    print("\nVerifying Canonical Outputs (Script 37)...")
    ver_cmd = [PYTHON, "scripts/37_verify_canonical_outputs.py"]
    subprocess.run(ver_cmd, check=True, env={**os.environ, "PYTHONPATH": "src"})

    total_time = time.time() - t_start
    print("\n" + "=" * 90)
    print(f"MASTER PARALLEL PIPELINE COMPLETED IN {total_time:.2f}s ({total_time/3600:.2f} hours)!")
    print("=" * 90)

if __name__ == "__main__":
    main()
