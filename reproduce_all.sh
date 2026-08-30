#!/usr/bin/env bash
# ==============================================================================
# Master Pipeline Reproduction Script for:
# "Causal Emergence in Financial Markets: Dynamic Organization and Effective
#  Dimensionality During Systemic Stress"
# ==============================================================================

set -e

echo "=== [1/5] Running Unit Tests ==="
PYTHONPATH=src pytest tests/ -v

echo "=== [2/5] Generating Main Rolling Time Series (1992-2026) ==="
PYTHONPATH=src /opt/anaconda3/bin/python scripts/06_run_rolling_analysis.py --step 2

echo "=== [3/5] Running Matched Null Models Statistical Inference (B=999) ==="
PYTHONPATH=src /opt/anaconda3/bin/python scripts/07_run_null_inference.py --B 999

echo "=== [4/5] Running External Cross-Method Validation (Liu 2024, PRE 2025) ==="
PYTHONPATH=src /opt/anaconda3/bin/python scripts/09_framework_robustness.py --step 10

echo "=== [5/5] Generating Publication Vector Figures & Tables ==="
PYTHONPATH=src /opt/anaconda3/bin/python scripts/14_generate_manuscript_figures.py
PYTHONPATH=src /opt/anaconda3/bin/python scripts/15_financial_benchmarks_h4.py

echo "========================================================================"
echo "Reproduction Complete! All results generated in reports/figures/ & reports/tables/"
echo "========================================================================"
