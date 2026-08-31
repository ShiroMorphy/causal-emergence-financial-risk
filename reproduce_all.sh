#!/usr/bin/env bash
# ==============================================================================
# Master Pipeline Reproduction Script for:
# "Causal Emergence in Financial Markets: Dynamic Organization and Effective
#  Dimensionality During Systemic Stress"
# Target Journal: International Review of Financial Analysis (IRFA)
# ==============================================================================

set -e

echo "=== [1/5] Running Complete Test Suite ==="
PYTHONPATH=src python3 -m pytest tests/ -v

echo "=== [2/5] Running Master Pre-Submission Diagnostics Suite ==="
PYTHONPATH=src python3 scripts/17_run_master_closure_diagnostics.py

echo "=== [3/5] Generating Publication Figures & Tables ==="
PYTHONPATH=src python3 scripts/14_generate_manuscript_figures.py
PYTHONPATH=src python3 scripts/15_financial_benchmarks_h4.py

echo "=== [4/5] Compiling LaTeX Documents (Manuscript, Title Page, Appendix, Cover Letter) ==="
pdflatex -interaction=nonstopmode manuscript.tex > /dev/null 2>&1
pdflatex -interaction=nonstopmode Title_Page.tex > /dev/null 2>&1
pdflatex -interaction=nonstopmode Supplementary_Appendix.tex > /dev/null 2>&1
pdflatex -interaction=nonstopmode Cover_Letter.tex > /dev/null 2>&1

echo "========================================================================"
echo "Reproduction Complete! All outputs generated in reports/ & PDFs compiled."
echo "========================================================================"
