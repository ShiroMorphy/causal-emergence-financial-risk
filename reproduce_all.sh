#!/usr/bin/env bash
# Canonical end-to-end reproduction (isotropic constructed-macro estimator).

set -euo pipefail

echo "=== [1/6] Running tests and deterministic estimator preflight ==="
PYTHONPATH=src python3 -m pytest tests/ -q
PYTHONPATH=src python3 scripts/36_validate_canonical_cuda.py

echo "=== [2/6] Computing canonical FF30 series, matched nulls, tables, and benchmarks ==="
PYTHONPATH=src python3 scripts/33_run_canonical_cuda_production.py --phase finalize

echo "=== [3/6] Computing canonical robustness and FF49 replication ==="
PYTHONPATH=src python3 scripts/34_rerun_all_robustness_12_100.py

echo "=== [4/6] Regenerating publication figures from provenance-checked results ==="
PYTHONPATH=src python3 scripts/35_regenerate_all_publication_figures.py

echo "=== [5/6] Verifying the complete numerical package ==="
PYTHONPATH=src python3 scripts/37_verify_canonical_outputs.py

echo "=== [6/7] Compiling submission documents ==="
for document in manuscript Title_Page Supplementary_Appendix Cover_Letter; do
    pdflatex -halt-on-error -interaction=nonstopmode "${document}.tex" >/dev/null
    pdflatex -halt-on-error -interaction=nonstopmode "${document}.tex" >/dev/null
done

echo "=== [7/7] Running exhaustive PDF-level pre-submission audit ==="
python3 scripts/29_exhaustive_consistency_audit.py

echo "Canonical reproduction complete."
