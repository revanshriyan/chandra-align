#!/usr/bin/env bash
# CPU-only end-to-end reproduction from a clean clone.
#
# Steps (fail loudly, non-zero exit, at the first failed step — set -e):
#   1. create a virtual environment and install requirements-cpu.txt
#   2. ChandraBench evaluator self-test (chandrabench/eval.py --self-test)
#   3. fast deterministic test subset (~7 s of pytest; chosen and documented
#      below)
#   4. regenerate docs/RESULTS_ONE_PAGER.md and verify it has not drifted
#      from the committed file (git diff --exit-code)
#
# Test subset choice (deterministic, minutes not the ~10-minute full suite):
#   - tests/test_canonical_table.py       (0.2 s) canonical evidence table
#                                          regenerates from its sources and
#                                          fails on drift
#   - tests/test_results_one_pager_drift.py (0.3 s) the one-pager produced in
#                                          step 4 is the committed bytes
#   - tests/test_crs_guard.py             (3.0 s) projection-trap guard
#   - tests/test_mutual_nn.py             (0.1 s) matcher correspondence
#                                          regression
#   - tests/test_no_ai_watermark.py       (0.6 s) hygiene
#   - tests/test_output.py                (2.5 s) export contract
#   Total pytest time ~7 s on the reference machine.
#
# Usage:  bash reproduce.sh          (from the repository root)
# The script is POSIX sh/bash and works under Git Bash on Windows (the venv
# python lives in Scripts/, not bin/ — detected below).

set -euo pipefail
cd "$(dirname "$0")"

PY=""
if command -v python >/dev/null 2>&1; then PY=python; fi
if [ -z "$PY" ] && command -v python3 >/dev/null 2>&1; then PY=python3; fi
if [ -z "$PY" ]; then
    echo "FAIL: no python on PATH" >&2
    exit 1
fi

echo "== [1/4] environment from requirements-cpu.txt =="
if [ ! -x ".venv-repro/Scripts/python.exe" ] && [ ! -x ".venv-repro/bin/python" ]; then
    "$PY" -m venv .venv-repro
fi
if [ -x ".venv-repro/Scripts/python.exe" ]; then
    VPY=".venv-repro/Scripts/python.exe"
else
    VPY=".venv-repro/bin/python"
fi
"$VPY" -m pip install --quiet --upgrade pip
"$VPY" -m pip install --quiet -r requirements-cpu.txt
"$VPY" -c "import numpy, cv2, torch; print('env ok:', torch.__version__)"

echo "== [2/4] ChandraBench v0.1 evaluator self-test =="
"$VPY" chandrabench/eval.py --self-test

echo "== [3/4] fast deterministic test subset =="
# pytest is not in requirements-cpu.txt (CI installs it separately); the
# test subset cannot run without it, so install it into the repro venv.
"$VPY" -m pip install --quiet pytest
"$VPY" -m pytest -q -p no:cacheprovider \
    tests/test_canonical_table.py \
    tests/test_results_one_pager_drift.py \
    tests/test_crs_guard.py \
    tests/test_mutual_nn.py \
    tests/test_no_ai_watermark.py \
    tests/test_output.py

echo "== [4/4] regenerate docs/RESULTS_ONE_PAGER.md =="
"$VPY" scripts/generate_results_one_pager.py
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git diff --exit-code -- docs/RESULTS_ONE_PAGER.md \
        || { echo "FAIL: docs/RESULTS_ONE_PAGER.md drifted from the committed file" >&2; exit 1; }
    echo "one-pager matches the committed file"
else
    echo "not a git worktree: drift check skipped (regeneration succeeded)"
fi

echo "REPRODUCTION OK"
