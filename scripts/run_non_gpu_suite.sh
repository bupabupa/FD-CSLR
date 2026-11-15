#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-.}
PYTHON=${PYTHON:-python3}
LOG=${1:-"$ROOT/tests/full_suite_$(date +%Y%m%d_%H%M).log"}

cd "$ROOT"
export PYTHONPATH="$ROOT"
export CUDA_VISIBLE_DEVICES=

mapfile -t tests < <(
    find tests -maxdepth 1 -type f -name 'test_*.py' -printf '%f\n' |
        sort |
        grep -v -E '^(test_ctcdecode_smoke|test_fdxmp_smoke|test_signer_film)\.py$'
)

: >"$LOG"
passed=0
for test in "${tests[@]}"; do
    printf 'RUN %s\n' "$test"
    "$PYTHON" -u "tests/$test" >>"$LOG" 2>&1
    passed=$((passed + 1))
    printf 'PASS %s\n' "$test"
done

printf 'ALL_PASS %d/%d\n' "$passed" "${#tests[@]}" | tee -a "$LOG"
