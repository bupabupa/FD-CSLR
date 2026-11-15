#!/usr/bin/env bash
set -u

ROOT=.
PYTHON=python3
MANIFEST="$ROOT/preprocess/phoenix2014/split_manifest.json"
STATUS_DIR="$ROOT/work_dir/fd_xmp_v3/signer_diagnostics"
STATUS_FILE="$STATUS_DIR/status.txt"
H1_STATUS="$ROOT/work_dir/fd_xmp_v3/h1_protocol/status.txt"
SPLIT_STATUS="$ROOT/work_dir/fd_xmp_v3/split_protocol/status.txt"

mkdir -p "$STATUS_DIR"
exec >>"$STATUS_DIR/protocol.log" 2>&1

log() {
    printf '[%s] %s\n' "$(date '+%F %T %Z')" "$1" | tee "$STATUS_FILE"
}

wait_for_marker() {
    local path="$1"
    local marker="$2"
    while ! grep -q "$marker" "$path" 2>/dev/null; do
        sleep 60
    done
}

best_checkpoint() {
    find "$1" -maxdepth 1 -name 'dev_*_epoch*_model.pt' -printf '%f\n' |
        sort -t_ -k2,2n |
        head -1
}

run_diagnostic() {
    local label="$1"
    local config="$2"
    local run_dir="$3"
    local output="$4"
    local checkpoint
    checkpoint="$run_dir/$(best_checkpoint "$run_dir")"
    if [[ ! -f "$checkpoint" ]]; then
        log "FAILED $label checkpoint not found"
        return 10
    fi
    if [[ -f "$output" ]]; then
        log "SKIP existing $label diagnostic"
        return 0
    fi
    log "START $label signer diagnostic"
    cd "$ROOT" || return 11
    "$PYTHON" scripts/evaluate_signer_invariance.py \
        --config "$config" \
        --load-weights "$checkpoint" \
        --split-manifest "$MANIFEST" \
        --output "$output" \
        --device 0 \
        --batch-size 2 \
        --num-workers 4
    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        log "FAILED $label signer diagnostic exit_code=$exit_code"
        return "$exit_code"
    fi
    log "COMPLETE $label signer diagnostic"
}

log "WAIT H1 protocol completion"
wait_for_marker "$H1_STATUS" "H1_PROTOCOL_COMPLETE"
if ! grep -q "PROTOCOL_COMPLETE" "$SPLIT_STATUS"; then
    log "FAILED split protocol is not complete"
    exit 20
fi

BASELINE_DIR="$ROOT/work_dir/fd_xmp_v3/E0_baseline_split_seed0"
H1_DIR="$ROOT/work_dir/fd_xmp_v3/E2_fdxmp_h1_split_seed0"
BASELINE_OUTPUT="$STATUS_DIR/baseline_signer_invariance.json"
H1_OUTPUT="$STATUS_DIR/h1_signer_invariance.json"
COMPARISON_OUTPUT="$STATUS_DIR/comparison.json"

run_diagnostic \
    "strict baseline" \
    "./configs/baseline_split.yaml" \
    "$BASELINE_DIR" \
    "$BASELINE_OUTPUT" || exit $?
run_diagnostic \
    "H1" \
    "./configs/fdxmp_h1_split.yaml" \
    "$H1_DIR" \
    "$H1_OUTPUT" || exit $?

"$PYTHON" scripts/compare_signer_diagnostics.py \
    --baseline "$BASELINE_OUTPUT" \
    --h1 "$H1_OUTPUT" \
    --output "$COMPARISON_OUTPUT"
exit_code=$?
if [[ $exit_code -ne 0 ]]; then
    log "FAILED signer diagnostic comparison exit_code=$exit_code"
    exit "$exit_code"
fi

log "SIGNER_DIAGNOSTICS_COMPLETE"
