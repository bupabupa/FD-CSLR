#!/usr/bin/env bash
set -u

ROOT=.
QUEUE_DIR="$ROOT/work_dir/fd_xmp_v3/queue"
PYTHON=python3
BASELINE_PID=698258
BASELINE_LOG="$ROOT/work_dir/fd_xmp_v3/E0_baseline_seqctc_seed0/log.txt"
STATUS_FILE="$QUEUE_DIR/status.txt"

mkdir -p "$QUEUE_DIR"
exec >>"$QUEUE_DIR/queue.log" 2>&1

log() {
    printf '[%s] %s\n' "$(date '+%F %T %Z')" "$1" | tee "$STATUS_FILE"
}

wait_for_pid() {
    local pid="$1"
    while kill -0 "$pid" 2>/dev/null; do
        sleep 60
    done
}

training_completed() {
    local log_path="$1"
    grep -q 'Epoch: 39, Batch(2800/2835)' "$log_path"
}

run_job() {
    local name="$1"
    local config="$2"
    local log_path="$3"
    local console_path="$4"

    log "START $name"
    "$PYTHON" -u main.py \
        --device 0 \
        --work-dir "$(dirname "$log_path")/" \
        --config "$config" \
        --num-epoch 40 \
        --save-interval 5 \
        --evaluate-tool sclite \
        >"$console_path" 2>&1
    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        log "FAILED $name exit_code=$exit_code"
        exit "$exit_code"
    fi
    if ! training_completed "$log_path"; then
        log "FAILED $name training_log_incomplete"
        exit 2
    fi
    log "COMPLETE $name"
}

cd "$ROOT" || exit 3
log "WAIT baseline pid=$BASELINE_PID"
wait_for_pid "$BASELINE_PID"
if ! training_completed "$BASELINE_LOG"; then
    log "FAILED baseline training_log_incomplete"
    exit 4
fi
log "COMPLETE E0_baseline_seqctc_seed0"

run_job \
    "E0_vac_ctc_seed0" \
    "./configs/vac_ctc.yaml" \
    "$ROOT/work_dir/fd_xmp_v3/E0_vac_ctc_seed0/log.txt" \
    "$QUEUE_DIR/E0_vac_ctc_seed0.console.log"

run_job \
    "E1_proto_seed0" \
    "./configs/fdxmp_proto.yaml" \
    "$ROOT/work_dir/fd_xmp_v3/E1_proto_seed0/log.txt" \
    "$QUEUE_DIR/E1_proto_seed0.console.log"

log "QUEUE_COMPLETE"
