#!/usr/bin/env bash
set -u

ROOT=.
PYTHON=python3
MANIFEST="$ROOT/preprocess/phoenix2014/split_manifest.json"
SPLIT_PID="${1:-812933}"
STATUS_DIR="$ROOT/work_dir/fd_xmp_v3/h1_protocol"
STATUS_FILE="$STATUS_DIR/status.txt"
RUN_DIR="$ROOT/work_dir/fd_xmp_v3/E2_fdxmp_h1_split_seed0"

mkdir -p "$STATUS_DIR"
exec >>"$STATUS_DIR/protocol.log" 2>&1

log() {
    printf '[%s] %s\n' "$(date '+%F %T %Z')" "$1" | tee "$STATUS_FILE"
}

wait_for_pid() {
    local pid="$1"
    while kill -0 "$pid" 2>/dev/null; do
        sleep 60
    done
}

best_checkpoint() {
    find "$1" -maxdepth 1 -name 'dev_*_epoch*_model.pt' -printf '%f\n' |
        sort -t_ -k2,2n |
        head -1
}

log "WAIT split protocol pid=$SPLIT_PID"
wait_for_pid "$SPLIT_PID"
if ! grep -q 'PROTOCOL_COMPLETE' \
    "$ROOT/work_dir/fd_xmp_v3/split_protocol/status.txt"; then
    log "FAILED split protocol did not complete"
    exit 10
fi

cd "$ROOT" || exit 11
log "START training E2_fdxmp_h1_split_seed0"
"$PYTHON" -u main.py \
    --device 0 \
    --work-dir "$RUN_DIR/" \
    --config ./configs/fdxmp_h1_split.yaml \
    --num-epoch 40 \
    --save-interval 5 \
    --evaluate-tool sclite \
    >"$RUN_DIR.console.log" 2>&1
EXIT_CODE=$?
if [[ $EXIT_CODE -ne 0 ]]; then
    log "FAILED H1 training exit_code=$EXIT_CODE"
    exit "$EXIT_CODE"
fi
COMPLETED_EPOCHS=$(grep -c 'Mean training loss:' "$RUN_DIR/log.txt" || true)
if [[ "$COMPLETED_EPOCHS" -lt 40 ]]; then
    log "FAILED H1 training incomplete_log"
    exit 12
fi
log "COMPLETE H1 training"

BEST_CHECKPOINT="$RUN_DIR/$(best_checkpoint "$RUN_DIR")"
if [[ ! -f "$BEST_CHECKPOINT" ]]; then
    log "FAILED H1 checkpoint not found"
    exit 13
fi
log "START H1 prototype coverage export"
"$PYTHON" scripts/export_prototype_coverage.py \
    --checkpoint "$BEST_CHECKPOINT" \
    --output "$RUN_DIR/prototype_coverage.json" \
    >>"$STATUS_DIR/prototype_coverage.console.log" 2>&1
COVERAGE_EXIT=$?
if [[ $COVERAGE_EXIT -ne 0 ]]; then
    log "FAILED H1 prototype coverage exit_code=$COVERAGE_EXIT"
    exit "$COVERAGE_EXIT"
fi
log "COMPLETE H1 prototype coverage export"

SELECTION_DIR="$RUN_DIR/selection"
log "START H1 Dev selection"
"$PYTHON" -u main.py \
    --device 0 \
    --phase test \
    --work-dir "$SELECTION_DIR/" \
    --config ./configs/fdxmp_h1_split.yaml \
    --load-weights "$BEST_CHECKPOINT" \
    --eval-splits dev \
    --evaluate-tool sclite \
    >"$SELECTION_DIR.console.log" 2>&1
EXIT_CODE=$?
if [[ $EXIT_CODE -ne 0 ]]; then
    log "FAILED H1 Dev selection exit_code=$EXIT_CODE"
    exit "$EXIT_CODE"
fi
log "COMPLETE H1 Dev selection"

FDGC_DIR="$RUN_DIR/fdgc"
log "START H1 FDGC calibration"
"$PYTHON" -u scripts/train_fdgc.py \
    --device 0 \
    --config ./configs/fdxmp_h1_split.yaml \
    --load-weights "$BEST_CHECKPOINT" \
    --split-manifest "$MANIFEST" \
    --difficulty-report "$SELECTION_DIR/frequency-metrics-dev.json" \
    --work-dir "$FDGC_DIR/" \
    --fdgc-epochs 10 \
    --fdgc-lr 0.001 \
    >"$FDGC_DIR.console.log" 2>&1
EXIT_CODE=$?
if [[ $EXIT_CODE -ne 0 ]]; then
    log "FAILED H1 FDGC calibration exit_code=$EXIT_CODE"
    exit "$EXIT_CODE"
fi
log "COMPLETE H1 FDGC calibration"

CALIBRATED_CHECKPOINT="$FDGC_DIR/checkpoint_calibrated.pt"
"$PYTHON" scripts/merge_fdgc_checkpoint.py \
    --base-checkpoint "$BEST_CHECKPOINT" \
    --fdgc-checkpoint "$FDGC_DIR/checkpoint_fdgc.pt" \
    --split-manifest "$MANIFEST" \
    --output "$CALIBRATED_CHECKPOINT"
EXIT_CODE=$?
if [[ $EXIT_CODE -ne 0 ]]; then
    log "FAILED H1 checkpoint merge exit_code=$EXIT_CODE"
    exit "$EXIT_CODE"
fi

FINAL_DIR="$RUN_DIR/final"
log "START H1 final Dev evaluation"
"$PYTHON" -u main.py \
    --device 0 \
    --phase test \
    --work-dir "$FINAL_DIR/" \
    --config ./configs/fdxmp_h1_split.yaml \
    --load-weights "$CALIBRATED_CHECKPOINT" \
    --eval-splits dev \
    --evaluate-tool sclite \
    >"$FINAL_DIR.console.log" 2>&1
EXIT_CODE=$?
if [[ $EXIT_CODE -ne 0 ]]; then
    log "FAILED H1 final Dev evaluation exit_code=$EXIT_CODE"
    exit "$EXIT_CODE"
fi
log "COMPLETE H1 final Dev evaluation"

log "H1_PROTOCOL_COMPLETE"
