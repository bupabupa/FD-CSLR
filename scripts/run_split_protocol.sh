#!/usr/bin/env bash
set -u

ROOT=.
PYTHON=python3
MANIFEST="$ROOT/preprocess/phoenix2014/split_manifest.json"
QUEUE_PID="${1:-771744}"
STATUS_DIR="$ROOT/work_dir/fd_xmp_v3/split_protocol"
STATUS_FILE="$STATUS_DIR/status.txt"

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

training_completed() {
    local log_path="$1"
    local completed_epochs
    completed_epochs=$(grep -c 'Mean training loss:' "$log_path" || true)
    [[ "$completed_epochs" -ge 40 ]]
}

best_checkpoint() {
    find "$1" -maxdepth 1 -name 'dev_*_epoch*_model.pt' -printf '%f\n' |
        sort -t_ -k2,2n |
        head -1
}

run_training() {
    local name="$1"
    local config="$2"
    local run_dir="$3"

    log "START training $name"
    cd "$ROOT" || return 10
    "$PYTHON" -u main.py \
        --device 0 \
        --work-dir "$run_dir/" \
        --config "$config" \
        --num-epoch 40 \
        --save-interval 5 \
        --evaluate-tool sclite \
        >"$run_dir.console.log" 2>&1
    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        log "FAILED training $name exit_code=$exit_code"
        return "$exit_code"
    fi
    if ! training_completed "$run_dir/log.txt"; then
        log "FAILED training $name incomplete_log"
        return 11
    fi
    log "COMPLETE training $name"
}

final_evaluation() {
    local name="$1"
    local config="$2"
    local checkpoint="$3"
    local output_dir="$4"
    local splits="$5"

    log "START evaluation $name"
    cd "$ROOT" || return 10
    "$PYTHON" -u main.py \
        --device 0 \
        --phase test \
        --work-dir "$output_dir/" \
        --config "$config" \
        --load-weights "$checkpoint" \
        --eval-splits $splits \
        --evaluate-tool sclite \
        >"$output_dir.console.log" 2>&1
    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        log "FAILED evaluation $name exit_code=$exit_code"
        return "$exit_code"
    fi
    log "COMPLETE evaluation $name"
}

log "WAIT current queue pid=$QUEUE_PID"
wait_for_pid "$QUEUE_PID"
if ! grep -q 'QUEUE_COMPLETE' "$ROOT/work_dir/fd_xmp_v3/queue/status.txt"; then
    log "FAILED current queue did not complete"
    exit 20
fi

BASELINE_DIR="$ROOT/work_dir/fd_xmp_v3/E0_baseline_split_seed0"
run_training \
    "E0_baseline_split_seed0" \
    "./configs/baseline_split.yaml" \
    "$BASELINE_DIR" || exit $?
BASELINE_CHECKPOINT="$BASELINE_DIR/$(best_checkpoint "$BASELINE_DIR")"
final_evaluation \
    "E0_baseline_split_seed0" \
    "./configs/baseline_split.yaml" \
    "$BASELINE_CHECKPOINT" \
    "$BASELINE_DIR/final" \
    "dev" || exit $?

VAC_DIR="$ROOT/work_dir/fd_xmp_v3/E0_vac_ctc_split_seed0"
run_training \
    "E0_vac_ctc_split_seed0" \
    "./configs/vac_ctc_split.yaml" \
    "$VAC_DIR" || exit $?
VAC_CHECKPOINT="$VAC_DIR/$(best_checkpoint "$VAC_DIR")"
final_evaluation \
    "E0_vac_ctc_split_seed0" \
    "./configs/vac_ctc_split.yaml" \
    "$VAC_CHECKPOINT" \
    "$VAC_DIR/final" \
    "dev" || exit $?

FDXMP_DIR="$ROOT/work_dir/fd_xmp_v3/E1_fdxmp_v3_split_seed0"
run_training \
    "E1_fdxmp_v3_split_seed0" \
    "./configs/fdxmp_v3_split.yaml" \
    "$FDXMP_DIR" || exit $?
FDXMP_CHECKPOINT="$FDXMP_DIR/$(best_checkpoint "$FDXMP_DIR")"
log "START cosine prototype coverage export"
"$PYTHON" scripts/export_prototype_coverage.py \
    --checkpoint "$FDXMP_CHECKPOINT" \
    --output "$FDXMP_DIR/prototype_coverage.json" \
    >>"$STATUS_DIR/prototype_coverage.console.log" 2>&1
COVERAGE_EXIT=$?
if [[ $COVERAGE_EXIT -ne 0 ]]; then
    log "FAILED cosine prototype coverage exit_code=$COVERAGE_EXIT"
    exit "$COVERAGE_EXIT"
fi
log "COMPLETE cosine prototype coverage export"

SELECTION_DIR="$FDXMP_DIR/selection"
final_evaluation \
    "E1_fdxmp_v3_split_seed0_dev_selection" \
    "./configs/fdxmp_v3_split.yaml" \
    "$FDXMP_CHECKPOINT" \
    "$SELECTION_DIR" \
    "dev" || exit $?

FDGC_DIR="$FDXMP_DIR/fdgc"
log "START FDGC calibration"
cd "$ROOT" || exit 21
"$PYTHON" -u scripts/train_fdgc.py \
    --device 0 \
    --config ./configs/fdxmp_v3_split.yaml \
    --load-weights "$FDXMP_CHECKPOINT" \
    --split-manifest "$MANIFEST" \
    --difficulty-report "$SELECTION_DIR/frequency-metrics-dev.json" \
    --work-dir "$FDGC_DIR/" \
    --fdgc-epochs 10 \
    --fdgc-lr 0.001 \
    >"$FDGC_DIR.console.log" 2>&1
FDGC_EXIT=$?
if [[ $FDGC_EXIT -ne 0 ]]; then
    log "FAILED FDGC calibration exit_code=$FDGC_EXIT"
    exit "$FDGC_EXIT"
fi
log "COMPLETE FDGC calibration"

CALIBRATED_CHECKPOINT="$FDGC_DIR/checkpoint_calibrated.pt"
"$PYTHON" scripts/merge_fdgc_checkpoint.py \
    --base-checkpoint "$FDXMP_CHECKPOINT" \
    --fdgc-checkpoint "$FDGC_DIR/checkpoint_fdgc.pt" \
    --split-manifest "$MANIFEST" \
    --output "$CALIBRATED_CHECKPOINT"
MERGE_EXIT=$?
if [[ $MERGE_EXIT -ne 0 ]]; then
    log "FAILED FDGC checkpoint merge exit_code=$MERGE_EXIT"
    exit "$MERGE_EXIT"
fi

final_evaluation \
    "E1_fdxmp_v3_split_seed0_final" \
    "./configs/fdxmp_v3_split.yaml" \
    "$CALIBRATED_CHECKPOINT" \
    "$FDXMP_DIR/final" \
    "dev" || exit $?

log "PROTOCOL_COMPLETE"
