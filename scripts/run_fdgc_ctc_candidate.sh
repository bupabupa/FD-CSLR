#!/usr/bin/env bash
set -u

ROOT=.
PYTHON=python3
MANIFEST="$ROOT/preprocess/phoenix2014/split_manifest.json"
BASE_RUN="${1:-$ROOT/work_dir/fd_xmp_v3/E1_fdxmp_v3_split_seed0}"
BASE_CHECKPOINT="${2:-}"
DIFFICULTY_REPORT="${3:-$BASE_RUN/selection/frequency-metrics-dev.json}"
OUTPUT_DIR="${4:-$BASE_RUN/fdgc_ctc}"
CONFIG="${5:-./configs/fdxmp_v3_split.yaml}"

if [[ -z "$BASE_CHECKPOINT" ]]; then
    BASE_CHECKPOINT=$(find "$BASE_RUN" -maxdepth 1 \
        -name 'dev_*_epoch*_model.pt' -printf '%f\n' |
        sort -t_ -k2,2n |
        head -1)
    BASE_CHECKPOINT="$BASE_RUN/$BASE_CHECKPOINT"
fi

if [[ ! -f "$BASE_CHECKPOINT" ]]; then
    echo "base checkpoint not found: $BASE_CHECKPOINT" >&2
    exit 2
fi
if [[ ! -f "$DIFFICULTY_REPORT" ]]; then
    echo "difficulty report not found: $DIFFICULTY_REPORT" >&2
    exit 3
fi

mkdir -p "$OUTPUT_DIR"
cd "$ROOT" || exit 4

"$PYTHON" -u scripts/train_fdgc.py \
    --device 0 \
    --config "$CONFIG" \
    --load-weights "$BASE_CHECKPOINT" \
    --split-manifest "$MANIFEST" \
    --difficulty-report "$DIFFICULTY_REPORT" \
    --work-dir "$OUTPUT_DIR/" \
    --fdgc-epochs 10 \
    --fdgc-lr 0.001 \
    --fdgc-objective ctc \
    >"$OUTPUT_DIR.console.log" 2>&1
EXIT_CODE=$?
if [[ $EXIT_CODE -ne 0 ]]; then
    echo "FDGC CTC calibration failed: $EXIT_CODE" >&2
    exit "$EXIT_CODE"
fi

CALIBRATED_CHECKPOINT="$OUTPUT_DIR/checkpoint_calibrated.pt"
"$PYTHON" scripts/merge_fdgc_checkpoint.py \
    --base-checkpoint "$BASE_CHECKPOINT" \
    --fdgc-checkpoint "$OUTPUT_DIR/checkpoint_fdgc.pt" \
    --split-manifest "$MANIFEST" \
    --output "$CALIBRATED_CHECKPOINT"
EXIT_CODE=$?
if [[ $EXIT_CODE -ne 0 ]]; then
    echo "FDGC checkpoint merge failed: $EXIT_CODE" >&2
    exit "$EXIT_CODE"
fi

FINAL_DIR="$OUTPUT_DIR/final"
log_file="$OUTPUT_DIR/dev_evaluation.log"
"$PYTHON" -u main.py \
    --device 0 \
    --phase test \
    --work-dir "$FINAL_DIR/" \
    --config "$CONFIG" \
    --load-weights "$CALIBRATED_CHECKPOINT" \
    --eval-splits dev \
    --evaluate-tool sclite \
    >"$log_file" 2>&1
EXIT_CODE=$?
if [[ $EXIT_CODE -ne 0 ]]; then
    echo "FDGC CTC Dev evaluation failed: $EXIT_CODE" >&2
    exit "$EXIT_CODE"
fi

echo "FDGC_CTC_CANDIDATE_COMPLETE: $OUTPUT_DIR"
