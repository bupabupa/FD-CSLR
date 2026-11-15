#!/usr/bin/env bash
set -u

ROOT=.
PYTHON=python3
SIGNER="${1:?usage: run_leave_one_signer_training.sh SIGNER [CONFIG] [TAG]}"
CONFIG="${2:-./configs/leave_one_signers/$SIGNER.yaml}"
TAG="${3:-fdxmp_h1}"
TAG="${TAG//\//_}"
EPOCHS="${NUM_EPOCHS:-40}"
MANIFEST="$ROOT/preprocess/phoenix2014/leave_one_signers/$SIGNER/split_manifest.json"
RUN_DIR="$ROOT/work_dir/fd_xmp_v3/leave_one_signers/$SIGNER/$TAG"

if [[ ! -f "$MANIFEST" ]]; then
    echo "leave-one-signer manifest not found: $MANIFEST" >&2
    exit 2
fi
mkdir -p "$RUN_DIR"
cd "$ROOT" || exit 3

"$PYTHON" -u main.py \
    --device 0 \
    --work-dir "$RUN_DIR/" \
    --config "$CONFIG" \
    --split-manifest "$MANIFEST" \
    --num-epoch "$EPOCHS" \
    --save-interval 5 \
    --evaluate-tool sclite \
    >"$RUN_DIR.console.log" 2>&1
EXIT_CODE=$?
if [[ $EXIT_CODE -ne 0 ]]; then
    echo "leave-one-signer training failed: $EXIT_CODE" >&2
    exit "$EXIT_CODE"
fi

BEST_CHECKPOINT=$(find "$RUN_DIR" -maxdepth 1 \
    -name 'dev_*_epoch*_model.pt' -printf '%f\n' |
    sort -t_ -k2,2n |
    head -1)
if [[ -z "$BEST_CHECKPOINT" ]]; then
    echo "leave-one-signer checkpoint not found in $RUN_DIR" >&2
    exit 4
fi

"$PYTHON" -u main.py \
    --device 0 \
    --phase test \
    --work-dir "$RUN_DIR/final/" \
    --config "$CONFIG" \
    --split-manifest "$MANIFEST" \
    --load-weights "$RUN_DIR/$BEST_CHECKPOINT" \
    --eval-splits dev \
    --evaluate-tool sclite \
    >"$RUN_DIR/final.console.log" 2>&1
EXIT_CODE=$?
if [[ $EXIT_CODE -ne 0 ]]; then
    echo "leave-one-signer Dev evaluation failed: $EXIT_CODE" >&2
    exit "$EXIT_CODE"
fi

echo "LEAVE_ONE_SIGNER_COMPLETE: $SIGNER $TAG"
echo "checkpoint: $RUN_DIR/$BEST_CHECKPOINT"
