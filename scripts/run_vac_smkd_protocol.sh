#!/usr/bin/env bash
set -u

ROOT=.
PYTHON=python3
NGRAM_PID="${1:-771744}"
STATUS_DIR="$ROOT/work_dir/fd_xmp_v3/vac_smkd_protocol"
STATUS_FILE="$STATUS_DIR/status.txt"
NGRAM_STATUS="$ROOT/work_dir/fd_xmp_v3/ngram_protocol/status.txt"
RUN_DIR="$ROOT/work_dir/fd_xmp_v3/E0_vac_smkd_split_seed0"
CONFIG="./configs/vac_smkd_split.yaml"
OPTIONAL_PATH="$ROOT/work_dir/fd_xmp_v3/candidate_protocol/optional_candidates.json"

mkdir -p "$STATUS_DIR" "$RUN_DIR/final"
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

log "WAIT strict n-gram pid=$NGRAM_PID"
wait_for_pid "$NGRAM_PID"
if ! grep -q 'NGRAM_PROTOCOL_COMPLETE' "$NGRAM_STATUS"; then
    log "FAILED strict n-gram protocol did not complete"
    exit 20
fi

cd "$ROOT" || exit 21
log "START VAC+SMKD strict-split training"
"$PYTHON" -u main.py \
    --device 0 \
    --work-dir "$RUN_DIR/" \
    --config "$CONFIG" \
    --num-epoch 40 \
    --save-interval 5 \
    --evaluate-tool sclite \
    >"$RUN_DIR.console.log" 2>&1
EXIT_CODE=$?
if [[ "$EXIT_CODE" -ne 0 ]]; then
    log "FAILED VAC+SMKD training exit_code=$EXIT_CODE"
    exit "$EXIT_CODE"
fi
if ! training_completed "$RUN_DIR/log.txt"; then
    log "FAILED VAC+SMKD training incomplete_log"
    exit 22
fi
log "COMPLETE VAC+SMKD training"

CHECKPOINT_NAME=$(best_checkpoint "$RUN_DIR")
if [[ -z "$CHECKPOINT_NAME" ]]; then
    log "FAILED VAC+SMKD checkpoint not found"
    exit 23
fi
CHECKPOINT="$RUN_DIR/$CHECKPOINT_NAME"

log "START VAC+SMKD Dev selection"
"$PYTHON" -u main.py \
    --device 0 \
    --phase test \
    --work-dir "$RUN_DIR/final/" \
    --config "$CONFIG" \
    --load-weights "$CHECKPOINT" \
    --eval-splits dev \
    --evaluate-tool sclite \
    >"$RUN_DIR/final.console.log" 2>&1
EXIT_CODE=$?
if [[ "$EXIT_CODE" -ne 0 ]]; then
    log "FAILED VAC+SMKD Dev selection exit_code=$EXIT_CODE"
    exit "$EXIT_CODE"
fi
log "COMPLETE VAC+SMKD Dev selection"

VAC_SMKD_NGRAM_STATUS_DIR="$STATUS_DIR/ngram"
log "START VAC+SMKD same-rescorer evaluation"
"$PYTHON" -u scripts/run_ngram_for_checkpoint.py \
    --base-run "$RUN_DIR" \
    --base-config "$ROOT/configs/vac_smkd_split.yaml" \
    --checkpoint "$CHECKPOINT" \
    --status-dir "$VAC_SMKD_NGRAM_STATUS_DIR" \
    --candidate-name vac_smkd_ngram \
    --candidate-priority 94
EXIT_CODE=$?
if [[ "$EXIT_CODE" -ne 0 ]]; then
    log "FAILED VAC+SMKD same-rescorer evaluation exit_code=$EXIT_CODE"
    exit "$EXIT_CODE"
fi
log "COMPLETE VAC+SMKD same-rescorer evaluation"

DEV_PREDICTION="$RUN_DIR/final/predictions-dev.jsonl"
if [[ ! -f "$DEV_PREDICTION" ]]; then
    log "FAILED VAC+SMKD Dev prediction missing"
    exit 24
fi

"$PYTHON" - "$OPTIONAL_PATH" "$RUN_DIR" "$CONFIG" "$CHECKPOINT" \
    "$DEV_PREDICTION" <<'PY'
import json
from pathlib import Path
import sys

optional_path = Path(sys.argv[1])
run_dir = Path(sys.argv[2]).resolve()
config = (Path.cwd() / sys.argv[3]).resolve()
checkpoint = Path(sys.argv[4]).resolve()
dev_prediction = Path(sys.argv[5]).resolve()
candidate = {
    "name": "vac_smkd",
    "dev": str(dev_prediction),
    "run": str(run_dir),
    "config": str(config),
    "checkpoint": str(checkpoint),
    "priority": 95,
}
items = []
if optional_path.is_file():
    items = json.loads(optional_path.read_text(encoding="utf-8"))
    if not isinstance(items, list):
        raise ValueError("optional candidate file must contain a list")
items = [item for item in items if item.get("name") != candidate["name"]]
items.append(candidate)
items.sort(key=lambda item: item["name"])
optional_path.parent.mkdir(parents=True, exist_ok=True)
temporary = optional_path.with_suffix(optional_path.suffix + ".tmp")
temporary.write_text(
    json.dumps(items, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
temporary.replace(optional_path)
PY
EXIT_CODE=$?
if [[ "$EXIT_CODE" -ne 0 ]]; then
    log "FAILED VAC+SMKD candidate registration exit_code=$EXIT_CODE"
    exit "$EXIT_CODE"
fi

log "VAC_SMKD_PROTOCOL_COMPLETE checkpoint=$CHECKPOINT"
