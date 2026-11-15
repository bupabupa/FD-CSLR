#!/usr/bin/env bash
set -u

ROOT=.
PYTHON=python3
MANIFEST="$ROOT/preprocess/phoenix2014/split_manifest.json"
PRIOR="$ROOT/preprocess/phoenix2014/train_prior_split0.npy"
PROTOCOL="$ROOT/work_dir/fd_xmp_v3/weighted_ensemble_protocol"
STATUS="$PROTOCOL/status.txt"
LOG="$PROTOCOL/protocol.log"
OPTIONAL="$ROOT/work_dir/fd_xmp_v3/candidate_protocol/optional_candidates.json"
SPEC="$PROTOCOL/ensemble_spec.json"
SUMMARY="$PROTOCOL/selection_summary.json"
CHECKPOINT="$PROTOCOL/weighted_ensemble.pt"
CONFIG="$PROTOCOL/weighted_ensemble.yaml"
MANIFEST_OUT="$PROTOCOL/weighted_ensemble_manifest.json"
CAL_DIR="$PROTOCOL/cal"
DEV_DIR="$PROTOCOL/final_dev"
PLACEHOLDER_READY="$PROTOCOL/placeholder_ready.marker"

mkdir -p "$PROTOCOL"
exec >>"$LOG" 2>&1

log() {
    printf '[%s] %s\n' "$(date '+%F %T %Z')" "$1" | tee "$STATUS"
}

fail() {
    log "FAILED $1"
    exit 1
}

wait_for_training() {
    local name="$1"
    local run_dir="$2"
    local log_path="$run_dir/log.txt"
    local count
    while true; do
        count=$(grep -c 'Mean training loss:' "$log_path" 2>/dev/null || true)
        if [[ "$count" -ge 40 ]]; then
            log "READY training $name epochs=$count"
            return 0
        fi
        sleep 60
    done
}

register_placeholder() {
    "$PYTHON" - "$OPTIONAL" "$PROTOCOL" <<'PY'
import json
import os
from pathlib import Path
import sys

optional = Path(sys.argv[1])
protocol = Path(sys.argv[2])
candidate = {
    "name": "weighted_ensemble",
    "dev": str(protocol / "final_dev" / "predictions-dev.jsonl"),
    "run": str(protocol),
    "config": str(protocol / "weighted_ensemble.yaml"),
    "checkpoint": str(protocol / "weighted_ensemble.pt"),
    "priority": 120,
}
items = []
if optional.is_file():
    items = json.loads(optional.read_text(encoding="utf-8"))
    if not isinstance(items, list):
        raise SystemExit("optional candidate registry must be a list")
items = [item for item in items if item.get("name") != candidate["name"]]
items.append(candidate)
items.sort(key=lambda item: str(item.get("name", "")))
optional.parent.mkdir(parents=True, exist_ok=True)
temporary = optional.with_name(optional.name + ".weighted.tmp")
temporary.write_text(
    json.dumps(items, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
os.replace(temporary, optional)
PY
}

average_members() {
    local name="$1"
    local run_dir="$2"
    local config="$3"
    local output_dir="$PROTOCOL/members/$name"
    local checkpoint="$output_dir/checkpoint.pt"
    mkdir -p "$output_dir"
    if [[ ! -f "$checkpoint" ]]; then
        "$PYTHON" scripts/average_checkpoints.py \
            --run-dir "$run_dir" \
            --last-n 3 \
            --output "$checkpoint" \
            >"$output_dir/average.log" 2>&1
        if [[ $? -ne 0 ]]; then
            fail "checkpoint averaging for $name"
        fi
    fi
    printf 'name=%s;config=%s;checkpoint=%s\n' \
        "$name" "$config" "$checkpoint"
}

cd "$ROOT" || fail "cannot enter project root"

register_placeholder
if [[ $? -ne 0 ]]; then
    fail "could not register blocking placeholder"
fi
touch "$PLACEHOLDER_READY"
log "REGISTERED blocking placeholder for weighted_ensemble"

BASELINE="$ROOT/work_dir/fd_xmp_v3/E0_baseline_split_seed0"
VAC="$ROOT/work_dir/fd_xmp_v3/E0_vac_ctc_split_seed0"
VAC_SMKD="$ROOT/work_dir/fd_xmp_v3/E0_vac_smkd_split_seed0"
H2="$ROOT/work_dir/fd_xmp_v3/E4_fdxmp_h2_split_seed0"

wait_for_training baseline "$BASELINE"
wait_for_training vac_ctc "$VAC"
wait_for_training vac_smkd "$VAC_SMKD"
wait_for_training fdxmp_h2 "$H2"

log "START last-three checkpoint averaging"
mapfile -t MEMBER_LINES < <(
    average_members baseline "$BASELINE" "./configs/baseline_split.yaml"
    average_members vac_ctc "$VAC" "./configs/vac_ctc_split.yaml"
    average_members vac_smkd "$VAC_SMKD" "./configs/vac_smkd_split.yaml"
    average_members fdxmp_h2 "$H2" "./configs/fdxmp_h2_split.yaml"
)
if [[ ${#MEMBER_LINES[@]} -ne 4 ]]; then
    fail "expected four averaged members, got ${#MEMBER_LINES[@]}"
fi
log "COMPLETE last-three checkpoint averaging"

MEMBER_ARGS=()
for member in "${MEMBER_LINES[@]}"; do
    MEMBER_ARGS+=(--member "$member")
done

log "START D_cal-only weighted ensemble selection"
"$PYTHON" -u scripts/select_weighted_ensemble.py \
    "${MEMBER_ARGS[@]}" \
    --split-manifest "$MANIFEST" \
    --class-prior-path "$PRIOR" \
    --output-spec "$SPEC" \
    --output-summary "$SUMMARY" \
    --device 0 \
    --batch-size 2 \
    --num-workers 4 \
    --num-processes 10 \
    --beam-width 10 \
    --max-members 3 \
    --refinement-subsets 5 \
    --refinement-samples 30 \
    --decode-candidates 16 \
    --cache-dir "$PROTOCOL/cal_logits_cache" \
    >"$PROTOCOL/selection.console.log" 2>&1
if [[ $? -ne 0 ]]; then
    fail "D_cal weighted ensemble selection"
fi
log "COMPLETE D_cal weighted ensemble selection"

log "START weighted ensemble checkpoint bundle"
"$PYTHON" -u scripts/bundle_weighted_ensemble.py \
    --spec "$SPEC" \
    --split-manifest "$MANIFEST" \
    --output-checkpoint "$CHECKPOINT" \
    --output-config "$CONFIG" \
    --output-manifest "$MANIFEST_OUT" \
    >"$PROTOCOL/bundle.console.log" 2>&1
if [[ $? -ne 0 ]]; then
    fail "weighted ensemble checkpoint bundle"
fi
log "COMPLETE weighted ensemble checkpoint bundle"

mkdir -p "$CAL_DIR"
log "START final D_cal evaluation of frozen ensemble"
"$PYTHON" -u scripts/evaluate_logit_ensemble.py \
    --ensemble-spec "$SPEC" \
    --split cal \
    --split-manifest "$MANIFEST" \
    --output-dir "$CAL_DIR" \
    --device 0 \
    --batch-size 2 \
    --num-workers 4 \
    --beam-width 10 \
    --num-processes 10 \
    --evaluate-tool python \
    >"$CAL_DIR.console.log" 2>&1
if [[ $? -ne 0 ]]; then
    fail "final D_cal ensemble evaluation"
fi
log "COMPLETE final D_cal evaluation"

if [[ -f "$ROOT/work_dir/fd_xmp_v3/candidate_protocol/locked_candidate.json" ]]; then
    fail "candidate lock was created before weighted ensemble registration"
fi
if [[ -f "$ROOT/work_dir/fd_xmp_v3/candidate_protocol/test_access_claim.json" ]]; then
    fail "Test access claim exists before weighted ensemble registration"
fi
if [[ -f "$DEV_DIR/predictions-dev.jsonl" ]]; then
    fail "D_sel was already evaluated for weighted ensemble"
fi

log "START one-time D_sel evaluation of frozen ensemble"
"$PYTHON" -u main.py \
    --device 0 \
    --phase test \
    --work-dir "$DEV_DIR/" \
    --config "$CONFIG" \
    --load-weights "$CHECKPOINT" \
    --eval-splits dev \
    --evaluate-tool sclite \
    >"$DEV_DIR.console.log" 2>&1
if [[ $? -ne 0 ]]; then
    fail "one-time D_sel weighted ensemble evaluation"
fi
if [[ ! -f "$DEV_DIR/predictions-dev.jsonl" ]]; then
    fail "D_sel prediction artifact is missing"
fi
log "COMPLETE one-time D_sel evaluation"

"$PYTHON" - "$OPTIONAL" "$PROTOCOL" <<'PY'
import json
import os
from pathlib import Path
import sys

optional = Path(sys.argv[1])
protocol = Path(sys.argv[2])
candidate = {
    "name": "weighted_ensemble",
    "dev": str(protocol / "final_dev" / "predictions-dev.jsonl"),
    "run": str(protocol),
    "config": str(protocol / "weighted_ensemble.yaml"),
    "checkpoint": str(protocol / "weighted_ensemble.pt"),
    "priority": 120,
}
items = []
if optional.is_file():
    items = json.loads(optional.read_text(encoding="utf-8"))
    if not isinstance(items, list):
        raise SystemExit("optional candidate registry must be a list")
items = [item for item in items if item.get("name") != candidate["name"]]
items.append(candidate)
items.sort(key=lambda item: str(item.get("name", "")))
optional.parent.mkdir(parents=True, exist_ok=True)
temporary = optional.with_name(optional.name + ".weighted.tmp")
temporary.write_text(
    json.dumps(items, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
os.replace(temporary, optional)
PY
if [[ $? -ne 0 ]]; then
    fail "weighted ensemble optional candidate registration"
fi

log "WEIGHTED_ENSEMBLE_PROTOCOL_COMPLETE config=$CONFIG checkpoint=$CHECKPOINT"
