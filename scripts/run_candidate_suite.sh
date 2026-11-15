#!/usr/bin/env bash
set -u

ROOT=.
PYTHON=python3
MANIFEST="$ROOT/preprocess/phoenix2014/split_manifest.json"
H1_PID="${1:-826569}"
VAC_SMKD_PID="${2:-0}"
STATUS_DIR="$ROOT/work_dir/fd_xmp_v3/candidate_protocol"
STATUS_FILE="$STATUS_DIR/status.txt"
VAC_SMKD_STATUS="$ROOT/work_dir/fd_xmp_v3/vac_smkd_protocol/status.txt"
VAC_SMKD_NGRAM_STATUS="$ROOT/work_dir/fd_xmp_v3/vac_smkd_protocol/ngram/status.txt"
NGRAM_STATUS="$ROOT/work_dir/fd_xmp_v3/ngram_protocol/status.txt"
OPTIONAL_PATH="$STATUS_DIR/optional_candidates.json"
COSINE_DIR="$ROOT/work_dir/fd_xmp_v3/E1_fdxmp_v3_split_seed0"
GAUSSIAN_DIR="$ROOT/work_dir/fd_xmp_v3/E3_fdxmp_gaussian_split_seed0"
VAC_SMKD_RUN_DIR="$ROOT/work_dir/fd_xmp_v3/E0_vac_smkd_split_seed0"
COSINE_CONFIG="./configs/fdxmp_v3_split.yaml"
GAUSSIAN_CONFIG="./configs/fdxmp_gaussian_split.yaml"
AVERAGE_LAST_N=3

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
    local completed_epochs
    completed_epochs=$(grep -c 'Mean training loss:' "$run_dir/log.txt" || true)
    if [[ "$completed_epochs" -lt 40 ]]; then
        log "FAILED training $name incomplete_log"
        return 11
    fi
    log "COMPLETE training $name"
}

select_checkpoint() {
    local run_dir="$1"
    local checkpoint
    checkpoint=$(best_checkpoint "$run_dir")
    if [[ -z "$checkpoint" ]]; then
        log "FAILED checkpoint not found in $run_dir"
        return 12
    fi
    printf '%s/%s\n' "$run_dir" "$checkpoint"
}

run_dev_selection() {
    local name="$1"
    local config="$2"
    local checkpoint="$3"
    local output_dir="$4"

    log "START Dev selection $name"
    cd "$ROOT" || return 10
    "$PYTHON" -u main.py \
        --device 0 \
        --phase test \
        --work-dir "$output_dir/" \
        --config "$config" \
        --load-weights "$checkpoint" \
        --eval-splits dev \
        --evaluate-tool sclite \
        >"$output_dir.console.log" 2>&1
    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        log "FAILED Dev selection $name exit_code=$exit_code"
        return "$exit_code"
    fi
    log "COMPLETE Dev selection $name"
}

run_average_candidate() {
    local name="$1"
    local config="$2"
    local run_dir="$3"
    local output_dir="$run_dir/parity_average_last${AVERAGE_LAST_N}"
    local checkpoint="$output_dir/checkpoint.pt"

    log "START parity-averaged candidate $name"
    mkdir -p "$output_dir"
    "$PYTHON" scripts/average_checkpoints.py \
        --run-dir "$run_dir" \
        --last-n "$AVERAGE_LAST_N" \
        --output "$checkpoint" \
        >"$output_dir.average.log" 2>&1
    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        log "FAILED checkpoint averaging $name exit_code=$exit_code"
        return "$exit_code"
    fi
    run_dev_evaluation \
        "parity_average_last${AVERAGE_LAST_N}_$name" \
        "$config" \
        "$checkpoint" \
        "$output_dir"
}

run_fdgc_viterbi() {
    local name="$1"
    local config="$2"
    local checkpoint="$3"
    local selection_dir="$4"
    local output_dir="$5"
    local use_difficulty="${6:-false}"
    local difficulty_args=()

    if [[ "$use_difficulty" == "true" ]]; then
        difficulty_args+=(--fdgc-use-difficulty)
    fi

    log "START FDGC-Viterbi $name"
    cd "$ROOT" || return 10
    "$PYTHON" -u scripts/train_fdgc.py \
        --device 0 \
        --config "$config" \
        --load-weights "$checkpoint" \
        --split-manifest "$MANIFEST" \
        --difficulty-report "$selection_dir/frequency-metrics-dev.json" \
        --work-dir "$output_dir/" \
        --fdgc-epochs 10 \
        --fdgc-lr 0.001 \
        --fdgc-objective viterbi \
        "${difficulty_args[@]}" \
        >"$output_dir.console.log" 2>&1
    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        log "FAILED FDGC-Viterbi $name exit_code=$exit_code"
        return "$exit_code"
    fi
    "$PYTHON" scripts/merge_fdgc_checkpoint.py \
        --base-checkpoint "$checkpoint" \
        --fdgc-checkpoint "$output_dir/checkpoint_fdgc.pt" \
        --split-manifest "$MANIFEST" \
        --output "$output_dir/checkpoint_calibrated.pt"
    exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        log "FAILED FDGC merge $name exit_code=$exit_code"
        return "$exit_code"
    fi
    log "COMPLETE FDGC-Viterbi $name"
}

run_dev_evaluation() {
    local name="$1"
    local config="$2"
    local checkpoint="$3"
    local output_dir="$4"

    log "START final Dev evaluation $name"
    cd "$ROOT" || return 10
    "$PYTHON" -u main.py \
        --device 0 \
        --phase test \
        --work-dir "$output_dir/" \
        --config "$config" \
        --load-weights "$checkpoint" \
        --eval-splits dev \
        --evaluate-tool sclite \
        >"$output_dir.console.log" 2>&1
    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        log "FAILED final Dev evaluation $name exit_code=$exit_code"
        return "$exit_code"
    fi
    log "COMPLETE final Dev evaluation $name"
}

run_test_evaluation() {
    local name="$1"
    local config="$2"
    local checkpoint="$3"
    local output_dir="$4"

    log "START locked Test evaluation $name"
    cd "$ROOT" || return 10
    "$PYTHON" -u main.py \
        --device 0 \
        --phase test \
        --work-dir "$output_dir/" \
        --config "$config" \
        --load-weights "$checkpoint" \
        --eval-splits test \
        --evaluate-tool sclite \
        >"$output_dir.console.log" 2>&1
    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        log "FAILED locked Test evaluation $name exit_code=$exit_code"
        return "$exit_code"
    fi
    log "COMPLETE locked Test evaluation $name"
}

log "WAIT H1 protocol pid=$H1_PID"
wait_for_pid "$H1_PID"
if ! grep -q 'H1_PROTOCOL_COMPLETE' \
    "$ROOT/work_dir/fd_xmp_v3/h1_protocol/status.txt"; then
    log "FAILED H1 protocol did not complete"
    exit 20
fi

if [[ "$VAC_SMKD_PID" -gt 0 ]]; then
    log "WAIT VAC+SMKD protocol pid=$VAC_SMKD_PID"
    wait_for_pid "$VAC_SMKD_PID"
fi
if ! grep -q 'VAC_SMKD_PROTOCOL_COMPLETE' "$VAC_SMKD_STATUS"; then
    log "FAILED VAC+SMKD protocol did not complete"
    exit 22
fi
if ! grep -q 'NGRAM_PROTOCOL_COMPLETE' "$NGRAM_STATUS"; then
    log "FAILED strict n-gram protocol did not complete"
    exit 23
fi
"$PYTHON" - "$OPTIONAL_PATH" "$NGRAM_STATUS" "$VAC_SMKD_NGRAM_STATUS" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
ngram_status = Path(sys.argv[2]).read_text(
    encoding="utf-8", errors="replace"
)
vac_ngram_status = Path(sys.argv[3]).read_text(
    encoding="utf-8", errors="replace"
)
if not path.is_file():
    raise SystemExit("optional candidate registry is missing: {}".format(path))
candidates = json.loads(path.read_text(encoding="utf-8"))
if not isinstance(candidates, list):
    raise SystemExit("optional candidate registry must contain a list")
available = {
    item.get("name")
    for item in candidates
    if isinstance(item, dict)
}
required = {"vac_smkd"}
if "no_eligible_candidate" not in ngram_status:
    required.add("fdxmp_ngram_calibrated")
if "NGRAM_PROTOCOL_COMPLETE" not in vac_ngram_status:
    raise SystemExit(
        "VAC+SMKD same-rescorer protocol did not complete"
    )
if "no_eligible_candidate" not in vac_ngram_status:
    required.add("vac_smkd_ngram")
missing = required.difference(available)
if missing:
    raise SystemExit(
        "optional candidate registry is missing: {}".format(
            ", ".join(sorted(missing))
        )
    )
PY
exit_code=$?
if [[ $exit_code -ne 0 ]]; then
    log "FAILED optional candidate registry validation exit_code=$exit_code"
    exit "$exit_code"
fi
log "VALIDATED optional candidates and both applicable n-gram statuses"

VAC_SMKD_NGRAM_CONFIG=""
VAC_SMKD_NGRAM_CHECKPOINT=""
IFS='|' read -r VAC_SMKD_NGRAM_CONFIG VAC_SMKD_NGRAM_CHECKPOINT < <(
    "$PYTHON" - "$OPTIONAL_PATH" <<'PY'
import json
import sys

items = json.load(open(sys.argv[1], encoding="utf-8"))
selected = next(
    (
        item for item in items
        if isinstance(item, dict) and item.get("name") == "vac_smkd_ngram"
    ),
    None,
)
if selected is not None:
    print("{}|{}".format(selected["config"], selected["checkpoint"]))
PY
)

COSINE_CHECKPOINT=$(select_checkpoint "$COSINE_DIR") || exit $?
if [[ ! -f "$COSINE_DIR/selection/frequency-metrics-dev.json" ]]; then
    log "FAILED cosine difficulty report missing"
    exit 21
fi

log "START FDGC-CTC cosine candidate"
"$ROOT/scripts/run_fdgc_ctc_candidate.sh" \
    "$COSINE_DIR" \
    "$COSINE_CHECKPOINT" \
    "$COSINE_DIR/selection/frequency-metrics-dev.json" \
    "$COSINE_DIR/fdgc_ctc" \
    "$COSINE_CONFIG"
exit_code=$?
if [[ $exit_code -ne 0 ]]; then
    log "FAILED FDGC-CTC cosine candidate exit_code=$exit_code"
    exit "$exit_code"
fi
log "COMPLETE FDGC-CTC cosine candidate"

run_fdgc_viterbi \
    "E1_fdxmp_cosine_fdgc_fd" \
    "$COSINE_CONFIG" \
    "$COSINE_CHECKPOINT" \
    "$COSINE_DIR/selection" \
    "$COSINE_DIR/fdgc_fd" \
    true || exit $?
run_dev_evaluation \
    "E1_fdxmp_cosine_fdgc_fd" \
    "$COSINE_CONFIG" \
    "$COSINE_DIR/fdgc_fd/checkpoint_calibrated.pt" \
    "$COSINE_DIR/final_fdgc_fd" || exit $?

run_training \
    "E3_fdxmp_gaussian_split_seed0" \
    "$GAUSSIAN_CONFIG" \
    "$GAUSSIAN_DIR" || exit $?
GAUSSIAN_CHECKPOINT=$(select_checkpoint "$GAUSSIAN_DIR") || exit $?
log "START Gaussian prototype coverage export"
"$PYTHON" scripts/export_prototype_coverage.py \
    --checkpoint "$GAUSSIAN_CHECKPOINT" \
    --output "$GAUSSIAN_DIR/prototype_coverage.json" \
    >>"$STATUS_DIR/prototype_coverage.console.log" 2>&1
exit_code=$?
if [[ $exit_code -ne 0 ]]; then
    log "FAILED Gaussian prototype coverage exit_code=$exit_code"
    exit "$exit_code"
fi
log "COMPLETE Gaussian prototype coverage export"
GAUSSIAN_SELECTION="$GAUSSIAN_DIR/selection"
run_dev_selection \
    "E3_fdxmp_gaussian" \
    "$GAUSSIAN_CONFIG" \
    "$GAUSSIAN_CHECKPOINT" \
    "$GAUSSIAN_SELECTION" || exit $?

run_fdgc_viterbi \
    "E3_fdxmp_gaussian" \
    "$GAUSSIAN_CONFIG" \
    "$GAUSSIAN_CHECKPOINT" \
    "$GAUSSIAN_SELECTION" \
    "$GAUSSIAN_DIR/fdgc" || exit $?
run_dev_evaluation \
    "E3_fdxmp_gaussian_viterbi" \
    "$GAUSSIAN_CONFIG" \
    "$GAUSSIAN_DIR/fdgc/checkpoint_calibrated.pt" \
    "$GAUSSIAN_DIR/final_viterbi" || exit $?

log "START FDGC-CTC Gaussian candidate"
"$ROOT/scripts/run_fdgc_ctc_candidate.sh" \
    "$GAUSSIAN_DIR" \
    "$GAUSSIAN_CHECKPOINT" \
    "$GAUSSIAN_SELECTION/frequency-metrics-dev.json" \
    "$GAUSSIAN_DIR/fdgc_ctc" \
    "$GAUSSIAN_CONFIG"
exit_code=$?
if [[ $exit_code -ne 0 ]]; then
    log "FAILED FDGC-CTC Gaussian candidate exit_code=$exit_code"
    exit "$exit_code"
fi
log "COMPLETE FDGC-CTC Gaussian candidate"

H2_DIR="$ROOT/work_dir/fd_xmp_v3/E4_fdxmp_h2_split_seed0"
H2_CONFIG="./configs/fdxmp_h2_split.yaml"
H2_CALIBRATED_CONFIG="./configs/fdxmp_h2_calibrated.yaml"
run_training \
    "E4_fdxmp_h2_split_seed0" \
    "$H2_CONFIG" \
    "$H2_DIR" || exit $?
H2_CHECKPOINT=$(select_checkpoint "$H2_DIR") || exit $?
log "START H2 prototype coverage export"
"$PYTHON" scripts/export_prototype_coverage.py \
    --checkpoint "$H2_CHECKPOINT" \
    --output "$H2_DIR/prototype_coverage.json" \
    >>"$STATUS_DIR/prototype_coverage.console.log" 2>&1
exit_code=$?
if [[ $exit_code -ne 0 ]]; then
    log "FAILED H2 prototype coverage exit_code=$exit_code"
    exit "$exit_code"
fi
log "COMPLETE H2 prototype coverage export"

H2_BASE_SELECTION="$H2_DIR/selection"
run_dev_selection \
    "E4_fdxmp_h2_base" \
    "$H2_CONFIG" \
    "$H2_CHECKPOINT" \
    "$H2_BASE_SELECTION" || exit $?

H2_CALIBRATION_DIR="$H2_DIR/multichannel"
log "START H2 multichannel calibration"
"$PYTHON" -u scripts/train_channel_calibration.py \
    --device 0 \
    --config "$H2_CONFIG" \
    --load-weights "$H2_CHECKPOINT" \
    --split-manifest "$MANIFEST" \
    --work-dir "$H2_CALIBRATION_DIR/" \
    --calibration-epochs 10 \
    --calibration-lr 0.01 \
    --calibration-weight-decay 0.0 \
    --calibration-regularization 0.0001 \
    >"$H2_CALIBRATION_DIR.console.log" 2>&1
exit_code=$?
if [[ $exit_code -ne 0 ]]; then
    log "FAILED H2 multichannel calibration exit_code=$exit_code"
    exit "$exit_code"
fi
H2_USE_CALIBRATION=$("$PYTHON" -c \
    'import json,sys; print(str(json.load(open(sys.argv[1]))["use_calibration"]).lower())' \
    "$H2_CALIBRATION_DIR/multichannel_metrics.json")
exit_code=$?
if [[ $exit_code -ne 0 ]]; then
    log "FAILED H2 multichannel selection read exit_code=$exit_code"
    exit "$exit_code"
fi
if [[ "$H2_USE_CALIBRATION" == "true" ]]; then
    log "H2 multichannel calibration selected on D_cal"
    H2_EFFECTIVE_CHECKPOINT="$H2_CALIBRATION_DIR/checkpoint_multichannel.pt"
    "$PYTHON" scripts/merge_multichannel_checkpoint.py \
        --base-checkpoint "$H2_CHECKPOINT" \
        --calibration-checkpoint "$H2_CALIBRATION_DIR/checkpoint_calibration.pt" \
        --split-manifest "$MANIFEST" \
        --output "$H2_EFFECTIVE_CHECKPOINT"
    exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        log "FAILED H2 multichannel checkpoint merge exit_code=$exit_code"
        exit "$exit_code"
    fi
    H2_EFFECTIVE_CONFIG="$H2_CALIBRATED_CONFIG"
else
    log "H2 multichannel calibration did not beat native fusion on D_cal"
    H2_EFFECTIVE_CHECKPOINT="$H2_CHECKPOINT"
    H2_EFFECTIVE_CONFIG="$H2_CONFIG"
fi
log "COMPLETE H2 multichannel calibration"

H2_FDGC_DIR="$H2_DIR/fdgc"
log "START H2 FDGC calibration"
"$PYTHON" -u scripts/train_fdgc.py \
    --device 0 \
    --config "$H2_EFFECTIVE_CONFIG" \
    --load-weights "$H2_EFFECTIVE_CHECKPOINT" \
    --split-manifest "$MANIFEST" \
    --work-dir "$H2_FDGC_DIR/" \
    --fdgc-epochs 10 \
    --fdgc-lr 0.001 \
    --fdgc-objective viterbi \
    >"$H2_FDGC_DIR.console.log" 2>&1
exit_code=$?
if [[ $exit_code -ne 0 ]]; then
    log "FAILED H2 FDGC calibration exit_code=$exit_code"
    exit "$exit_code"
fi
H2_FINAL_CHECKPOINT="$H2_FDGC_DIR/checkpoint_calibrated.pt"
"$PYTHON" scripts/merge_fdgc_checkpoint.py \
    --base-checkpoint "$H2_EFFECTIVE_CHECKPOINT" \
    --fdgc-checkpoint "$H2_FDGC_DIR/checkpoint_fdgc.pt" \
    --split-manifest "$MANIFEST" \
    --output "$H2_FINAL_CHECKPOINT"
exit_code=$?
if [[ $exit_code -ne 0 ]]; then
    log "FAILED H2 FDGC checkpoint merge exit_code=$exit_code"
    exit "$exit_code"
fi
log "COMPLETE H2 FDGC calibration"

H2_FINAL_DIR="$H2_DIR/final"
run_dev_evaluation \
    "E4_fdxmp_h2_final" \
    "$H2_EFFECTIVE_CONFIG" \
    "$H2_FINAL_CHECKPOINT" \
    "$H2_FINAL_DIR" || exit $?

BASELINE_DIR="$ROOT/work_dir/fd_xmp_v3/E0_baseline_split_seed0"
VAC_DIR="$ROOT/work_dir/fd_xmp_v3/E0_vac_ctc_split_seed0"
H1_DIR="$ROOT/work_dir/fd_xmp_v3/E2_fdxmp_h1_split_seed0"
BASELINE_CHECKPOINT=$(select_checkpoint "$BASELINE_DIR") || exit $?
VAC_CHECKPOINT=$(select_checkpoint "$VAC_DIR") || exit $?
VAC_SMKD_CHECKPOINT=$(select_checkpoint "$VAC_SMKD_RUN_DIR") || exit $?
H1_CHECKPOINT=$(select_checkpoint "$H1_DIR") || exit $?

BASELINE_AVG_DIR="$BASELINE_DIR/parity_average_last${AVERAGE_LAST_N}"
VAC_AVG_DIR="$VAC_DIR/parity_average_last${AVERAGE_LAST_N}"
VAC_SMKD_AVG_DIR="$VAC_SMKD_RUN_DIR/parity_average_last${AVERAGE_LAST_N}"
COSINE_AVG_DIR="$COSINE_DIR/parity_average_last${AVERAGE_LAST_N}"
GAUSSIAN_AVG_DIR="$GAUSSIAN_DIR/parity_average_last${AVERAGE_LAST_N}"
H1_AVG_DIR="$H1_DIR/parity_average_last${AVERAGE_LAST_N}"
H2_AVG_DIR="$H2_DIR/parity_average_last${AVERAGE_LAST_N}"

run_average_candidate \
    "baseline" \
    "./configs/baseline_split.yaml" \
    "$BASELINE_DIR" || exit $?
run_average_candidate \
    "vac_ctc" \
    "./configs/vac_ctc_split.yaml" \
    "$VAC_DIR" || exit $?
run_average_candidate \
    "vac_smkd" \
    "./configs/vac_smkd_split.yaml" \
    "$VAC_SMKD_RUN_DIR" || exit $?
run_average_candidate \
    "fdxmp_cosine" \
    "$COSINE_CONFIG" \
    "$COSINE_DIR" || exit $?
run_average_candidate \
    "fdxmp_gaussian" \
    "$GAUSSIAN_CONFIG" \
    "$GAUSSIAN_DIR" || exit $?
run_average_candidate \
    "fdxmp_h1" \
    "./configs/fdxmp_h1_split.yaml" \
    "$H1_DIR" || exit $?
run_average_candidate \
    "fdxmp_h2" \
    "$H2_CONFIG" \
    "$H2_DIR" || exit $?

LOCK_JSON="$STATUS_DIR/locked_candidate.json"
LOCK_ENV="$STATUS_DIR/locked_candidate.env"
LOCKED_TEST_ROOT="$STATUS_DIR/locked_test"
SUMMARY_PATH="$STATUS_DIR/final_comparison.json"
TEST_ACCESS_CLAIM="$STATUS_DIR/test_access_claim.json"

if [[ -e "$LOCK_JSON" || -e "$LOCK_ENV" ]]; then
    log "FAILED a candidate lock already exists; refusing to overwrite it"
    exit 30
fi
if [[ -e "$TEST_ACCESS_CLAIM" ]]; then
    log "FAILED Test access was already claimed; refusing a second pass"
    exit 30
fi
if [[ -e "$SUMMARY_PATH" ]]; then
    log "FAILED a locked Test comparison already exists; refusing a second pass"
    exit 30
fi
if find "$LOCKED_TEST_ROOT" -type f -name 'predictions-test.jsonl' \
    -print -quit 2>/dev/null | grep -q .; then
    log "FAILED locked Test predictions already exist; refusing a second pass"
    exit 30
fi

log "START Dev-only candidate lock"
cd "$ROOT" || exit 22
"$PYTHON" scripts/select_candidate.py \
    --candidate "baseline;dev=$BASELINE_DIR/final/predictions-dev.jsonl;run=$BASELINE_DIR;config=./configs/baseline_split.yaml;checkpoint=$BASELINE_CHECKPOINT;priority=100" \
    --candidate "vac_ctc;dev=$VAC_DIR/final/predictions-dev.jsonl;run=$VAC_DIR;config=./configs/vac_ctc_split.yaml;checkpoint=$VAC_CHECKPOINT;priority=90" \
    --candidate "fdxmp_viterbi;dev=$COSINE_DIR/final/predictions-dev.jsonl;run=$COSINE_DIR;config=$COSINE_CONFIG;checkpoint=$COSINE_DIR/fdgc/checkpoint_calibrated.pt;calibration=$COSINE_DIR/fdgc/fdgc_metrics.json;priority=60" \
    --candidate "fdxmp_fdgc_ctc;dev=$COSINE_DIR/fdgc_ctc/final/predictions-dev.jsonl;run=$COSINE_DIR;config=$COSINE_CONFIG;checkpoint=$COSINE_DIR/fdgc_ctc/checkpoint_calibrated.pt;calibration=$COSINE_DIR/fdgc_ctc/fdgc_metrics.json;priority=50" \
    --candidate "fdxmp_fdgc_fd;dev=$COSINE_DIR/final_fdgc_fd/predictions-dev.jsonl;run=$COSINE_DIR;config=$COSINE_CONFIG;checkpoint=$COSINE_DIR/fdgc_fd/checkpoint_calibrated.pt;calibration=$COSINE_DIR/fdgc_fd/fdgc_metrics.json;priority=40" \
    --candidate "gaussian_viterbi;dev=$GAUSSIAN_DIR/final_viterbi/predictions-dev.jsonl;run=$GAUSSIAN_DIR;config=$GAUSSIAN_CONFIG;checkpoint=$GAUSSIAN_DIR/fdgc/checkpoint_calibrated.pt;calibration=$GAUSSIAN_DIR/fdgc/fdgc_metrics.json;priority=60" \
    --candidate "gaussian_fdgc_ctc;dev=$GAUSSIAN_DIR/fdgc_ctc/final/predictions-dev.jsonl;run=$GAUSSIAN_DIR;config=$GAUSSIAN_CONFIG;checkpoint=$GAUSSIAN_DIR/fdgc_ctc/checkpoint_calibrated.pt;calibration=$GAUSSIAN_DIR/fdgc_ctc/fdgc_metrics.json;priority=50" \
    --candidate "fdxmp_h1;dev=$H1_DIR/final/predictions-dev.jsonl;run=$H1_DIR;config=./configs/fdxmp_h1_split.yaml;checkpoint=$H1_DIR/fdgc/checkpoint_calibrated.pt;calibration=$H1_DIR/fdgc/fdgc_metrics.json;priority=30" \
    --candidate "fdxmp_h2;dev=$H2_FINAL_DIR/predictions-dev.jsonl;run=$H2_DIR;config=$H2_EFFECTIVE_CONFIG;checkpoint=$H2_FINAL_CHECKPOINT;calibration=$H2_FDGC_DIR/fdgc_metrics.json;priority=0" \
    --candidate "baseline_average3;dev=$BASELINE_AVG_DIR/predictions-dev.jsonl;run=$BASELINE_DIR;config=./configs/baseline_split.yaml;checkpoint=$BASELINE_AVG_DIR/checkpoint.pt;priority=101" \
    --candidate "vac_ctc_average3;dev=$VAC_AVG_DIR/predictions-dev.jsonl;run=$VAC_DIR;config=./configs/vac_ctc_split.yaml;checkpoint=$VAC_AVG_DIR/checkpoint.pt;priority=91" \
    --candidate "vac_smkd_average3;dev=$VAC_SMKD_AVG_DIR/predictions-dev.jsonl;run=$VAC_SMKD_RUN_DIR;config=./configs/vac_smkd_split.yaml;checkpoint=$VAC_SMKD_AVG_DIR/checkpoint.pt;priority=96" \
    --candidate "fdxmp_cosine_average3;dev=$COSINE_AVG_DIR/predictions-dev.jsonl;run=$COSINE_DIR;config=$COSINE_CONFIG;checkpoint=$COSINE_AVG_DIR/checkpoint.pt;priority=61" \
    --candidate "fdxmp_gaussian_average3;dev=$GAUSSIAN_AVG_DIR/predictions-dev.jsonl;run=$GAUSSIAN_DIR;config=$GAUSSIAN_CONFIG;checkpoint=$GAUSSIAN_AVG_DIR/checkpoint.pt;priority=61" \
    --candidate "fdxmp_h1_average3;dev=$H1_AVG_DIR/predictions-dev.jsonl;run=$H1_DIR;config=./configs/fdxmp_h1_split.yaml;checkpoint=$H1_AVG_DIR/checkpoint.pt;priority=31" \
    --candidate "fdxmp_h2_average3;dev=$H2_AVG_DIR/predictions-dev.jsonl;run=$H2_DIR;config=$H2_CONFIG;checkpoint=$H2_AVG_DIR/checkpoint.pt;priority=1" \
    --output "$LOCK_JSON" \
    --shell-output "$LOCK_ENV" \
    >"$STATUS_DIR/candidate_lock.console.log" 2>&1
exit_code=$?
if [[ $exit_code -ne 0 ]]; then
    log "FAILED Dev-only candidate lock exit_code=$exit_code"
    exit "$exit_code"
fi
# shellcheck disable=SC1090
source "$LOCK_ENV"
log "LOCKED candidate $LOCKED_NAME Dev WER=$LOCKED_DEV_WER"

if [[ -f "$TEST_ACCESS_CLAIM" ]]; then
    log "FAILED Test access was already claimed; refusing a second pass"
    exit 30
fi
cp "$LOCK_JSON" "$TEST_ACCESS_CLAIM"
BASELINE_TEST_DIR="$LOCKED_TEST_ROOT/baseline"
VAC_TEST_DIR="$LOCKED_TEST_ROOT/vac_ctc"
VAC_SMKD_TEST_DIR="$LOCKED_TEST_ROOT/vac_smkd"
VAC_SMKD_NGRAM_TEST_DIR="$LOCKED_TEST_ROOT/vac_smkd_ngram"
run_test_evaluation \
    "baseline" \
    "./configs/baseline_split.yaml" \
    "$BASELINE_CHECKPOINT" \
    "$BASELINE_TEST_DIR" || exit $?
run_test_evaluation \
    "vac_ctc" \
    "./configs/vac_ctc_split.yaml" \
    "$VAC_CHECKPOINT" \
    "$VAC_TEST_DIR" || exit $?
run_test_evaluation \
    "vac_smkd" \
    "./configs/vac_smkd_split.yaml" \
    "$VAC_SMKD_CHECKPOINT" \
    "$VAC_SMKD_TEST_DIR" || exit $?
if [[ -n "$VAC_SMKD_NGRAM_CONFIG" ]]; then
    run_test_evaluation \
        "vac_smkd_ngram" \
        "$VAC_SMKD_NGRAM_CONFIG" \
        "$VAC_SMKD_NGRAM_CHECKPOINT" \
        "$VAC_SMKD_NGRAM_TEST_DIR" || exit $?
fi

if [[ "$LOCKED_NAME" == "baseline" ]]; then
    SELECTED_TEST_DIR="$BASELINE_TEST_DIR"
elif [[ "$LOCKED_NAME" == "baseline_average3" ]]; then
    SELECTED_TEST_DIR="$LOCKED_TEST_ROOT/$LOCKED_NAME"
    run_test_evaluation \
        "$LOCKED_NAME" \
        "$LOCKED_CONFIG" \
        "$LOCKED_CHECKPOINT" \
        "$SELECTED_TEST_DIR" || exit $?
elif [[ "$LOCKED_NAME" == "vac_ctc" ]]; then
    SELECTED_TEST_DIR="$VAC_TEST_DIR"
elif [[ "$LOCKED_NAME" == "vac_ctc_average3" ]]; then
    SELECTED_TEST_DIR="$LOCKED_TEST_ROOT/$LOCKED_NAME"
    run_test_evaluation \
        "$LOCKED_NAME" \
        "$LOCKED_CONFIG" \
        "$LOCKED_CHECKPOINT" \
        "$SELECTED_TEST_DIR" || exit $?
elif [[ "$LOCKED_NAME" == "vac_smkd" ]]; then
    SELECTED_TEST_DIR="$VAC_SMKD_TEST_DIR"
elif [[ "$LOCKED_NAME" == "vac_smkd_average3" ]]; then
    SELECTED_TEST_DIR="$LOCKED_TEST_ROOT/$LOCKED_NAME"
    run_test_evaluation \
        "$LOCKED_NAME" \
        "$LOCKED_CONFIG" \
        "$LOCKED_CHECKPOINT" \
        "$SELECTED_TEST_DIR" || exit $?
elif [[ "$LOCKED_NAME" == "vac_smkd_ngram" ]]; then
    SELECTED_TEST_DIR="$VAC_SMKD_NGRAM_TEST_DIR"
else
    SELECTED_TEST_DIR="$LOCKED_TEST_ROOT/$LOCKED_NAME"
    run_test_evaluation \
        "$LOCKED_NAME" \
        "$LOCKED_CONFIG" \
        "$LOCKED_CHECKPOINT" \
        "$SELECTED_TEST_DIR" || exit $?
fi

SUMMARY_ARGS=(
    --reference baseline
    --prediction "baseline=$BASELINE_TEST_DIR/predictions-test.jsonl"
    --prediction "vac_ctc=$VAC_TEST_DIR/predictions-test.jsonl"
    --prediction "vac_smkd=$VAC_SMKD_TEST_DIR/predictions-test.jsonl"
    --bootstrap 5000
    --seed 0
    --output "$SUMMARY_PATH"
)
if [[ -n "$VAC_SMKD_NGRAM_CONFIG" ]]; then
    SUMMARY_ARGS+=(
        --prediction "vac_smkd_ngram=$VAC_SMKD_NGRAM_TEST_DIR/predictions-test.jsonl"
    )
fi
if [[ "$LOCKED_NAME" != "baseline" \
    && "$LOCKED_NAME" != "vac_ctc" \
    && "$LOCKED_NAME" != "vac_smkd" \
    && "$LOCKED_NAME" != "vac_smkd_ngram" ]]; then
    SUMMARY_ARGS+=(
        --prediction "$LOCKED_NAME=$SELECTED_TEST_DIR/predictions-test.jsonl"
    )
fi
log "START locked Test comparison"
"$PYTHON" scripts/summarize_predictions.py \
    "${SUMMARY_ARGS[@]}" \
    >"$STATUS_DIR/final_comparison.console.log" 2>&1
exit_code=$?
if [[ $exit_code -ne 0 ]]; then
    log "FAILED locked Test comparison exit_code=$exit_code"
    exit "$exit_code"
fi
log "COMPLETE locked Test comparison"

log "TEST_LOCKED_COMPLETE"
log "CANDIDATE_SUITE_COMPLETE"
