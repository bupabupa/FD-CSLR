#!/usr/bin/env bash
set -u

ROOT=.
PYTHON=python3
STATUS_DIR="$ROOT/work_dir/fd_xmp_v3/candidate_protocol"
STATUS_FILE="$STATUS_DIR/fallback_status.txt"
LOCK_JSON="$STATUS_DIR/locked_candidate.json"
LOCK_ENV="$STATUS_DIR/locked_candidate.env"
TEST_ACCESS_CLAIM="$STATUS_DIR/test_access_claim.json"
LOCKED_TEST_ROOT="$STATUS_DIR/locked_test"
SUMMARY_PATH="$STATUS_DIR/final_comparison.json"

mkdir -p "$STATUS_DIR"
exec >>"$STATUS_DIR/fallback_protocol.log" 2>&1

log() {
    printf '[%s] %s\n' "$(date '+%F %T %Z')" "$1" | tee "$STATUS_FILE"
}

fail() {
    log "FAILED $1"
    exit 1
}

run_test_evaluation() {
    local name="$1"
    local config="$2"
    local checkpoint="$3"
    local output_dir="$4"
    log "START locked Test $name"
    "$PYTHON" -u main.py \
        --device 0 \
        --phase test \
        --work-dir "$output_dir/" \
        --config "$config" \
        --load-weights "$checkpoint" \
        --eval-splits test \
        --evaluate-tool sclite \
        >"$output_dir.console.log" 2>&1
    if [[ $? -ne 0 ]]; then
        fail "locked Test $name"
    fi
    log "COMPLETE locked Test $name"
}

cd "$ROOT" || fail "cannot enter project root"

if ! grep -q 'WEIGHTED_ENSEMBLE_PROTOCOL_COMPLETE' \
    "$ROOT/work_dir/fd_xmp_v3/weighted_ensemble_protocol/status.txt"; then
    fail "weighted ensemble protocol is incomplete"
fi
if [[ -e "$LOCK_JSON" || -e "$LOCK_ENV" || -e "$TEST_ACCESS_CLAIM" ]]; then
    fail "a candidate lock or Test claim already exists"
fi
if [[ -e "$SUMMARY_PATH" ]]; then
    fail "a locked Test comparison already exists"
fi
if find "$LOCKED_TEST_ROOT" -type f -name 'predictions-test.jsonl' \
    -print -quit 2>/dev/null | grep -q .; then
    fail "Test predictions already exist"
fi

BASELINE_DIR="$ROOT/work_dir/fd_xmp_v3/E0_baseline_split_seed0"
VAC_DIR="$ROOT/work_dir/fd_xmp_v3/E0_vac_ctc_split_seed0"
VAC_SMKD_DIR="$ROOT/work_dir/fd_xmp_v3/E0_vac_smkd_split_seed0"
COSINE_DIR="$ROOT/work_dir/fd_xmp_v3/E1_fdxmp_v3_split_seed0"
GAUSSIAN_DIR="$ROOT/work_dir/fd_xmp_v3/E3_fdxmp_gaussian_split_seed0"
H1_DIR="$ROOT/work_dir/fd_xmp_v3/E2_fdxmp_h1_split_seed0"
H2_DIR="$ROOT/work_dir/fd_xmp_v3/E4_fdxmp_h2_split_seed0"
H2_FINAL_DIR="$H2_DIR/final"

best_checkpoint() {
    find "$1" -maxdepth 1 -name 'dev_*_epoch*_model.pt' -printf '%f\n' |
        sort -t_ -k2,2n |
        head -1
}

BASELINE_CHECKPOINT="$BASELINE_DIR/$(best_checkpoint "$BASELINE_DIR")"
VAC_CHECKPOINT="$VAC_DIR/$(best_checkpoint "$VAC_DIR")"
VAC_SMKD_CHECKPOINT="$VAC_SMKD_DIR/$(best_checkpoint "$VAC_SMKD_DIR")"
H2_CHECKPOINT="$H2_DIR/fdgc/checkpoint_calibrated.pt"
if [[ -f "$H2_DIR/multichannel/checkpoint_multichannel.pt" ]]; then
    H2_CONFIG="./configs/fdxmp_h2_calibrated.yaml"
else
    H2_CONFIG="./configs/fdxmp_h2_split.yaml"
fi

log "START fallback Dev-only candidate lock"
"$PYTHON" scripts/select_candidate.py \
    --candidate "baseline;dev=$BASELINE_DIR/final/predictions-dev.jsonl;run=$BASELINE_DIR;config=./configs/baseline_split.yaml;checkpoint=$BASELINE_CHECKPOINT;priority=100" \
    --candidate "vac_ctc;dev=$VAC_DIR/final/predictions-dev.jsonl;run=$VAC_DIR;config=./configs/vac_ctc_split.yaml;checkpoint=$VAC_CHECKPOINT;priority=90" \
    --candidate "vac_smkd;dev=$VAC_SMKD_DIR/final/predictions-dev.jsonl;run=$VAC_SMKD_DIR;config=./configs/vac_smkd_split.yaml;checkpoint=$VAC_SMKD_CHECKPOINT;priority=95" \
    --candidate "fdxmp_viterbi;dev=$COSINE_DIR/final/predictions-dev.jsonl;run=$COSINE_DIR;config=./configs/fdxmp_v3_split.yaml;checkpoint=$COSINE_DIR/fdgc/checkpoint_calibrated.pt;calibration=$COSINE_DIR/fdgc/fdgc_metrics.json;priority=60" \
    --candidate "fdxmp_fdgc_ctc;dev=$COSINE_DIR/fdgc_ctc/final/predictions-dev.jsonl;run=$COSINE_DIR;config=./configs/fdxmp_v3_split.yaml;checkpoint=$COSINE_DIR/fdgc_ctc/checkpoint_calibrated.pt;calibration=$COSINE_DIR/fdgc_ctc/fdgc_metrics.json;priority=50" \
    --candidate "fdxmp_fdgc_fd;dev=$COSINE_DIR/final_fdgc_fd/predictions-dev.jsonl;run=$COSINE_DIR;config=./configs/fdxmp_v3_split.yaml;checkpoint=$COSINE_DIR/fdgc_fd/checkpoint_calibrated.pt;calibration=$COSINE_DIR/fdgc_fd/fdgc_metrics.json;priority=40" \
    --candidate "gaussian_viterbi;dev=$GAUSSIAN_DIR/final_viterbi/predictions-dev.jsonl;run=$GAUSSIAN_DIR;config=./configs/fdxmp_gaussian_split.yaml;checkpoint=$GAUSSIAN_DIR/fdgc/checkpoint_calibrated.pt;calibration=$GAUSSIAN_DIR/fdgc/fdgc_metrics.json;priority=60" \
    --candidate "gaussian_fdgc_ctc;dev=$GAUSSIAN_DIR/fdgc_ctc/final/predictions-dev.jsonl;run=$GAUSSIAN_DIR;config=./configs/fdxmp_gaussian_split.yaml;checkpoint=$GAUSSIAN_DIR/fdgc_ctc/checkpoint_calibrated.pt;calibration=$GAUSSIAN_DIR/fdgc_ctc/fdgc_metrics.json;priority=50" \
    --candidate "fdxmp_h1;dev=$H1_DIR/final/predictions-dev.jsonl;run=$H1_DIR;config=./configs/fdxmp_h1_split.yaml;checkpoint=$H1_DIR/fdgc/checkpoint_calibrated.pt;calibration=$H1_DIR/fdgc/fdgc_metrics.json;priority=30" \
    --candidate "fdxmp_h2;dev=$H2_FINAL_DIR/predictions-dev.jsonl;run=$H2_DIR;config=$H2_CONFIG;checkpoint=$H2_CHECKPOINT;calibration=$H2_DIR/fdgc/fdgc_metrics.json;priority=0" \
    --candidate "baseline_average3;dev=$BASELINE_DIR/parity_average_last3/predictions-dev.jsonl;run=$BASELINE_DIR;config=./configs/baseline_split.yaml;checkpoint=$BASELINE_DIR/parity_average_last3/checkpoint.pt;priority=101" \
    --candidate "vac_ctc_average3;dev=$VAC_DIR/parity_average_last3/predictions-dev.jsonl;run=$VAC_DIR;config=./configs/vac_ctc_split.yaml;checkpoint=$VAC_DIR/parity_average_last3/checkpoint.pt;priority=91" \
    --candidate "vac_smkd_average3;dev=$VAC_SMKD_DIR/parity_average_last3/predictions-dev.jsonl;run=$VAC_SMKD_DIR;config=./configs/vac_smkd_split.yaml;checkpoint=$VAC_SMKD_DIR/parity_average_last3/checkpoint.pt;priority=96" \
    --candidate "fdxmp_cosine_average3;dev=$COSINE_DIR/parity_average_last3/predictions-dev.jsonl;run=$COSINE_DIR;config=./configs/fdxmp_v3_split.yaml;checkpoint=$COSINE_DIR/parity_average_last3/checkpoint.pt;priority=61" \
    --candidate "fdxmp_gaussian_average3;dev=$GAUSSIAN_DIR/parity_average_last3/predictions-dev.jsonl;run=$GAUSSIAN_DIR;config=./configs/fdxmp_gaussian_split.yaml;checkpoint=$GAUSSIAN_DIR/parity_average_last3/checkpoint.pt;priority=61" \
    --candidate "fdxmp_h1_average3;dev=$H1_DIR/parity_average_last3/predictions-dev.jsonl;run=$H1_DIR;config=./configs/fdxmp_h1_split.yaml;checkpoint=$H1_DIR/parity_average_last3/checkpoint.pt;priority=31" \
    --candidate "fdxmp_h2_average3;dev=$H2_DIR/parity_average_last3/predictions-dev.jsonl;run=$H2_DIR;config=./configs/fdxmp_h2_split.yaml;checkpoint=$H2_DIR/parity_average_last3/checkpoint.pt;priority=1" \
    --output "$LOCK_JSON" \
    --shell-output "$LOCK_ENV" \
    >"$STATUS_DIR/fallback_candidate_lock.console.log" 2>&1
if [[ $? -ne 0 ]]; then
    fail "fallback Dev-only candidate lock"
fi
# shellcheck disable=SC1090
source "$LOCK_ENV"
log "LOCKED candidate $LOCKED_NAME Dev WER=$LOCKED_DEV_WER"

cp "$LOCK_JSON" "$TEST_ACCESS_CLAIM"
BASELINE_TEST_DIR="$LOCKED_TEST_ROOT/baseline"
VAC_TEST_DIR="$LOCKED_TEST_ROOT/vac_ctc"
VAC_SMKD_TEST_DIR="$LOCKED_TEST_ROOT/vac_smkd"
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

if [[ "$LOCKED_NAME" == "baseline" ]]; then
    SELECTED_TEST_DIR="$BASELINE_TEST_DIR"
elif [[ "$LOCKED_NAME" == "vac_ctc" ]]; then
    SELECTED_TEST_DIR="$VAC_TEST_DIR"
elif [[ "$LOCKED_NAME" == "vac_smkd" ]]; then
    SELECTED_TEST_DIR="$VAC_SMKD_TEST_DIR"
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
if [[ "$LOCKED_NAME" != "baseline" \
    && "$LOCKED_NAME" != "vac_ctc" \
    && "$LOCKED_NAME" != "vac_smkd" ]]; then
    SUMMARY_ARGS+=(
        --prediction "$LOCKED_NAME=$SELECTED_TEST_DIR/predictions-test.jsonl"
    )
fi
log "START locked Test comparison"
"$PYTHON" scripts/summarize_predictions.py \
    "${SUMMARY_ARGS[@]}" \
    >"$STATUS_DIR/fallback_final_comparison.console.log" 2>&1
if [[ $? -ne 0 ]]; then
    fail "locked Test comparison"
fi

log "TEST_LOCKED_COMPLETE"
log "FALLBACK_CANDIDATE_SUITE_COMPLETE"
