import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(
    os.environ.get(
        "CANDIDATE_SUITE_SCRIPT",
        ROOT / "scripts" / "run_candidate_suite.sh",
    )
)


def test_candidate_suite_contract():
    source = SCRIPT.read_text(encoding="utf-8")
    required_fragments = (
        'OPTIONAL_PATH="$STATUS_DIR/optional_candidates.json"',
        'required = {"vac_smkd"}',
        'if "no_eligible_candidate" not in ngram_status:',
        'required.add("fdxmp_ngram_calibrated")',
        'if "no_eligible_candidate" not in vac_ngram_status:',
        'required.add("vac_smkd_ngram")',
        'VAC_SMKD_RUN_DIR="$ROOT/work_dir/fd_xmp_v3/'
        'E0_vac_smkd_split_seed0"',
        'VAC_SMKD_TEST_DIR="$LOCKED_TEST_ROOT/vac_smkd"',
        'VAC_SMKD_NGRAM_TEST_DIR="$LOCKED_TEST_ROOT/vac_smkd_ngram"',
        '--prediction "vac_smkd=$VAC_SMKD_TEST_DIR/'
        'predictions-test.jsonl"',
        '--prediction "vac_smkd_ngram=$VAC_SMKD_NGRAM_TEST_DIR/'
        'predictions-test.jsonl"',
        'if [[ -e "$LOCK_JSON" || -e "$LOCK_ENV" ]]; then',
        'if [[ -e "$TEST_ACCESS_CLAIM" ]]; then',
        'if [[ -e "$SUMMARY_PATH" ]]; then',
        'find "$LOCKED_TEST_ROOT" -type f -name '
        "'predictions-test.jsonl'",
        'H2 multichannel calibration selected on D_cal',
        'H2 multichannel calibration did not beat native fusion on D_cal',
        "AVERAGE_LAST_N=3",
        'scripts/average_checkpoints.py',
        '--last-n "$AVERAGE_LAST_N"',
        'run_average_candidate \\\n    "baseline"',
        'run_average_candidate \\\n    "vac_ctc"',
        'run_average_candidate \\\n    "vac_smkd"',
        'run_average_candidate \\\n    "fdxmp_h2"',
        '--candidate "baseline_average3;dev=$BASELINE_AVG_DIR/',
        '--candidate "vac_ctc_average3;dev=$VAC_AVG_DIR/',
        '--candidate "vac_smkd_average3;dev=$VAC_SMKD_AVG_DIR/',
        '--candidate "fdxmp_cosine_average3;dev=$COSINE_AVG_DIR/',
        '--candidate "fdxmp_gaussian_average3;dev=$GAUSSIAN_AVG_DIR/',
        '--candidate "fdxmp_h1_average3;dev=$H1_AVG_DIR/',
        '--candidate "fdxmp_h2_average3;dev=$H2_AVG_DIR/',
    )
    for fragment in required_fragments:
        assert fragment in source, fragment

    vac_protocol = (
        ROOT / "scripts" / "run_vac_smkd_protocol.sh"
    ).read_text(encoding="utf-8")
    assert "--candidate-name vac_smkd_ngram" in vac_protocol


if __name__ == "__main__":
    test_candidate_suite_contract()
    print("candidate suite contract tests passed")
