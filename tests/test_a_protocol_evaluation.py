import json
from pathlib import Path
import sys
import tempfile

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.check_protocol_manifest import validate_manifest
from scripts.run_a_protocol import (
    build_evaluation_commands,
    build_lock_command,
    discover_best_cal_checkpoint,
)


MANIFEST = ROOT / "protocols" / "a_protocol_manifest.json"


def test_discovers_lowest_wer_d_cal_checkpoint_only():
    with tempfile.TemporaryDirectory() as directory:
        run_dir = Path(directory)
        (run_dir / "dev_1.00_epoch9_model.pt").write_bytes(b"ignored")
        (run_dir / "cal_24.00_epoch4_model.pt").write_bytes(b"checkpoint")
        selected = run_dir / "cal_22.50_epoch7_model.pt"
        selected.write_bytes(b"checkpoint")
        (run_dir / "cal_22.50_epoch8_model.pt").write_bytes(b"checkpoint")
        record = discover_best_cal_checkpoint(run_dir)
    assert record["path"].name == selected.name
    assert record["cal_wer"] == 22.5
    assert record["epoch"] == 7


def test_evaluation_plan_uses_d_cal_selection_and_d_sel_once():
    report = validate_manifest(MANIFEST, root=ROOT, require_runtime=True)
    checkpoints = {
        job["name"]: {
            "path": Path(job["work_dir"])
            / "cal_20.00_epoch1_model.pt",
            "cal_wer": 20.0,
            "epoch": 1,
        }
        for job in report["jobs"]
    }
    with tempfile.TemporaryDirectory() as directory:
        commands, outputs = build_evaluation_commands(
            report,
            checkpoints,
            output_root=Path(directory),
            python="/env/bin/python",
        )
    assert len(commands) == 17
    assert len(outputs["predictions"]) == 12
    assert outputs["summary"].endswith("a_protocol_summary.json")
    for item in commands:
        command = item["command"]
        if "--split" in command:
            split = command[command.index("--split") + 1]
            assert split == "dev"
        assert "test" not in command
    prior_commands = [
        item for item in commands if item["name"] == "build_train_prior"
    ]
    assert len(prior_commands) == 1
    selector_commands = [
        item for item in commands if item["name"].startswith("select_")
    ]
    assert len(selector_commands) == 3
    summary = next(
        item for item in commands if item["name"] == "summarize_selection"
    )["command"]
    assert summary.count("--prediction") == 12
    assert summary[summary.index("--bootstrap") + 1] == "5000"

    with open(MANIFEST, encoding="utf-8") as source:
        manifest = json.load(source)
    assert manifest["checkpoint_selection_split"] == "cal"
    assert manifest["selection_split"] == "sel"


def test_lock_plan_uses_twelve_d_sel_candidates_without_test():
    report = validate_manifest(MANIFEST, root=ROOT, require_runtime=True)
    checkpoints = {
        job["name"]: {
            "path": Path(job["work_dir"])
            / "cal_20.00_epoch1_model.pt",
            "cal_wer": 20.0,
            "epoch": 1,
        }
        for job in report["jobs"]
    }
    with tempfile.TemporaryDirectory() as directory:
        command, outputs = build_lock_command(
            report,
            checkpoints,
            output_root=Path(directory),
            python="/env/bin/python",
        )
    assert command["command"].count("--candidate") == 12
    assert "test" not in command["command"]
    assert command["command"][command["command"].index("--protocol") + 1] == "A"
    assert "--split-manifest" in command["command"]
    assert outputs["lock"].endswith("candidate_lock/locked_candidate.json")
    assert outputs["env"].endswith("candidate_lock/locked_candidate.env")


if __name__ == "__main__":
    test_discovers_lowest_wer_d_cal_checkpoint_only()
    test_evaluation_plan_uses_d_cal_selection_and_d_sel_once()
    test_lock_plan_uses_twelve_d_sel_candidates_without_test()
    print("A protocol evaluation tests passed")
