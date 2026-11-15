import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.check_protocol_manifest import validate_manifest
from scripts.run_a_training import build_training_command


MANIFEST = ROOT / "protocols" / "a_protocol_manifest.json"


def test_training_commands_apply_job_seeds_and_exclude_test():
    report = validate_manifest(MANIFEST, root=ROOT, require_runtime=True)
    commands = {}
    for job in report["jobs"]:
        command = build_training_command(job, sys.executable, "0")
        commands[job["name"]] = command
        seed_index = command.index("--random-seed") + 1
        assert int(command[seed_index]) == job["seed"]
        split_index = command.index("--load-splits")
        assert command[split_index + 1:] == ["train", "train_eval", "cal"]
        assert "test" not in command[split_index + 1:]
        assert command[command.index("--evaluate-tool") + 1] == "sclite"
        assert command[command.index("--checkpoint-split") + 1] == "cal"
        assert command[command.index("--phase") + 1] == "train"

    assert len(commands) == 9
    assert len({
        tuple(command) for command in commands.values()
    }) == 9


def test_materialized_configs_are_seed_bound():
    with open(MANIFEST, encoding="utf-8") as source:
        manifest = json.load(source)
    for job in manifest["jobs"]:
        config = ROOT / job["config"]
        assert config.is_file()
        assert config.parent.name == "protocol_a"
        assert config.name.endswith("seed{}.yaml".format(job["seed"]))


if __name__ == "__main__":
    test_training_commands_apply_job_seeds_and_exclude_test()
    test_materialized_configs_are_seed_bound()
    print("protocol training runner tests passed")
