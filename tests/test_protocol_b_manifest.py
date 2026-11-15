import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.check_protocol_manifest import expand_manifest_jobs
from scripts.materialize_protocol_configs import _render_overrides
from scripts.run_a_training import build_training_command


MANIFEST = ROOT / "protocols" / "b_protocol_manifest.json"


def _manifest():
    with open(MANIFEST, encoding="utf-8") as source:
        return json.load(source)


def test_protocol_b_expands_five_seeds_and_two_backbone_tables():
    manifest = _manifest()
    jobs = expand_manifest_jobs(manifest)
    assert manifest["protocol"] == "B"
    assert manifest["seeds"] == [0, 1, 2, 3, 4]
    assert len(jobs) == 65
    assert {job["backbone"] for job in jobs} == {"resnet18", "resnet34"}
    assert {
        job["seed"] for job in jobs
    } == {0, 1, 2, 3, 4}
    assert len({job["name"] for job in jobs}) == len(jobs)
    assert all(
        job["comparison_group"] == job["backbone"]
        for job in jobs
    )


def test_protocol_b_commands_never_load_or_select_on_test():
    command = build_training_command(
        expand_manifest_jobs(_manifest())[0],
        python="python",
        device="0",
    )
    split_index = command.index("--load-splits")
    assert command[split_index + 1:] == ["train", "train_eval", "cal"]
    assert command[command.index("--checkpoint-split") + 1] == "cal"
    assert command[command.index("--evaluate-tool") + 1] == "sclite"
    assert "test" not in command[split_index + 1:]


def test_all_ablation_roles_are_present():
    jobs = expand_manifest_jobs(_manifest())
    roles = {job["role"] for job in jobs}
    assert roles == {
        "baseline",
        "vac_ctc",
        "weighted_ctc",
        "pra",
        "weighted_pra",
        "fdgc",
        "no_signer",
        "real_signer",
        "pseudo_signer",
    }


def test_pseudo_signer_maps_are_seed_specific():
    pseudo_jobs = [
        job
        for job in expand_manifest_jobs(_manifest())
        if job["role"] == "pseudo_signer"
    ]
    templates = {
        job["overrides"]["feeder_args"]["signer_map_path"]
        for job in pseudo_jobs
    }
    assert templates == {
        "preprocess/phoenix2014/pseudo_signers/seed{seed}/signer_map.json"
    }
    rendered = {
        _render_overrides(job["overrides"], job)[
            "feeder_args"
        ]["signer_map_path"]
        for job in pseudo_jobs
    }
    assert len(rendered) == 5
    assert all(
        any(
            "seed{}/signer_map.json".format(job["seed"]) in value
            for value in rendered
        )
        for job in pseudo_jobs
    )


if __name__ == "__main__":
    test_protocol_b_expands_five_seeds_and_two_backbone_tables()
    test_protocol_b_commands_never_load_or_select_on_test()
    test_all_ablation_roles_are_present()
    test_pseudo_signer_maps_are_seed_specific()
    print("protocol B manifest tests passed")
