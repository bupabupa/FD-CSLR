import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.check_protocol_manifest import validate_manifest


def build_training_command(job, python, device):
    return [
        str(python),
        "-u",
        "main.py",
        "--phase",
        "train",
        "--device",
        str(device),
        "--work-dir",
        job["work_dir"] + "/",
        "--config",
        job["config"],
        "--random-seed",
        str(job["seed"]),
        "--num-epoch",
        "40",
        "--save-interval",
        "5",
        "--evaluate-tool",
        "sclite",
        "--checkpoint-split",
        "cal",
        "--load-splits",
        "train",
        "train_eval",
        "cal",
    ]


def verify_saved_seed(work_dir, expected_seed):
    import yaml

    config_path = Path(work_dir) / "config.yaml"
    if not config_path.is_file():
        raise RuntimeError(
            "training did not save a resolved config: {}".format(config_path)
        )
    with open(config_path, encoding="utf-8") as source:
        saved = yaml.safe_load(source)
    if int(saved.get("random_seed", -1)) != int(expected_seed):
        raise RuntimeError(
            "resolved config seed {} does not match expected {}".format(
                saved.get("random_seed"),
                expected_seed,
            )
        )
    if saved.get("decode_mode") != "beam":
        raise RuntimeError("resolved config does not use ctcdecode beam search")
    if saved.get("evaluate_tool") != "sclite":
        raise RuntimeError("resolved config does not use sclite")
    if saved.get("checkpoint_split") != "cal":
        raise RuntimeError("resolved config does not select checkpoints on D_cal")


def run_plan(plan, python, device, execute=False):
    rendered = []
    for job in plan["jobs"]:
        command = build_training_command(job, python, device)
        rendered.append({
            "name": job["name"],
            "seed": job["seed"],
            "work_dir": job["work_dir"],
            "command": command,
        })
    if not execute:
        return rendered

    root = Path(plan["manifest"]).resolve().parents[1]
    for item in rendered:
        work_dir = Path(item["work_dir"])
        work_dir.mkdir(parents=True, exist_ok=True)
        console_path = work_dir / "console.log"
        with open(console_path, "w", encoding="utf-8") as console:
            completed = subprocess.run(
                item["command"],
                cwd=str(root),
                stdout=console,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode != 0:
            raise RuntimeError(
                "{} failed with exit code {}".format(
                    item["name"],
                    completed.returncode,
                )
            )
        verify_saved_seed(work_dir, item["seed"])
        provenance = {
            "name": item["name"],
            "seed": item["seed"],
            "command": item["command"],
            "resolved_config": str(work_dir / "config.yaml"),
        }
        with open(
            work_dir / "seed_provenance.json",
            "w",
            encoding="utf-8",
        ) as output:
            json.dump(provenance, output, indent=2, sort_keys=True)
            output.write("\n")
    return rendered


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="protocols/a_protocol_manifest.json",
    )
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument(
        "--python",
        default="python3",
    )
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="run the nine training jobs; without this flag print the plan",
    )
    args = parser.parse_args()
    plan = validate_manifest(
        args.manifest,
        root=Path(args.root),
        require_runtime=True,
    )
    rendered = run_plan(
        plan,
        python=args.python,
        device=args.device,
        execute=args.execute,
    )
    print(json.dumps(rendered, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
