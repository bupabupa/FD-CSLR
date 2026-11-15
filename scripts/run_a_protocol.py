import argparse
import json
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.check_protocol_manifest import validate_manifest
from scripts.run_a_training import (
    run_plan as run_training_plan,
)


ROLES = ("baseline", "vac_ctc", "vac_smkd")
CHECKPOINT_PATTERN = re.compile(
    r"^cal_([0-9]+(?:\.[0-9]+)?)_epoch([0-9]+)_model\.pt$"
)


def _resolve(root, value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def discover_best_cal_checkpoint(run_dir):
    run_dir = Path(run_dir)
    candidates = []
    for path in run_dir.glob("cal_*_epoch*_model.pt"):
        match = CHECKPOINT_PATTERN.match(path.name)
        if match is None:
            continue
        candidates.append({
            "path": path.resolve(),
            "cal_wer": float(match.group(1)),
            "epoch": int(match.group(2)),
        })
    if not candidates:
        raise FileNotFoundError(
            "no D_cal-selected checkpoint found in {}".format(run_dir)
        )
    return min(
        candidates,
        key=lambda item: (
            item["cal_wer"],
            item["epoch"],
            str(item["path"]),
        ),
    )


def member_spec(job, checkpoint):
    return (
        "name={};config={};checkpoint={};weight=1.0;temperature=1.0"
    ).format(
        job["name"],
        job["config"],
        checkpoint,
    )


def _append_members(command, jobs, checkpoints):
    for job in jobs:
        command.extend([
            "--member",
            member_spec(job, checkpoints[job["name"]]["path"]),
        ])


def build_evaluation_commands(
    report,
    checkpoints,
    output_root,
    python,
    device="0",
    batch_size=2,
    num_workers=4,
    num_processes=10,
    beam_width=10,
    bootstrap=5000,
    seed=0,
):
    root = Path(report["manifest"]).resolve().parents[1]
    output_root = Path(output_root).resolve()
    python = str(python)
    split_manifest = _resolve(root, report["split_manifest"])
    preprocess_dir = split_manifest.parent
    prior_path = output_root / "calibration" / "train_prior.npy"
    jobs = report["jobs"]
    selected_jobs = {
        role: [job for job in jobs if job["role"] == role]
        for role in ROLES
    }
    commands = []

    commands.append({
        "name": "build_train_prior",
        "stage": "calibration",
        "command": [
            python,
            "scripts/build_train_prior.py",
            "--preprocess-dir",
            str(preprocess_dir),
            "--split-manifest",
            str(split_manifest),
            "--output",
            str(prior_path),
        ],
    })

    prediction_paths = {}
    for job in jobs:
        output_dir = output_root / "selection" / "individual" / job["name"]
        prediction_path = output_dir / "predictions-dev.jsonl"
        prediction_paths[job["name"]] = prediction_path
        commands.append({
            "name": "evaluate_" + job["name"],
            "stage": "selection",
            "prediction": str(prediction_path),
            "command": [
                python,
                "scripts/evaluate_logit_ensemble.py",
                "--member",
                member_spec(job, checkpoints[job["name"]]["path"]),
                "--split",
                "dev",
                "--split-manifest",
                str(split_manifest),
                "--output-dir",
                str(output_dir),
                "--device",
                str(device),
                "--batch-size",
                str(batch_size),
                "--num-workers",
                str(num_workers),
                "--num-processes",
                str(num_processes),
                "--beam-width",
                str(beam_width),
                "--evaluate-tool",
                "sclite",
                "--class-prior-path",
                str(prior_path),
                "--prior-tau",
                "0.0",
            ],
        })

    for role in ROLES:
        role_jobs = selected_jobs[role]
        role_root = output_root / "selection" / role
        spec_path = role_root / "ensemble-spec.json"
        cal_summary = role_root / "cal-selection.json"
        selector = [
            python,
            "scripts/select_weighted_ensemble.py",
        ]
        _append_members(selector, role_jobs, checkpoints)
        selector.extend([
            "--split-manifest",
            str(split_manifest),
            "--class-prior-path",
            str(prior_path),
            "--output-spec",
            str(spec_path),
            "--output-summary",
            str(cal_summary),
            "--device",
            str(device),
            "--batch-size",
            str(batch_size),
            "--num-workers",
            str(num_workers),
            "--num-processes",
            str(num_processes),
            "--beam-width",
            str(beam_width),
            "--max-members",
            "3",
            "--refinement-subsets",
            "2",
            "--refinement-samples",
            "30",
            "--decode-candidates",
            "10",
            "--seed",
            str(seed),
            "--cache-dir",
            str(role_root / "cal-cache"),
        ])
        commands.append({
            "name": "select_" + role,
            "stage": "calibration",
            "command": selector,
        })

        prediction_path = role_root / "dev" / "predictions-dev.jsonl"
        prediction_paths[role] = prediction_path
        commands.append({
            "name": "evaluate_" + role,
            "stage": "selection",
            "prediction": str(prediction_path),
            "command": [
                python,
                "scripts/evaluate_logit_ensemble.py",
                "--ensemble-spec",
                str(spec_path),
                "--split",
                "dev",
                "--split-manifest",
                str(split_manifest),
                "--output-dir",
                str(role_root / "dev"),
                "--device",
                str(device),
                "--batch-size",
                str(batch_size),
                "--num-workers",
                str(num_workers),
                "--num-processes",
                str(num_processes),
                "--beam-width",
                str(beam_width),
                "--evaluate-tool",
                "sclite",
            ],
        })

    summary_path = output_root / "selection" / "a_protocol_summary.json"
    summary = [
        python,
        "scripts/summarize_predictions.py",
    ]
    for name in sorted(prediction_paths):
        summary.extend([
            "--prediction",
            "{}={}".format(name, prediction_paths[name]),
        ])
    summary.extend([
        "--reference",
        "baseline_seed0",
        "--num-classes",
        str(report["num_classes"]),
        "--bootstrap",
        str(bootstrap),
        "--seed",
        str(seed),
        "--output",
        str(summary_path),
    ])
    commands.append({
        "name": "summarize_selection",
        "stage": "summary",
        "prediction": str(summary_path),
        "command": summary,
    })
    return commands, {
        "prior": str(prior_path),
        "predictions": {
            name: str(path) for name, path in prediction_paths.items()
        },
        "summary": str(summary_path),
    }


def build_lock_command(
    report,
    checkpoints,
    output_root,
    python,
):
    root = Path(report["manifest"]).resolve().parents[1]
    output_root = Path(output_root).resolve()
    lock_dir = output_root / "candidate_lock"
    lock_path = lock_dir / "locked_candidate.json"
    env_path = lock_dir / "locked_candidate.env"
    jobs = report["jobs"]
    command = [
        str(python),
        "scripts/select_candidate.py",
        "--num-classes",
        str(report["num_classes"]),
        "--output",
        str(lock_path),
        "--shell-output",
        str(env_path),
        "--protocol",
        "A",
        "--split-manifest",
        report["split_manifest"],
    ]
    for job in jobs:
        prediction = (
            output_root
            / "selection"
            / "individual"
            / job["name"]
            / "predictions-dev.jsonl"
        )
        command.extend([
            "--candidate",
            (
                "name={};dev={};run={};config={};checkpoint={};priority=1"
            ).format(
                job["name"],
                prediction,
                _resolve(root, job["work_dir"]),
                _resolve(root, job["config"]),
                checkpoints[job["name"]]["path"],
            ),
        ])
    for role in ROLES:
        role_jobs = [job for job in jobs if job["role"] == role]
        role_root = output_root / "selection" / role
        command.extend([
            "--candidate",
            (
                "name={};dev={};run={};config={};checkpoint={};priority=10"
            ).format(
                role + "_ensemble",
                role_root / "dev" / "predictions-dev.jsonl",
                role_root,
                _resolve(root, role_jobs[0]["config"]),
                role_root / "ensemble-spec.json",
            ),
        ])
    return {
        "name": "lock_candidate",
        "stage": "lock",
        "command": command,
    }, {
        "lock": str(lock_path),
        "env": str(env_path),
    }


def execute_commands(commands, root, log_dir, expected_outputs):
    root = Path(root)
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    completed = []
    for item in commands:
        log_path = log_dir / (item["name"] + ".log")
        with open(log_path, "w", encoding="utf-8") as log:
            result = subprocess.run(
                item["command"],
                cwd=str(root),
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode != 0:
            raise RuntimeError(
                "{} failed with exit code {}; log: {}".format(
                    item["name"],
                    result.returncode,
                    log_path,
                )
            )
        completed.append({
            "name": item["name"],
            "log": str(log_path),
            "exit_code": result.returncode,
        })
    missing = [
        path for path in expected_outputs if not Path(path).is_file()
    ]
    if missing:
        raise RuntimeError(
            "protocol outputs are missing: {}".format(", ".join(missing))
        )
    return completed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="protocols/a_protocol_manifest.json",
    )
    parser.add_argument(
        "--stage",
        choices=["training", "evaluation", "lock"],
        default="training",
    )
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument(
        "--python",
        default="python3",
    )
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-processes", type=int, default=10)
    parser.add_argument("--beam-width", type=int, default=10)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="execute the selected stage; otherwise print the plan",
    )
    args = parser.parse_args()
    root = Path(args.root).resolve()
    report = validate_manifest(
        args.manifest,
        root=root,
        require_runtime=True,
    )

    if args.stage == "training":
        rendered = run_training_plan(
            report,
            python=args.python,
            device=args.device,
            execute=args.execute,
        )
        print(json.dumps(rendered, indent=2, ensure_ascii=True))
        return

    checkpoints = {
        job["name"]: discover_best_cal_checkpoint(job["work_dir"])
        for job in report["jobs"]
    }
    output_root = root / "work_dir" / "fd_xmp_v3" / "A_protocol"
    if args.stage == "lock":
        command, outputs = build_lock_command(
            report,
            checkpoints,
            output_root,
            python=args.python,
        )
        lock_path = Path(outputs["lock"])
        if args.execute and lock_path.exists():
            raise RuntimeError(
                "candidate lock already exists: {}".format(lock_path)
            )
        payload = {
            "protocol": "A",
            "split_sha256": report["split_sha256"],
            "commands": [command],
            "expected_outputs": outputs,
        }
        if args.execute:
            payload["completed"] = execute_commands(
                [command],
                root=root,
                log_dir=output_root / "candidate_lock" / "logs",
                expected_outputs=[outputs["lock"], outputs["env"]],
            )
            with open(lock_path, encoding="utf-8") as source:
                payload["selected"] = json.load(source)["selected"]
        print(json.dumps(payload, indent=2, ensure_ascii=True))
        return

    commands, outputs = build_evaluation_commands(
        report,
        checkpoints,
        output_root,
        python=args.python,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_processes=args.num_processes,
        beam_width=args.beam_width,
        bootstrap=args.bootstrap,
        seed=args.seed,
    )
    payload = {
        "protocol": "A",
        "split_sha256": report["split_sha256"],
        "checkpoints": {
            name: {
                **record,
                "path": str(record["path"]),
            }
            for name, record in checkpoints.items()
        },
        "commands": commands,
        "expected_outputs": outputs,
    }
    plan_path = output_root / "evaluation_plan.json"
    if args.execute:
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        with open(plan_path, "w", encoding="utf-8") as output:
            json.dump(payload, output, indent=2, sort_keys=True)
            output.write("\n")
        expected = list(outputs["predictions"].values())
        expected.extend([outputs["prior"], outputs["summary"]])
        payload["completed"] = execute_commands(
            commands,
            root=root,
            log_dir=output_root / "logs",
            expected_outputs=expected,
        )
    print(json.dumps(payload, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
