import argparse
from collections import defaultdict
import importlib.metadata
import json
import os
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset.splits import load_split_manifest
from utils import get_parser


EXPECTED_SPLIT_SHA256 = (
    "3488f9974636304f480e9e1ab7dd273f75d14ea461a8e769aa21dec72fdd7a91"
)
PROTOCOL_SEEDS = {
    "A": {0, 1, 2},
    "B": {0, 1, 2, 3, 4},
}
PROTOCOL_ROLES = {
    "A": {"baseline", "vac_ctc", "vac_smkd"},
    "B": {
        "baseline",
        "vac_ctc",
        "weighted_ctc",
        "pra",
        "weighted_pra",
        "fdgc",
        "no_signer",
        "real_signer",
        "pseudo_signer",
    },
}
TEST_ARTIFACTS = {
    "locked_candidate.json",
    "test_access_claim.json",
    "final_comparison.json",
    "predictions-test.jsonl",
}


def _resolve_path(root, value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _parse_config(root, config_path):
    parser = get_parser()
    with open(config_path, encoding="utf-8") as source:
        defaults = yaml.load(source, Loader=yaml.FullLoader)
    parser.set_defaults(**defaults)
    return parser.parse_args(["--config", str(config_path)])


def expand_manifest_jobs(manifest):
    protocol = manifest.get("protocol")
    if protocol == "A":
        jobs = manifest.get("jobs")
        if not isinstance(jobs, list) or not jobs:
            raise ValueError("manifest jobs must be a non-empty list")
        return list(jobs)
    if protocol != "B":
        raise ValueError("manifest protocol must be A or B")

    experiments = manifest.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("protocol B requires non-empty experiments")
    default_seeds = [int(value) for value in manifest.get("seeds", [])]
    jobs = []
    for experiment in experiments:
        if not isinstance(experiment, dict):
            raise ValueError("protocol B experiments must be objects")
        seeds = [
            int(value)
            for value in experiment.get("seeds", default_seeds)
        ]
        for seed in seeds:
            name_template = experiment.get(
                "name_template",
                str(experiment.get("name", "job")) + "_seed{seed}",
            )
            job = dict(experiment)
            job.update({
                "name": str(name_template).format(seed=seed),
                "seed": seed,
                "config": str(
                    experiment["config_template"]
                ).format(seed=seed),
                "work_dir": str(
                    experiment["work_dir_template"]
                ).format(seed=seed),
            })
            jobs.append(job)
    return jobs


def _validate_recipe(role, args):
    weights = {key: float(value) for key, value in args.loss_weights.items()}
    if role == "baseline":
        if weights != {"SeqCTC": 1.0}:
            raise ValueError("baseline must use SeqCTC=1.0 only")
        if args.model_args.get("share_classifier"):
            raise ValueError("baseline must not share the classifier")
        if args.model_args.get("weight_norm"):
            raise ValueError("baseline must not use weight normalization")
        return

    expected = {"SeqCTC": 1.0, "ConvCTC": 1.0}
    if weights != expected:
        raise ValueError(
            "{} must use SeqCTC=ConvCTC=1.0".format(role)
        )
    if role == "vac_ctc":
        if args.model_args.get("share_classifier"):
            raise ValueError("VAC-CTC must not share the classifier")
        if args.model_args.get("weight_norm"):
            raise ValueError("VAC-CTC must not use weight normalization")
    elif role == "vac_smkd":
        if not args.model_args.get("share_classifier"):
            raise ValueError("VAC+SMKD README recipe requires sharing")
        if not args.model_args.get("weight_norm"):
            raise ValueError("VAC+SMKD README recipe requires weight norm")


def _validate_protocol_b_recipe(role, args):
    weights = {key: float(value) for key, value in args.loss_weights.items()}
    model_args = args.model_args
    if role == "baseline":
        if weights != {"SeqCTC": 1.0}:
            raise ValueError("B baseline must use SeqCTC=1.0 only")
        return
    if role == "vac_ctc":
        if weights != {"SeqCTC": 1.0, "ConvCTC": 1.0}:
            raise ValueError("B VAC-CTC must use SeqCTC=ConvCTC=1.0")
        return
    if role in {"weighted_ctc", "weighted_pra"}:
        if weights.get("WeightedCTC", 0.0) <= 0.0:
            raise ValueError("{} requires WeightedCTC > 0".format(role))
        if not model_args.get("weighted_ctc_enabled"):
            raise ValueError("{} requires weighted_ctc_enabled".format(role))
    if role in {"pra", "weighted_pra"}:
        if weights.get("PRA", 0.0) <= 0.0:
            raise ValueError("{} requires PRA > 0".format(role))
        if not model_args.get("pra_enabled"):
            raise ValueError("{} requires pra_enabled".format(role))
    if role in {"no_signer", "fdgc"} and model_args.get("signer_film"):
        raise ValueError("{} must disable signer conditioning".format(role))
    if role in {"real_signer", "pseudo_signer"}:
        if not model_args.get("signer_film"):
            raise ValueError("{} requires signer_film".format(role))
    if role == "pseudo_signer":
        if not args.feeder_args.get("signer_map_path"):
            raise ValueError("pseudo_signer requires feeder signer_map_path")
    if role == "real_signer" and args.feeder_args.get("signer_map_path"):
        raise ValueError("real_signer must not use a pseudo signer map")


def _pseudo_signer_artifacts(root, job):
    signer_map = _resolve_path(
        root,
        job["args"].feeder_args["signer_map_path"],
    )
    artifact_dir = signer_map.parent
    return {
        "signer_map": signer_map,
        "cluster_report": artifact_dir / "cluster_report.json",
        "centers": artifact_dir / "centers.npy",
    }


def _find_test_artifacts(root):
    found = []
    search_roots = [root / "work_dir", root / "protocols"]
    for base in search_roots:
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if path.is_file() and path.name in TEST_ARTIFACTS:
                found.append(str(path))
    return sorted(found)


def runtime_status(root):
    sclite = root / "software" / "sclite"
    return {
        "python": sys.executable,
        "ctcdecode_version": importlib.metadata.version("ctcdecode"),
        "sclite_path": str(sclite.resolve()) if sclite.exists() else None,
        "sclite_executable": bool(
            sclite.exists() and os.access(str(sclite), os.X_OK)
        ),
    }


def validate_manifest(
    manifest_path,
    root=None,
    require_runtime=False,
    require_artifacts=False,
):
    root = (root or ROOT).resolve()
    manifest_path = _resolve_path(root, manifest_path)
    with open(manifest_path, encoding="utf-8") as source:
        manifest = json.load(source)

    protocol = manifest.get("protocol")
    if protocol not in PROTOCOL_SEEDS:
        raise ValueError("manifest protocol must be A or B")
    expected_seeds = PROTOCOL_SEEDS[protocol]
    expected_roles = PROTOCOL_ROLES[protocol]
    if manifest.get("split_sha256") != EXPECTED_SPLIT_SHA256:
        raise ValueError("manifest split hash is not the locked formal split")
    manifest_seeds = {int(value) for value in manifest.get("seeds", [])}
    if manifest_seeds != expected_seeds:
        raise ValueError(
            "protocol {} requires seeds {}".format(
                protocol,
                sorted(expected_seeds),
            )
        )
    if int(manifest.get("num_epochs", 0)) != 40:
        raise ValueError("protocol {} requires 40 epochs".format(protocol))
    if int(manifest.get("num_classes", 0)) != 1296:
        raise ValueError(
            "protocol {} requires 1296 output classes".format(protocol)
        )
    if manifest.get("checkpoint_selection_split") != "cal":
        raise ValueError(
            "protocol {} requires D_cal checkpoint selection".format(protocol)
        )
    if manifest.get("selection_split") != "sel":
        raise ValueError(
            "protocol {} requires D_sel candidate selection".format(protocol)
        )
    split_manifest_path = _resolve_path(root, manifest["split_manifest"])
    if not split_manifest_path.is_file():
        raise ValueError(
            "split manifest does not exist: {}".format(split_manifest_path)
        )
    protocol_split = load_split_manifest(
        str(split_manifest_path),
        str(split_manifest_path.parent),
    )
    if protocol_split["split_sha256"] != EXPECTED_SPLIT_SHA256:
        raise ValueError("manifest split hash is not the locked formal split")

    jobs = expand_manifest_jobs(manifest)

    names = set()
    work_dirs = set()
    role_seeds = defaultdict(set)
    observed_roles = set()
    observed_backbones = set()
    pseudo_signer_map_seeds = {}
    validated_jobs = []
    for job in jobs:
        name = str(job.get("name", ""))
        role = str(job.get("role", ""))
        seed = int(job.get("seed", -1))
        if not name or name in names:
            raise ValueError("job names must be non-empty and unique")
        if role not in expected_roles:
            raise ValueError("unknown job role: {}".format(role))
        if seed not in expected_seeds:
            raise ValueError(
                "job seed must be one of {}".format(sorted(expected_seeds))
            )
        backbone = str(job.get("backbone", ""))
        if protocol == "B" and not backbone:
            raise ValueError("protocol B jobs require a backbone")
        names.add(name)
        observed_roles.add(role)
        if backbone:
            observed_backbones.add(backbone)
        role_seeds[(backbone, role)].add(seed)

        work_dir = _resolve_path(root, job["work_dir"])
        if work_dir in work_dirs:
            raise ValueError("job work directories must be unique")
        work_dirs.add(work_dir)

        base_config_path = _resolve_path(root, job["base_config"])
        if not base_config_path.is_file():
            raise ValueError(
                "base config does not exist: {}".format(base_config_path)
            )
        config_path = _resolve_path(root, job["config"])
        if not config_path.is_file():
            raise ValueError("config does not exist: {}".format(config_path))
        args = _parse_config(root, config_path)
        if int(args.random_seed) != seed:
            raise ValueError(
                "{} config seed {} does not match job seed {}".format(
                    name,
                    args.random_seed,
                    seed,
                )
            )
        if _resolve_path(root, args.work_dir) != work_dir:
            raise ValueError(
                "{} config work_dir does not match the manifest".format(name)
            )
        if int(args.num_epoch) != 40:
            raise ValueError(
                "all protocol {} configs must use 40 epochs".format(protocol)
            )
        if int(args.model_args.get("num_classes", 0)) != int(
            manifest["num_classes"]
        ):
            raise ValueError(
                "config output class count does not match protocol {}".format(
                    protocol
                )
            )
        if args.decode_mode != "beam":
            raise ValueError(
                "all protocol {} configs must use beam decoding".format(
                    protocol
                )
            )
        if args.evaluate_tool != "sclite":
            raise ValueError(
                "all protocol {} configs must use sclite".format(protocol)
            )
        if args.checkpoint_split != "cal":
            raise ValueError(
                "all protocol {} configs must select checkpoints on D_cal".format(
                    protocol
                )
            )
        if args.split_manifest is None:
            raise ValueError(
                "all protocol {} configs require a split manifest".format(
                    protocol
                )
            )
        split_path = _resolve_path(root, args.split_manifest)
        split = load_split_manifest(str(split_path), str(split_path.parent))
        if split["split_sha256"] != EXPECTED_SPLIT_SHA256:
            raise ValueError(
                "config split hash does not match protocol {}".format(protocol)
            )
        if protocol == "A":
            _validate_recipe(role, args)
        else:
            if not backbone:
                raise ValueError("protocol B jobs require a backbone")
            if args.model_args.get("c2d_type") != backbone:
                raise ValueError(
                    "{} config backbone {} does not match {}".format(
                        name,
                        args.model_args.get("c2d_type"),
                        backbone,
                    )
                )
            comparison_group = job.get("comparison_group", backbone)
            if comparison_group != backbone:
                raise ValueError(
                    "{} mixes comparison group and backbone".format(name)
                )
            _validate_protocol_b_recipe(role, args)
        artifact_paths = {}
        if role == "pseudo_signer":
            artifact_paths = _pseudo_signer_artifacts(
                root,
                {"args": args},
            )
            signer_map = artifact_paths["signer_map"]
            previous_seed = pseudo_signer_map_seeds.get(signer_map)
            if previous_seed is not None and previous_seed != seed:
                raise ValueError(
                    "pseudo-signer maps must be seed-specific; "
                    "seeds {} and {} share {}".format(
                        previous_seed,
                        seed,
                        signer_map,
                    )
                )
            pseudo_signer_map_seeds[signer_map] = seed
            missing = [
                str(path)
                for path in artifact_paths.values()
                if not path.is_file()
            ]
            if require_artifacts and missing:
                raise ValueError(
                    "{} is missing required pseudo-signer artifacts: {}".format(
                        name,
                        ", ".join(missing),
                    )
                )
        validated_job = {
            "name": name,
            "role": role,
            "seed": seed,
            "base_config": str(base_config_path),
            "config": str(config_path),
            "work_dir": str(work_dir),
            "work_dir_exists": work_dir.exists(),
            "backbone": backbone or None,
            "comparison_group": job.get("comparison_group", backbone or None),
        }
        if artifact_paths:
            validated_job["pseudo_signer_artifacts"] = {
                key: {
                    "path": str(path),
                    "exists": path.is_file(),
                }
                for key, path in artifact_paths.items()
            }
        validated_jobs.append(validated_job)

    for (backbone, role), seeds in role_seeds.items():
        if seeds != expected_seeds:
            raise ValueError(
                "{}/{} must provide seeds {}".format(
                    backbone or protocol,
                    role,
                    sorted(expected_seeds),
                )
            )
    if protocol == "A" and observed_roles != expected_roles:
        raise ValueError("protocol A is missing required roles")
    if protocol == "B" and observed_roles != expected_roles:
        missing = expected_roles.difference(observed_roles)
        raise ValueError(
            "protocol B is missing roles: {}".format(sorted(missing))
        )

    artifacts = _find_test_artifacts(root)
    if artifacts:
        raise ValueError(
            "protocol {} planning found prohibited Test artifacts: {}".format(
                protocol,
                ", ".join(artifacts),
            )
        )
    runtime = runtime_status(root)
    if require_runtime:
        if runtime["ctcdecode_version"] != "1.0.3":
            raise ValueError("unexpected ctcdecode version")
        if not runtime["sclite_executable"]:
            raise ValueError("sclite is missing or not executable")

    return {
        "manifest": str(manifest_path),
        "protocol": protocol,
        "split_sha256": EXPECTED_SPLIT_SHA256,
        "split_manifest": str(split_manifest_path),
        "num_classes": int(manifest["num_classes"]),
        "num_jobs": len(validated_jobs),
        "seeds": sorted(expected_seeds),
        "roles": sorted(observed_roles),
        "backbones": sorted(observed_backbones),
        "comparison_groups": sorted(
            {
                job["comparison_group"]
                for job in validated_jobs
                if job["comparison_group"]
            }
        ),
        "test_artifacts": artifacts,
        "runtime": runtime,
        "jobs": validated_jobs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="protocols/a_protocol_manifest.json",
    )
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--require-runtime", action="store_true")
    parser.add_argument("--require-artifacts", action="store_true")
    args = parser.parse_args()
    report = validate_manifest(
        args.manifest,
        root=Path(args.root),
        require_runtime=args.require_runtime,
        require_artifacts=args.require_artifacts,
    )
    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
