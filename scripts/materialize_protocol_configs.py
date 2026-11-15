import argparse
import json
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.check_protocol_manifest import expand_manifest_jobs  # noqa: E402


def _resolve(root, value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _merge_overrides(config, overrides):
    if not isinstance(overrides, dict):
        raise ValueError("job overrides must be a mapping")
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(config.get(key), dict):
            _merge_overrides(config[key], value)
        else:
            config[key] = value


def _render_overrides(value, context):
    """Expand seed-dependent override values without mutating the manifest."""
    if isinstance(value, dict):
        return {
            key: _render_overrides(item, context)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_render_overrides(item, context) for item in value]
    if isinstance(value, str):
        return value.format(**context)
    return value


def materialize(manifest_path, root=ROOT):
    root = Path(root).resolve()
    manifest_path = _resolve(root, manifest_path)
    with open(manifest_path, encoding="utf-8") as source:
        manifest = json.load(source)
    outputs = []
    for job in expand_manifest_jobs(manifest):
        base_config = _resolve(root, job["base_config"])
        target = _resolve(root, job["config"])
        with open(base_config, encoding="utf-8") as source:
            config = yaml.safe_load(source)
        if not isinstance(config, dict):
            raise ValueError(
                "base config must contain a mapping: {}".format(base_config)
            )
        _merge_overrides(
            config,
            _render_overrides(job.get("overrides", {}), job),
        )
        config["random_seed"] = int(job["seed"])
        config["work_dir"] = str(_resolve(root, job["work_dir"])) + "/"
        config["num_epoch"] = int(manifest["num_epochs"])
        config["decode_mode"] = "beam"
        config["evaluate_tool"] = "sclite"
        config["checkpoint_split"] = "cal"
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        with open(temporary, "w", encoding="utf-8", newline="\n") as output:
            yaml.safe_dump(config, output, sort_keys=False)
        temporary.replace(target)
        outputs.append({
            "job": job["name"],
            "seed": int(job["seed"]),
            "config": str(target),
        })
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="protocols/a_protocol_manifest.json",
    )
    parser.add_argument("--root", default=str(ROOT))
    args = parser.parse_args()
    outputs = materialize(args.manifest, root=args.root)
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
