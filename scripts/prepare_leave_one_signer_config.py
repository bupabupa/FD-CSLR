import argparse
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-config",
        default=str(ROOT / "configs" / "fdxmp_h1_split.yaml"),
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--prior", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--work-dir", required=True)
    args = parser.parse_args()

    manifest = Path(args.manifest).expanduser().resolve()
    prior = Path(args.prior).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest}")
    if not prior.is_file():
        raise FileNotFoundError(f"prior not found: {prior}")
    with open(args.base_config, encoding="utf-8") as source:
        config = yaml.safe_load(source)
    config["split_manifest"] = str(manifest)
    config["work_dir"] = str(Path(args.work_dir).expanduser())
    config["model_args"] = dict(config["model_args"])
    config["model_args"]["class_prior_path"] = str(prior)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as destination:
        yaml.safe_dump(
            config,
            destination,
            sort_keys=False,
            allow_unicode=True,
        )
    print(f"config: {output}")
    print(f"manifest: {manifest}")
    print(f"prior: {prior}")


if __name__ == "__main__":
    main()
