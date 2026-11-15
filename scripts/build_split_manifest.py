import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset.splits import build_phoenix_split_manifest, write_split_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preprocess-dir",
        default=str(ROOT / "preprocess" / "phoenix2014"),
    )
    parser.add_argument(
        "--output",
        default=str(
            ROOT / "preprocess" / "phoenix2014" / "split_manifest.json"
        ),
    )
    parser.add_argument("--calibration-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    manifest = build_phoenix_split_manifest(
        args.preprocess_dir,
        calibration_fraction=args.calibration_fraction,
        seed=args.seed,
    )
    output = write_split_manifest(manifest, args.output)
    print(f"manifest: {output}")
    print(f"split_sha256: {manifest['split_sha256']}")
    for name, statistics in manifest["statistics"].items():
        print(f"{name}: {statistics['num_samples']} samples")


if __name__ == "__main__":
    main()
