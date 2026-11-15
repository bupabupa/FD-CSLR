import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset.splits import (
    build_leave_one_signer_manifest,
    write_split_manifest,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preprocess-dir",
        default=str(ROOT / "preprocess" / "phoenix2014"),
    )
    parser.add_argument("--heldout-signer", required=True)
    parser.add_argument(
        "--output-dir",
        default=str(
            ROOT
            / "preprocess"
            / "phoenix2014"
            / "leave_one_signers"
        ),
    )
    parser.add_argument("--calibration-fraction", type=float, default=0.1)
    parser.add_argument("--selection-fraction", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--minimum-heldout-samples", type=int, default=50)
    args = parser.parse_args()

    manifest = build_leave_one_signer_manifest(
        args.preprocess_dir,
        heldout_signer=args.heldout_signer,
        calibration_fraction=args.calibration_fraction,
        selection_fraction=args.selection_fraction,
        seed=args.seed,
        minimum_heldout_samples=args.minimum_heldout_samples,
    )
    output = write_split_manifest(
        manifest,
        Path(args.output_dir)
        / str(args.heldout_signer)
        / "split_manifest.json",
    )
    print(f"manifest: {output}")
    print(f"split_sha256: {manifest['split_sha256']}")
    for name, statistics in manifest["statistics"].items():
        print(f"{name}: {statistics['num_samples']} samples")


if __name__ == "__main__":
    main()
