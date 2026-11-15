import json
from pathlib import Path

import numpy as np
import yaml

from dataset.splits import load_split_manifest
from utils import get_parser


ROOT = Path(__file__).resolve().parents[1]
PREPROCESS = ROOT / "preprocess" / "phoenix2014"
SIGNERS = {
    "Signer01",
    "Signer03",
    "Signer04",
    "Signer05",
    "Signer07",
    "Signer08",
    "Signer09",
}


def test_leave_one_signer_artifacts():
    for signer in sorted(SIGNERS):
        base = PREPROCESS / "leave_one_signers" / signer
        manifest_path = base / "split_manifest.json"
        prior_path = base / "train_prior.npy"
        prior_report_path = base / "train_prior.json"
        config_path = ROOT / "configs" / "leave_one_signers" / (
            signer + ".yaml"
        )
        for path in (
            manifest_path,
            prior_path,
            prior_report_path,
            config_path,
        ):
            assert path.is_file(), path

        manifest = load_split_manifest(manifest_path, base)
        report = json.loads(prior_report_path.read_text())
        assert report["split_sha256"] == manifest["split_sha256"]
        assert report["num_train_samples"] == len(
            manifest["splits"]["train"]
        )
        prior = np.load(prior_path)
        assert prior.shape == (1296,)
        assert np.isclose(prior[1:].sum(), 1.0)

        with open(config_path, encoding="utf-8") as source:
            defaults = yaml.safe_load(source)
        parser = get_parser()
        parser.set_defaults(**defaults)
        args = parser.parse_args(["--config", str(config_path)])
        assert Path(args.split_manifest).resolve() == manifest_path.resolve()
        assert Path(
            args.model_args["class_prior_path"]
        ).resolve() == prior_path.resolve()
        assert args.model_args["signer_film"] is True
        assert args.loss_weights["SignerInv"] > 0.0
        assert args.loss_weights["SignerOrth"] > 0.0
        assert args.model_args["signer_inv_queue_size"] >= 2


if __name__ == "__main__":
    test_leave_one_signer_artifacts()
    print("leave-one-signer artifact tests passed")
