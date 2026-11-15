import inspect
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fd_xmp_network import FDXMPModel
from utils import get_parser


STRICT_CONFIGS = {
    ROOT / "configs" / "baseline_split.yaml",
    ROOT / "configs" / "vac_ctc_split.yaml",
    ROOT / "configs" / "vac_smkd_split.yaml",
    ROOT / "configs" / "fdxmp_v3_split.yaml",
    ROOT / "configs" / "fdxmp_gaussian_split.yaml",
    ROOT / "configs" / "fdxmp_h1_split.yaml",
    ROOT / "configs" / "fdxmp_h2_split.yaml",
    ROOT / "configs" / "fdxmp_h2_calibrated.yaml",
    ROOT / "configs" / "fdxmp_weighted_ctc_split.yaml",
    ROOT / "configs" / "fdxmp_pra_split.yaml",
    ROOT / "configs" / "fdxmp_rdb_split.yaml",
    ROOT / "configs" / "fdxmp_semantic_text_split.yaml",
    ROOT / "configs" / "fdxmp_no_signer_split.yaml",
    ROOT / "configs" / "fdxmp_pseudo_signer_split.yaml",
}
CONFIGS = STRICT_CONFIGS | {ROOT / "configs" / "fdxmp_proto.yaml"}
VAC_CTC_CONFIG = ROOT / "configs" / "vac_ctc_split.yaml"
VAC_SMKD_CONFIG = ROOT / "configs" / "vac_smkd_split.yaml"
SUPPORTED_LOSSES = {
    "SeqCTC",
    "ConvCTC",
    "Dist",
    "ProtoAlign",
    "TextAlign",
    "SignerReg",
    "SignerCE",
    "SignerAdv",
    "SignerInv",
    "SignerOrth",
    "WeightedCTC",
    "PRA",
    "RDBRecon",
    "RDBCTC",
    "RDBStyle",
}


def test_split_configs_match_parser():
    model_parameters = set(
        inspect.signature(FDXMPModel.__init__).parameters
    )
    for config_path in CONFIGS:
        parser = get_parser()
        initial = parser.parse_args(["--config", str(config_path)])
        with open(config_path, encoding="utf-8") as source:
            defaults = yaml.load(source, Loader=yaml.FullLoader)
        parser = get_parser()
        parser.set_defaults(**defaults)
        args = parser.parse_args(["--config", str(config_path)])
        if config_path in STRICT_CONFIGS:
            assert args.split_manifest.endswith("split_manifest.json")
            manifest_path = Path(args.split_manifest)
            if not manifest_path.is_absolute():
                manifest_path = ROOT / manifest_path
            assert manifest_path.is_file()
        else:
            assert args.split_manifest is None
        assert args.device == 0
        assert args.num_epoch == 40
        assert args.random_seed == 0
        assert args.evaluate_tool == "sclite"
        assert args.decode_mode == "beam"
        assert args.optimizer_args["grad_clip_norm"] == 5.0
        unknown_model_args = set(args.model_args).difference(model_parameters)
        assert not unknown_model_args, (
            config_path.name,
            sorted(unknown_model_args),
        )
        assert set(args.loss_weights).issubset(SUPPORTED_LOSSES)


def test_vac_readme_recipe():
    with open(VAC_CTC_CONFIG, encoding="utf-8") as source:
        vac_ctc = yaml.load(source, Loader=yaml.FullLoader)
    assert vac_ctc["loss_weights"]["SeqCTC"] == 1.0
    assert vac_ctc["loss_weights"]["ConvCTC"] == 1.0
    assert vac_ctc["model_args"]["share_classifier"] is False
    assert vac_ctc["model_args"]["weight_norm"] is False

    with open(VAC_SMKD_CONFIG, encoding="utf-8") as source:
        vac_smkd = yaml.load(source, Loader=yaml.FullLoader)
    assert vac_smkd["loss_weights"]["SeqCTC"] == 1.0
    assert vac_smkd["loss_weights"]["ConvCTC"] == 1.0
    assert vac_smkd["model_args"]["share_classifier"] is True
    assert vac_smkd["model_args"]["weight_norm"] is True


if __name__ == "__main__":
    test_split_configs_match_parser()
    test_vac_readme_recipe()
    print("split config tests passed")
