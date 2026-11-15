import os
from pathlib import Path
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from dataset.dataloader_video import BaseFeeder
from fd_xmp_network import FDXMPModel


def load_inputs():
    with open(ROOT / "configs/fdxmp_rdb_split.yaml", encoding="utf-8") as source:
        config = yaml.safe_load(source)
    with open(ROOT / "configs/phoenix14.yaml", encoding="utf-8") as source:
        dataset_info = yaml.safe_load(source)
    gloss_dict = np.load(
        ROOT / dataset_info["dict_path"], allow_pickle=True
    ).item()
    feeder = BaseFeeder(
        prefix=dataset_info["dataset_root"],
        gloss_dict=gloss_dict,
        drop_ratio=1.0,
        num_gloss=-1,
        mode="train",
        transform_mode=True,
        datatype="video",
        split_manifest=str(
            ROOT / "preprocess/phoenix2014/split_manifest.json"
        ),
        split_name="train",
    )
    loader = DataLoader(
        feeder,
        batch_size=2,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        collate_fn=feeder.collate_fn,
    )
    batch = next(iter(loader))
    return config, gloss_dict, batch


def to_device(batch):
    video, video_length, label, label_length, signer_ids, _ = batch
    return (
        video.cuda(),
        video_length.cuda(),
        label.cuda(),
        label_length.cuda(),
        signer_ids.cuda(),
    )


def run_signer_model(config, gloss_dict, batch):
    video, video_length, label, label_length, signer_ids = to_device(batch)
    model_args = dict(config["model_args"])
    model_args["num_classes"] = len(gloss_dict) + 1
    model = FDXMPModel(
        **model_args,
        gloss_dict=gloss_dict,
        loss_weights=config["loss_weights"],
    ).cuda()
    model.train()

    # Force two valid, distinct signer styles for the counterfactual branch.
    signer_ids = torch.tensor([4, 7], device=video.device, dtype=torch.long)
    output = model(
        video,
        video_length,
        label=label,
        label_lgt=label_length,
        signer_ids=signer_ids,
    )
    loss = model.criterion_calculation(output, label, label_length)
    assert torch.isfinite(loss)
    torch.testing.assert_close(
        output["rdb_features"],
        output["conditioned_features"],
        atol=0,
        rtol=0,
    )
    assert output["rdb_logits"] is not None
    assert output["rdb_style_logits"] is not None
    assert model.last_rdb_style_available
    assert torch.isfinite(output["rdb_recon_loss"])
    assert output["rdb_recon_loss"].item() >= 0.0
    assert 0.0 < output["rdb_strength"].item() < model_args["rdb_max_strength"]
    loss.backward()
    assert model.rdb.output.weight.grad is not None
    gradient_norms = {
        "rdb_output": model.rdb.output.weight.grad.abs().sum().item(),
        "rdb_strength": model.rdb.strength_raw.grad.abs().item(),
        "classifier": model.classifier.weight.grad.abs().sum().item(),
        "signer_gamma": model.signer_gamma_basis.grad.abs().sum().item(),
    }
    print("signer gradients:", gradient_norms)
    assert gradient_norms["rdb_output"] > 0.0
    result = {
        "loss": float(loss.item()),
        "recon": float(output["rdb_recon_loss"].item()),
        "active_fraction": float(output["rdb_active_fraction"].item()),
        "style_available": bool(model.last_rdb_style_available),
    }
    del model
    torch.cuda.empty_cache()
    return result


def run_no_signer_model(config, gloss_dict, batch):
    video, video_length, label, label_length, _ = to_device(batch)
    model_args = dict(config["model_args"])
    model_args["num_classes"] = len(gloss_dict) + 1
    model_args["signer_film"] = False
    loss_weights = {
        name: weight
        for name, weight in config["loss_weights"].items()
        if not name.startswith("Signer") and name != "RDBStyle"
    }
    model = FDXMPModel(
        **model_args,
        gloss_dict=gloss_dict,
        loss_weights=loss_weights,
    ).cuda()
    model.train()
    output = model(
        video,
        video_length,
        label=label,
        label_lgt=label_length,
    )
    loss = model.criterion_calculation(output, label, label_length)
    assert torch.isfinite(loss)
    assert output["rdb_logits"] is not None
    assert output["rdb_style_logits"] is None
    assert not model.last_rdb_style_available
    loss.backward()
    assert model.rdb.output.weight.grad is not None
    result = {
        "loss": float(loss.item()),
        "recon": float(output["rdb_recon_loss"].item()),
        "style_available": bool(model.last_rdb_style_available),
    }
    del model
    torch.cuda.empty_cache()
    return result


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("RDB smoke requires CUDA")
    config, gloss_dict, batch = load_inputs()
    signer_result = run_signer_model(config, gloss_dict, batch)
    no_signer_result = run_no_signer_model(config, gloss_dict, batch)
    print("RDB GPU smoke passed")
    print("signer:", signer_result)
    print("no_signer:", no_signer_result)


if __name__ == "__main__":
    main()
