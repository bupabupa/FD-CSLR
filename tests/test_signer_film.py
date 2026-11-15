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


def main():
    with open(ROOT / "configs/fdxmp_h1_split.yaml", encoding="utf-8") as source:
        config = yaml.safe_load(source)
    with open(ROOT / "configs/phoenix14.yaml", encoding="utf-8") as source:
        dataset_info = yaml.safe_load(source)
    gloss_dict = np.load(
        ROOT / dataset_info["dict_path"], allow_pickle=True
    ).item()
    model_args = dict(config["model_args"])
    model_args["num_classes"] = len(gloss_dict) + 1
    model = FDXMPModel(
        **model_args,
        gloss_dict=gloss_dict,
        loss_weights=config["loss_weights"],
    ).cuda()
    model.train()
    feeder = BaseFeeder(
        prefix=dataset_info["dataset_root"],
        gloss_dict=gloss_dict,
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
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=feeder.collate_fn,
    )
    video, video_length, label, label_length, signer_ids, _ = next(iter(loader))
    device = torch.device("cuda")
    video = video.to(device)
    video_length = video_length.to(device)
    label = label.to(device)
    label_length = label_length.to(device)
    signer_ids = signer_ids.to(device)

    output = model(
        video,
        video_length,
        label=label,
        label_lgt=label_length,
        signer_ids=signer_ids,
    )
    expected_hidden = model_args.get("hidden_size", 1024)
    assert output["signer_gamma"].shape == (1, expected_hidden)
    torch.testing.assert_close(
        output["signer_gamma"], torch.zeros_like(output["signer_gamma"])
    )
    loss = model.criterion_calculation(output, label, label_length)
    assert torch.isfinite(loss)
    assert torch.isfinite(output["signer_loss"])
    assert torch.isfinite(output["signer_inv_loss"])
    assert torch.isfinite(output["signer_orthogonal_loss"])
    assert output["signer_logits"].shape == (1, model_args["max_signer_id"] + 1)
    assert output["gloss_signer_logits"].shape == (
        1,
        model_args["max_signer_id"] + 1,
    )
    assert torch.isfinite(output["gloss_signer_loss"])
    assert model.signer_occurrence_valid.any()
    assert model.signer_occurrence_ptr.item() > 0
    loss.backward()
    assert model.signer_gamma_basis.grad is not None
    assert model.signer_gamma_basis.grad.abs().sum().item() > 0
    assert model.signer_classifier.weight.grad is not None
    assert model.signer_classifier.weight.grad.abs().sum().item() > 0
    assert model.gloss_signer_classifier.weight.grad is not None
    assert model.gloss_signer_classifier.weight.grad.abs().sum().item() > 0
    assert model.signer_codes.weight.grad is None

    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-4
    )
    optimizer.step()
    assert model.signer_gamma_basis.norm().item() > 0
    assert model.signer_beta_basis.norm().item() > 0

    model.eval()
    with torch.no_grad():
        other_signer = torch.where(
            signer_ids == 4, torch.tensor(7, device=device), torch.tensor(4, device=device)
        )
        first = model(video, video_length, decode=False, signer_ids=signer_ids)
        second = model(video, video_length, decode=False, signer_ids=other_signer)
        unseen_signer = torch.full_like(
            signer_ids, model_args["max_signer_id"]
        )
        unseen = model(
            video,
            video_length,
            decode=False,
            signer_ids=unseen_signer,
        )
    assert not torch.allclose(
        first["base_logits"], second["base_logits"]
    )
    torch.testing.assert_close(
        unseen["signer_gamma"], torch.zeros_like(unseen["signer_gamma"])
    )
    torch.testing.assert_close(
        unseen["signer_beta"], torch.zeros_like(unseen["signer_beta"])
    )
    print(
        "signer film test passed:",
        f"signer={signer_ids.item()}",
        f"gamma_norm={output['signer_gamma'].norm().item():.6g}",
        f"grad={model.signer_gamma_basis.grad.norm().item():.6g}",
        flush=True,
    )


if __name__ == "__main__":
    main()
