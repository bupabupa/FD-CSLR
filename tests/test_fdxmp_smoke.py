import os
import sys
from pathlib import Path

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
    with open(ROOT / "configs/fdxmp_v3_split.yaml", "r") as handle:
        config = yaml.safe_load(handle)
    with open(ROOT / "configs/phoenix14.yaml", "r") as handle:
        dataset_info = yaml.safe_load(handle)

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
    assert not model.supports_signer
    assert model.signer_classifier is None
    assert model.gloss_signer_classifier is None
    assert model.signer_codes is None

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
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        collate_fn=feeder.collate_fn,
    )
    video, video_length, label, label_length, signer_ids, _ = next(iter(loader))
    video = video.cuda()
    video_length = video_length.cuda()
    label = label.cuda()
    label_length = label_length.cuda()

    prototype_updates = []
    original_update = model.prototype_head.update

    def capture_update(features, gamma, confidence):
        prototype_updates.append(features.detach().clone())
        return original_update(features, gamma, confidence)

    model.prototype_head.update = capture_update
    output = model(
        video,
        video_length,
        label=label,
        label_lgt=label_length,
    )
    loss = model.criterion_calculation(output, label, label_length)
    assert torch.isfinite(loss)
    assert torch.isfinite(output["proto_align_loss"])
    assert output["proto_align_loss"].item() > 0
    assert len(prototype_updates) == 1
    torch.testing.assert_close(
        prototype_updates[0],
        output["conditioned_features"].detach(),
        atol=0,
        rtol=0,
    )
    loss.backward()

    assert model.prototype_head.support.sum().item() > 0
    assert torch.isfinite(model.prototype_head.prototypes).all()
    assert model.prototype_gate.grad is not None
    assert torch.isfinite(model.prototype_gate.grad)

    model.eval()
    with torch.no_grad():
        eval_output = model(
            video,
            video_length,
            decode=False,
        )
        fdgc_output = model.fdgc(
            eval_output["sequence_logits"],
            source_prior=model._nonblank_source_prior(),
            difficulty=model.fdgc_difficulty,
        )
    torch.testing.assert_close(
        fdgc_output["logits"],
        eval_output["sequence_logits"],
        atol=1e-6,
        rtol=1e-6,
    )
    assert eval_output["recognized_sents"] is None
    assert torch.count_nonzero(fdgc_output["tau"]) == 0
    assert fdgc_output["blank_bias"].abs().item() == 0

    print(
        "fdxmp smoke passed:",
        f"video={tuple(video.shape)}",
        f"logits={tuple(output['sequence_logits'].shape)}",
        f"loss={loss.item():.6f}",
        f"proto_align={output['proto_align_loss'].item():.6f}",
        f"gate={output['prototype_gate'].item():.6f}",
        f"gate_grad={model.prototype_gate.grad.item():.6g}",
        f"support_classes={(model.prototype_head.support > 0).sum().item()}",
        f"fdgc_identity={torch.max(torch.abs(fdgc_output['logits'] - eval_output['sequence_logits'])).item():.3g}",
        flush=True,
    )


if __name__ == "__main__":
    main()
