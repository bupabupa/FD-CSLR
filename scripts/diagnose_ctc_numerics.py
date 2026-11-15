import argparse
import gc

import torch
import torch.nn.functional as F

from diagnose_nonfinite_backward import load_model, nonfinite_names


def branch_logits(model, inputs, input_lgt, branch):
    output = model(inputs, input_lgt)
    key = "conv_logits" if branch == "ConvCTC" else "sequence_logits"
    return output[key], output["feat_len"]


def ctc_losses(log_probs, label, input_lgt, label_lgt):
    return F.ctc_loss(
        log_probs,
        label.cpu().int(),
        input_lgt.cpu().int(),
        label_lgt.cpu().int(),
        blank=0,
        reduction="none",
        zero_infinity=False,
    )


def run_variant(
    model,
    inputs,
    input_lgt,
    label,
    label_lgt,
    branch,
    name,
    transform,
):
    model.zero_grad(set_to_none=True)
    logits, ctc_input_lgt = branch_logits(
        model,
        inputs,
        input_lgt,
        branch,
    )
    log_probs = logits.log_softmax(-1)
    transformed = transform(log_probs)
    losses = ctc_losses(transformed, label, ctc_input_lgt, label_lgt)
    print(
        "\n",
        branch,
        name,
        "log_probs_range",
        [log_probs.min().item(), log_probs.max().item()],
        "losses",
        losses.detach().cpu().tolist(),
    )
    try:
        losses.mean().backward()
        error = None
    except RuntimeError as caught:
        error = str(caught)
    gradients = nonfinite_names(model.named_parameters(), gradients=True)
    print("error", error)
    print("nonfinite_gradients", gradients)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    payload = torch.load(args.artifact, map_location="cuda:0")
    payload.pop("optimizer_state_dict", None)
    gc.collect()
    torch.cuda.empty_cache()
    model, _ = load_model(args.config)
    model.load_state_dict(payload["model_state_dict"])
    model.train()
    inputs = payload["vid"].cuda()
    input_lgt = payload["vid_lgt"].cuda()
    label = payload["label"].cuda()
    label_lgt = payload["label_lgt"].cuda()

    variants = (
        ("f32", lambda value: value),
        ("f32_clamp_min_20", lambda value: value.clamp_min(-20.0)),
        ("f32_clamp_min_30", lambda value: value.clamp_min(-30.0)),
        ("f32_clamp_min_50", lambda value: value.clamp_min(-50.0)),
        ("float64", lambda value: value.double()),
    )
    for branch in ("SeqCTC", "ConvCTC"):
        for name, transform in variants:
            run_variant(
                model,
                inputs,
                input_lgt,
                label,
                label_lgt,
                branch,
                name,
                transform,
            )


if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    main()
