import argparse
import gc

import numpy as np
import torch
import yaml

from main import import_class
from modules.sync_batchnorm import convert_model


def nonfinite_names(parameters, gradients=False):
    names = []
    for name, parameter in parameters:
        value = parameter.grad if gradients else parameter
        if value is not None and not torch.isfinite(value).all():
            names.append(name)
    return names


def load_model(config_path):
    with open(config_path, encoding="utf-8") as source:
        config = yaml.load(source, Loader=yaml.FullLoader)
    with open("./configs/phoenix14.yaml", encoding="utf-8") as source:
        dataset_info = yaml.load(source, Loader=yaml.FullLoader)
    gloss_dict = np.load(
        dataset_info["dict_path"],
        allow_pickle=True,
    ).item()
    model_args = dict(config["model_args"])
    model_args["num_classes"] = len(gloss_dict) + 1
    model_class = import_class(config["model"])
    model = model_class(
        **model_args,
        gloss_dict=gloss_dict,
        loss_weights=config["loss_weights"],
    )
    model = model.cuda()
    model = convert_model(model).cuda()
    return model, config


def print_forward_stats(output):
    for name in ("sequence_logits", "conv_logits"):
        value = output[name]
        print(
            name,
            "shape={}".format(tuple(value.shape)),
            "min={:.6f}".format(value.min().item()),
            "max={:.6f}".format(value.max().item()),
            "finite={}".format(bool(torch.isfinite(value).all().item())),
        )


def run_backward(
    model,
    inputs,
    input_lgt,
    label,
    label_lgt,
    weights,
    name,
):
    print("\n=== {} ===".format(name))
    model.zero_grad(set_to_none=True)
    model.loss_weights = weights
    try:
        with torch.autograd.detect_anomaly():
            output = model(
                inputs,
                input_lgt,
                label=label,
                label_lgt=label_lgt,
            )
            print_forward_stats(output)
            loss = model.criterion_calculation(output, label, label_lgt)
            print(
                "loss={} finite={}".format(
                    loss.item(),
                    bool(torch.isfinite(loss).item()),
                )
            )
            loss.backward()
    except RuntimeError as error:
        print("BACKWARD_ERROR:", error)

    gradient_names = nonfinite_names(
        model.named_parameters(),
        gradients=True,
    )
    parameter_names = nonfinite_names(
        model.named_parameters(),
        gradients=False,
    )
    print("nonfinite_gradient_parameters:", gradient_names)
    print("nonfinite_model_parameters:", parameter_names)
    for parameter_name in (
        "conv1d.temporal_conv.0.weight",
        "conv1d.temporal_conv.4.weight",
        "conv1d.fc.weight",
        "temporal_model.rnn.weight_ih_l0",
        "classifier.weight",
    ):
        parameter = dict(model.named_parameters()).get(parameter_name)
        if parameter is not None and parameter.grad is not None:
            print(
                "grad_norm",
                parameter_name,
                torch.linalg.vector_norm(parameter.grad).item(),
            )


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
    label = payload["label"].cuda()
    label_lgt = payload["label_lgt"].cuda()
    input_lgt = payload["vid_lgt"].cuda()
    print("epoch", payload["epoch"], "batch", payload["batch_idx"])
    print("metadata", payload["metadata"])
    print("input_shape", tuple(inputs.shape))
    print("input_length", input_lgt.tolist())
    print("label_length", label_lgt.tolist())

    run_backward(
        model,
        inputs,
        input_lgt,
        label,
        label_lgt,
        {"SeqCTC": 1.0},
        "SeqCTC only",
    )
    run_backward(
        model,
        inputs,
        input_lgt,
        label,
        label_lgt,
        {"ConvCTC": 1.0},
        "ConvCTC only",
    )
    run_backward(
        model,
        inputs,
        input_lgt,
        label,
        label_lgt,
        {"SeqCTC": 1.0, "ConvCTC": 1.0},
        "SeqCTC + ConvCTC",
    )


if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    main()
