import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import utils
from dataset.splits import file_sha256
from main import Processor
from utils.decode import _collapsed_beam


def unique_nbest(
    beam_result,
    beam_scores,
    out_seq_len,
    blank_id=0,
):
    candidates = {}
    for beam_index in range(len(beam_scores)):
        tokens = tuple(_collapsed_beam(
            beam_result[beam_index],
            out_seq_len[beam_index],
            blank_id,
        ))
        score = float(beam_scores[beam_index])
        if tokens not in candidates or score < candidates[tokens]:
            candidates[tokens] = score
    return [
        {"tokens": list(tokens), "ctc_score": score}
        for tokens, score in sorted(
            candidates.items(), key=lambda item: item[1]
        )
    ]


def load_runtime_args(
    config_path,
    load_weights,
    work_dir,
    device,
    batch_size,
    num_workers,
    beam_width,
    num_processes,
):
    parser = utils.get_parser()
    parser.parse_args(["--config", str(config_path)])
    with open(config_path, encoding="utf-8") as source:
        defaults = yaml.load(source, Loader=yaml.FullLoader)
    parser = utils.get_parser()
    parser.set_defaults(**defaults)
    args = parser.parse_args([
        "--config",
        str(config_path),
        "--phase",
        "test",
        "--device",
        str(device),
        "--work-dir",
        str(work_dir),
        "--load-weights",
        str(load_weights),
        "--eval-splits",
        "dev",
        "--load-splits",
        "dev",
        "--test-batch-size",
        str(batch_size),
        "--num-worker",
        str(num_workers),
        "--evaluate-tool",
        "python",
    ])
    args.model_args = dict(args.model_args)
    args.model_args.update({
        "beam_width": int(beam_width),
        "num_processes": int(num_processes),
        "rescorer_path": None,
        "rescorer_alpha": 0.0,
        "rescorer_beta": 0.0,
    })
    dataset_config = ROOT / "configs" / "{}.yaml".format(args.dataset)
    with open(dataset_config, encoding="utf-8") as source:
        args.dataset_info = yaml.load(
            source, Loader=yaml.FullLoader
        )
    return args


def cache_nbest(args, output_path, limit=None):
    processor = Processor(args)
    processor.model.eval()
    loader = processor.data_loader["dev"]
    decoder = processor.model.decoder.ctc_decoder
    records = []
    for data in loader:
        video = processor.device.data_to_device(data[0])
        video_lengths = processor.device.data_to_device(data[1])
        labels = processor.device.data_to_device(data[2])
        label_lengths = processor.device.data_to_device(data[3])
        model_kwargs = {
            "label": labels,
            "label_lgt": label_lengths,
        }
        if getattr(processor.model, "supports_signer", False):
            model_kwargs["signer_ids"] = processor.device.data_to_device(
                data[4]
            )
        with torch.no_grad():
            ret_dict = processor.model(
                video,
                video_lengths,
                **model_kwargs,
            )
        logits = ret_dict["sequence_logits"].softmax(-1).permute(1, 0, 2)
        beam_result, beam_scores, _, out_seq_len = decoder.decode(
            logits.cpu(),
            ret_dict["feat_len"].cpu(),
        )
        flat_labels = labels.detach().cpu().tolist()
        flat_lengths = label_lengths.detach().cpu().tolist()
        metadata_items = list(data[-1])
        label_offset = 0
        for sample_index, metadata in enumerate(metadata_items):
            label_end = label_offset + int(flat_lengths[sample_index])
            reference = [
                int(value)
                for value in flat_labels[label_offset:label_end]
            ]
            label_offset = label_end
            metadata_fields = metadata.split("|")
            records.append({
                "sample_id": metadata_fields[0],
                "signer": (
                    metadata_fields[2]
                    if len(metadata_fields) > 2 else ""
                ),
                "reference_ids": reference,
                "beams": unique_nbest(
                    beam_result[sample_index],
                    beam_scores[sample_index],
                    out_seq_len[sample_index],
                    blank_id=processor.model.decoder.blank_id,
                ),
            })
            if limit is not None and len(records) >= int(limit):
                break
        if limit is not None and len(records) >= int(limit):
            break

    split_sha256 = None
    if args.split_manifest is not None:
        with open(args.split_manifest, encoding="utf-8") as source:
            split_sha256 = json.load(source)["split_sha256"]
    payload = {
        "version": 1,
        "protocol": "fd_xmp_ngram_nbest_v1",
        "split": "dev",
        "strict_split": args.split_manifest is not None,
        "split_sha256": split_sha256,
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(Path(args.load_weights).resolve()),
        "checkpoint_sha256": file_sha256(args.load_weights),
        "beam_width": int(args.model_args["beam_width"]),
        "ctc_score_semantics": "negative_log_probability",
        "num_samples": len(records),
        "records": records,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as destination:
        json.dump(payload, destination, separators=(",", ":"))
        destination.write("\n")
    temporary.replace(output_path)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--load-weights", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--beam-width", type=int, default=50)
    parser.add_argument("--num-processes", type=int, default=4)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit must be positive")
    runtime_args = load_runtime_args(
        config_path=args.config,
        load_weights=args.load_weights,
        work_dir=args.run_dir,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        beam_width=args.beam_width,
        num_processes=args.num_processes,
    )
    payload = cache_nbest(
        runtime_args,
        args.output,
        limit=args.limit,
    )
    print(json.dumps({
        "output": str(Path(args.output).resolve()),
        "num_samples": payload["num_samples"],
        "strict_split": payload["strict_split"],
        "split_sha256": payload["split_sha256"],
        "beam_width": payload["beam_width"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
