import os
import pdb
import sys
import copy
import json
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from evaluation.frequency_metrics import (
    class_frequency_report,
    gloss_occurrence_counts,
)
from dataset.splits import load_split_info
from evaluation.slr_eval.wer_calculation import evaluate
from sequence_normalization import normalized_wer


def _decode_gloss(token):
    if isinstance(token, (tuple, list)):
        return str(token[0])
    return str(token)


def _sequence_wer(references, hypotheses, preprocess_script):
    return normalized_wer(
        references,
        hypotheses,
        preprocess_script,
    )["wer"]


def _gradient_norm_without_clipping(parameters):
    parameters = [
        parameter for parameter in parameters
        if parameter.grad is not None
    ]
    if not parameters:
        return None
    get_total_norm = getattr(torch.nn.utils, "get_total_norm", None)
    if get_total_norm is not None:
        return get_total_norm(parameters, norm_type=2)
    parameter_norms = [
        torch.linalg.vector_norm(parameter.grad.detach(), ord=2)
        for parameter in parameters
    ]
    return torch.linalg.vector_norm(torch.stack(parameter_norms), ord=2)


def _nonfinite_parameter_names(model, gradient=False):
    names = []
    for name, parameter in model.named_parameters():
        value = parameter.grad if gradient else parameter
        if value is not None and not torch.isfinite(value).all():
            names.append(name)
    return names


def _save_nonfinite_batch(
    model,
    optimizer,
    recoder,
    epoch_idx,
    batch_idx,
    data,
    vid,
    vid_lgt,
    label,
    label_lgt,
):
    log_path = getattr(recoder, "log_path", None)
    if not log_path:
        return None
    output_dir = os.path.dirname(os.path.abspath(log_path))
    output_path = os.path.join(
        output_dir,
        "nonfinite_gradient_epoch{}_batch{}.pt".format(
            epoch_idx,
            batch_idx,
        ),
    )
    payload = {
        "epoch": epoch_idx,
        "batch_idx": batch_idx,
        "model_state_dict": {
            name: value.detach().cpu()
            for name, value in model.state_dict().items()
        },
        "vid": vid.detach().cpu(),
        "vid_lgt": vid_lgt.detach().cpu(),
        "label": label.detach().cpu(),
        "label_lgt": label_lgt.detach().cpu(),
        "metadata": [str(item) for item in data[-1]],
    }
    if hasattr(optimizer, "state_dict"):
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, output_path)
    return output_path


def seq_train(
    loader,
    model,
    optimizer,
    device,
    epoch_idx,
    recoder,
    grad_accum_steps=1,
    grad_clip_norm=0.0,
):
    model.train()
    set_training_epoch = getattr(model, "set_training_epoch", None)
    if set_training_epoch is not None:
        set_training_epoch(epoch_idx)
    grad_accum_steps = int(grad_accum_steps)
    if grad_accum_steps < 1:
        raise ValueError("grad_accum_steps must be positive")
    grad_clip_norm = float(grad_clip_norm)
    if grad_clip_norm < 0:
        raise ValueError("grad_clip_norm must be non-negative")
    loss_value = []
    signer_inv_coverage_value = []
    ctc_valid_counts = {}
    ctc_total_counts = {}
    clr = [group['lr'] for group in optimizer.optimizer.param_groups]
    optimizer.zero_grad()
    for batch_idx, data in enumerate(tqdm(loader)):
        vid = device.data_to_device(data[0])
        vid_lgt = device.data_to_device(data[1])
        label = device.data_to_device(data[2])
        label_lgt = device.data_to_device(data[3])
        model_kwargs = {"label": label, "label_lgt": label_lgt}
        if getattr(model, "supports_signer", False):
            if len(data) < 6:
                raise ValueError("signer model requires signer ids in each batch")
            model_kwargs["signer_ids"] = device.data_to_device(data[4])
        ret_dict = model(vid, vid_lgt, **model_kwargs)
        loss = model.criterion_calculation(ret_dict, label, label_lgt)
        for branch_name, validity in getattr(
            model,
            "last_ctc_validity",
            {},
        ).items():
            ctc_valid_counts[branch_name] = (
                ctc_valid_counts.get(branch_name, 0)
                + int(validity.sum().item())
            )
            ctc_total_counts[branch_name] = (
                ctc_total_counts.get(branch_name, 0)
                + int(validity.numel())
            )
        if not torch.isfinite(loss):
            print(data[-1])
            raise FloatingPointError(
                "non-finite training loss at epoch {}, batch {}: {}".format(
                    epoch_idx,
                    batch_idx,
                    loss.item(),
                )
            )
        group_start = (batch_idx // grad_accum_steps) * grad_accum_steps
        group_size = min(grad_accum_steps, len(loader) - group_start)
        (loss / float(group_size)).backward()
        should_step = (
            (batch_idx + 1) % grad_accum_steps == 0
            or batch_idx + 1 == len(loader)
        )
        if should_step:
            if grad_clip_norm > 0:
                preclip_norm = _gradient_norm_without_clipping(
                    model.parameters()
                )
                if (
                    preclip_norm is not None
                    and not torch.isfinite(preclip_norm)
                ):
                    gradient_names = _nonfinite_parameter_names(
                        model,
                        gradient=True,
                    )
                    parameter_names = _nonfinite_parameter_names(model)
                    try:
                        failure_path = _save_nonfinite_batch(
                            model,
                            optimizer,
                            recoder,
                            epoch_idx,
                            batch_idx,
                            data,
                            vid,
                            vid_lgt,
                            label,
                            label_lgt,
                        )
                    except Exception as save_error:
                        failure_path = "save failed: {}".format(save_error)
                    raise FloatingPointError(
                        "pre-clip gradient norm at epoch {}, batch {} is {}; "
                        "non-finite gradient parameters={}; non-finite model "
                        "parameters={}; diagnostic={}".format(
                            epoch_idx,
                            batch_idx,
                            preclip_norm.item(),
                            gradient_names,
                            parameter_names,
                            failure_path,
                        )
                    )
                total_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    grad_clip_norm,
                )
                if not torch.isfinite(total_norm):
                    raise FloatingPointError(
                        "non-finite gradient norm at epoch {}, batch {}: {}".format(
                            epoch_idx,
                            batch_idx,
                            total_norm.item(),
                        )
                    )
            optimizer.step()
            optimizer.zero_grad()
        loss_value.append(loss.item())
        if "signer_inv_coverage" in ret_dict:
            signer_inv_coverage_value.append(
                float(ret_dict["signer_inv_coverage"].detach().item())
            )
        if batch_idx % recoder.log_interval == 0:
            message = (
                '\tEpoch: {}, Batch({}/{}) done. Loss: {:.8f}  lr:{:.6f}'
                .format(epoch_idx, batch_idx, len(loader), loss.item(), clr[0])
            )
            if signer_inv_coverage_value:
                message += "  signer_inv_cov:{:.4f}".format(
                    signer_inv_coverage_value[-1]
                )
            if ctc_total_counts:
                valid_summary = []
                for branch_name in sorted(ctc_total_counts):
                    valid_summary.append(
                        "{}={}/{}".format(
                            branch_name,
                            ctc_valid_counts[branch_name],
                            ctc_total_counts[branch_name],
                        )
                    )
                message += "  CTC_valid[{}]".format(
                    ", ".join(valid_summary)
                )
            recoder.print_log(message)
    optimizer.scheduler.step()
    recoder.print_log('\tMean training loss: {:.10f}.'.format(np.mean(loss_value)))
    if signer_inv_coverage_value:
        recoder.print_log(
            '\tMean signer invariance coverage: {:.6f}.'.format(
                np.mean(signer_inv_coverage_value)
            )
        )
    return loss_value


def seq_eval(cfg, loader, model, device, mode, epoch, work_dir, recoder,
             evaluate_tool="python"):
    model.eval()
    total_sent = []
    total_info = []
    total_metadata = []
    total_conv_sent = []
    reference_sequences = []
    hypothesis_sequences = []
    unknown_hypothesis_tokens = 0
    gloss_to_id = {
        str(gloss): int(value[0]) for gloss, value in loader.dataset.dict.items()
    }
    stat = {i: [0, 0] for i in range(len(loader.dataset.dict))}
    for batch_idx, data in enumerate(tqdm(loader)):
        recoder.record_timer("device")
        vid = device.data_to_device(data[0])
        vid_lgt = device.data_to_device(data[1])
        label = device.data_to_device(data[2])
        label_lgt = device.data_to_device(data[3])
        model_kwargs = {"label": label, "label_lgt": label_lgt}
        if getattr(model, "supports_signer", False):
            if len(data) < 6:
                raise ValueError("signer model requires signer ids in each batch")
            model_kwargs["signer_ids"] = device.data_to_device(data[4])
        with torch.no_grad():
            ret_dict = model(vid, vid_lgt, **model_kwargs)

        total_metadata += list(data[-1])
        total_info += [file_name.split("|")[0] for file_name in data[-1]]
        total_sent += ret_dict['recognized_sents']
        total_conv_sent += ret_dict['conv_sents']
        label_offset = 0
        label_lengths = label_lgt.detach().cpu().tolist()
        flat_labels = label.detach().cpu().tolist()
        for sample_idx, label_length in enumerate(label_lengths):
            label_end = label_offset + int(label_length)
            reference_sequences.append(flat_labels[label_offset:label_end])
            hypothesis = []
            for token in ret_dict['recognized_sents'][sample_idx]:
                gloss = _decode_gloss(token)
                gloss_id = gloss_to_id.get(gloss, -1)
                if gloss_id < 0:
                    unknown_hypothesis_tokens += 1
                else:
                    hypothesis.append(gloss_id)
            hypothesis_sequences.append(hypothesis)
            label_offset = label_end
    python_eval = evaluate_tool == "python"
    write2file(
        work_dir + "output-hypothesis-{}.ctm".format(mode),
        total_info,
        total_sent,
    )
    write2file(
        work_dir + "output-hypothesis-{}-conv.ctm".format(mode),
        total_info,
        total_conv_sent,
    )
    if mode == "cal":
        # The calibration split is derived from train and intentionally has no
        # standalone STM file. Score it directly from manifest-aligned labels.
        conv_ret = None
        id_to_gloss = {
            int(value[0]): str(gloss)
            for gloss, value in loader.dataset.dict.items()
        }
        lstm_ret = _sequence_wer(
            [
                [id_to_gloss[class_id] for class_id in reference]
                for reference in reference_sequences
            ],
            [
                [_decode_gloss(token) for token in sentence]
                for sentence in total_sent
            ],
            os.path.join(
                cfg.dataset_info["evaluation_dir"],
                "preprocess.sh",
            ),
        )
    else:
        conv_ret = evaluate(
            prefix=work_dir,
            mode=mode,
            output_file="output-hypothesis-{}-conv.ctm".format(mode),
            evaluate_dir=cfg.dataset_info["evaluation_dir"],
            evaluate_prefix=cfg.dataset_info["evaluation_prefix"],
            output_dir="epoch_{}_{}_result/".format(epoch, mode),
            python_evaluate=python_eval,
        )
        lstm_ret = evaluate(
            prefix=work_dir,
            mode=mode,
            output_file="output-hypothesis-{}.ctm".format(mode),
            evaluate_dir=cfg.dataset_info["evaluation_dir"],
            evaluate_prefix=cfg.dataset_info["evaluation_prefix"],
            output_dir="epoch_{}_{}_result/".format(epoch, mode),
            python_evaluate=python_eval,
            triplet=True,
        )
    recoder.print_log(f"Epoch {epoch}, {mode} {lstm_ret: 2.2f}%", f"{work_dir}/{mode}.txt")
    try:
        dict_dir = os.path.dirname(cfg.dataset_info['dict_path'])
        configured_manifest = getattr(cfg, "split_manifest", None)
        if configured_manifest is not None:
            split_manifest = os.path.abspath(configured_manifest)
            train_info = load_split_info(
                os.path.dirname(split_manifest),
                "train",
                split_manifest,
            )
        else:
            train_info = np.load(
                os.path.join(dict_dir, "train_info.npy"), allow_pickle=True
            ).item()
        train_counts = gloss_occurrence_counts(
            train_info, loader.dataset.dict
        )
        report = class_frequency_report(
            reference_sequences,
            hypothesis_sequences,
            train_counts,
            blank_id=0,
        )
        report["decode"] = {
            "mode": mode,
            "epoch": int(epoch),
            "evaluate_tool": evaluate_tool,
            "unknown_hypothesis_tokens": unknown_hypothesis_tokens,
        }
        report_path = os.path.join(work_dir, f"frequency-metrics-{mode}.json")
        with open(report_path, "w", encoding="utf-8") as report_file:
            json.dump(report, report_file, indent=2, sort_keys=True)
        id_to_gloss = {
            int(value[0]): str(gloss)
            for gloss, value in loader.dataset.dict.items()
        }
        prediction_path = os.path.join(
            work_dir, f"predictions-{mode}.jsonl"
        )
        with open(prediction_path, "w", encoding="utf-8") as output:
            for sample_id, metadata, reference, hypothesis, sentence in zip(
                total_info,
                total_metadata,
                reference_sequences,
                hypothesis_sequences,
                total_sent,
            ):
                metadata_fields = metadata.split("|")
                output.write(json.dumps({
                    "sample_id": sample_id,
                    "signer": (
                        metadata_fields[2]
                        if len(metadata_fields) > 2 else ""
                    ),
                    "reference_ids": reference,
                    "hypothesis_ids": hypothesis,
                    "reference_glosses": [
                        id_to_gloss[class_id] for class_id in reference
                    ],
                    "hypothesis_glosses": [
                        _decode_gloss(token) for token in sentence
                    ],
                }, ensure_ascii=False, sort_keys=True))
                output.write("\n")
        recoder.print_log(
            f"Frequency metrics written to {report_path}; "
            f"predictions written to {prediction_path}",
            f"{work_dir}/{mode}.txt",
        )
    except Exception as error:
        recoder.print_log(
            f"Frequency metrics unavailable: {error}",
            f"{work_dir}/{mode}.txt",
        )
    return lstm_ret


def seq_feature_generation(loader, model, device, mode, work_dir, recoder):
    model.eval()

    src_path = os.path.abspath(f"{work_dir}{mode}")
    tgt_path = os.path.abspath(f"./features/{mode}")
    if not os.path.exists("./features/"):
        os.makedirs("./features/")

    if os.path.islink(tgt_path):
        curr_path = os.readlink(tgt_path)
        if work_dir[1:] in curr_path and os.path.isabs(curr_path):
            return
        else:
            os.unlink(tgt_path)
    else:
        if os.path.exists(src_path) and len(loader.dataset) == len(os.listdir(src_path)):
            os.symlink(src_path, tgt_path)
            return

    for batch_idx, data in tqdm(enumerate(loader)):
        recoder.record_timer("device")
        vid = device.data_to_device(data[0])
        vid_lgt = device.data_to_device(data[1])
        model_kwargs = {}
        if getattr(model, "supports_signer", False):
            model_kwargs["signer_ids"] = device.data_to_device(data[4])
        with torch.no_grad():
            ret_dict = model(vid, vid_lgt, **model_kwargs)
        if not os.path.exists(src_path):
            os.makedirs(src_path)
        start = 0
        for sample_idx in range(len(vid)):
            end = start + data[3][sample_idx]
            filename = f"{src_path}/{data[-1][sample_idx].split('|')[0]}_features.npy"
            save_file = {
                "label": data[2][start:end],
                "features": ret_dict['framewise_features'][sample_idx][:, :vid_lgt[sample_idx]].T.cpu().detach(),
            }
            np.save(filename, save_file)
            start = end
        assert end == len(data[2])
    os.symlink(src_path, tgt_path)


def write2file(path, info, output):
    filereader = open(path, "w")
    for sample_idx, sample in enumerate(output):
        for word_idx, word in enumerate(sample):
            filereader.writelines(
                "{} 1 {:.2f} {:.2f} {}\n".format(info[sample_idx],
                                                 word_idx * 1.0 / 100,
                                                 (word_idx + 1) * 1.0 / 100,
                                                 word[0]))
