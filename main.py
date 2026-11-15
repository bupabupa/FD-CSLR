import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import json
import pdb
import sys
import cv2
import yaml
import torch
import random
import importlib
import faulthandler
import numpy as np
import torch.nn as nn
from collections import OrderedDict
from pathlib import Path

faulthandler.enable()
import utils
from dataset.splits import load_split_manifest
from modules.calibration import (
    FUSION_PARAMETERIZATION_VERSION,
    validate_fusion_state,
)
from modules.sync_batchnorm import convert_model
from seq_scripts import seq_train, seq_eval, seq_feature_generation


def validate_test_access(arg):
    """Require a locked candidate claim before any Test split is loaded."""
    phase = getattr(arg, "phase", "train")
    load_splits = getattr(arg, "load_splits", None)
    if load_splits is not None:
        requested = set(load_splits)
    elif phase == "train":
        requested = {"train", "train_eval", "dev"}
    elif phase == "test":
        requested = set(getattr(arg, "eval_splits", ()))
    elif phase == "features":
        requested = {"train_eval", "dev", "test"}
    else:
        requested = {"train", "train_eval", "dev", "test"}
    if "test" not in requested:
        return None

    work_dir = Path(arg.work_dir).resolve()
    claim_path = None
    for parent in (work_dir, *work_dir.parents):
        candidate = parent / "test_access_claim.json"
        if candidate.is_file():
            claim_path = candidate
            break
    if claim_path is None:
        raise RuntimeError(
            "Test evaluation is locked: test_access_claim.json was not found "
            "in the work directory or its ancestors"
        )
    try:
        with open(claim_path, encoding="utf-8") as source:
            claim = json.load(source)
    except (OSError, ValueError) as error:
        raise RuntimeError(
            "invalid Test access claim: {}".format(claim_path)
        ) from error
    if not isinstance(claim, dict) or not isinstance(
        claim.get("selected"), dict
    ):
        raise RuntimeError(
            "invalid Test access claim: missing selected candidate"
        )
    required = {"name", "dev_prediction", "config", "checkpoint"}
    missing = required.difference(claim["selected"])
    if missing:
        raise RuntimeError(
            "invalid Test access claim: missing fields {}".format(
                sorted(missing)
            )
        )
    return claim_path


class Processor():
    def __init__(self, arg):
        self.arg = arg
        self.test_access_claim = validate_test_access(arg)
        self.save_arg()
        if self.arg.random_fix:
            self.rng = utils.RandomState(seed=self.arg.random_seed)
        self.device = utils.GpuDataParallel()
        self.recoder = utils.Recorder(self.arg.work_dir, self.arg.print_log, self.arg.log_interval)
        self.dataset = {}
        self.data_loader = {}
        self.gloss_dict = np.load(self.arg.dataset_info['dict_path'], allow_pickle=True).item()
        self.arg.model_args['num_classes'] = len(self.gloss_dict) + 1
        self.model, self.optimizer = self.loading()

    def start(self):
        if self.arg.phase == 'train':
            self.recoder.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            seq_model_list = []
            checkpoint_split = getattr(
                self.arg,
                "checkpoint_split",
                "dev",
            )
            best_split_wer = float('inf')
            last_split_wer = None
            grad_accum_steps = int(
                self.arg.optimizer_args.get("grad_accum_steps", 1)
            )
            grad_clip_norm = float(
                self.arg.optimizer_args.get("grad_clip_norm", 0.0)
            )
            for epoch in range(self.arg.optimizer_args['start_epoch'], self.arg.num_epoch):
                snapshot_epoch = epoch % self.arg.save_interval == 0
                eval_model = epoch % self.arg.eval_interval == 0
                # train end2end model
                seq_train(self.data_loader['train'], self.model, self.optimizer,
                          self.device, epoch, self.recoder,
                          grad_accum_steps=grad_accum_steps,
                          grad_clip_norm=grad_clip_norm)
                improved = False
                if eval_model:
                    split_wer = seq_eval(
                        self.arg,
                        self.data_loader[checkpoint_split],
                        self.model,
                        self.device,
                        checkpoint_split,
                        epoch,
                        self.arg.work_dir,
                        self.recoder,
                        self.arg.evaluate_tool,
                    )
                    self.recoder.print_log(
                        "{} WER: {:05.2f}%".format(
                            checkpoint_split,
                            split_wer,
                        )
                    )
                    last_split_wer = split_wer
                    if split_wer < best_split_wer - 1e-9:
                        best_split_wer = split_wer
                        improved = True
                if eval_model and improved:
                    model_path = "{}{}_{:05.2f}_epoch{}_model.pt".format(
                        self.arg.work_dir,
                        checkpoint_split,
                        split_wer,
                        epoch,
                    )
                    seq_model_list.append(model_path)
                    print("seq_model_list", seq_model_list)
                    self.save_model(epoch, model_path)
                elif snapshot_epoch and last_split_wer is not None:
                    model_path = "{}{}_{:05.2f}_epoch{}_model.pt".format(
                        self.arg.work_dir,
                        checkpoint_split,
                        last_split_wer,
                        epoch,
                    )
                    seq_model_list.append(model_path)
                    print("seq_model_list", seq_model_list)
                    self.save_model(epoch, model_path)
        elif self.arg.phase == 'test':
            if self.arg.load_weights is None and self.arg.load_checkpoints is None:
                raise ValueError('Please appoint --load-weights.')
            self.recoder.print_log('Model:   {}.'.format(self.arg.model))
            self.recoder.print_log('Weights: {}.'.format(self.arg.load_weights))
            for mode in self.arg.eval_splits:
                seq_eval(
                    self.arg,
                    self.data_loader[mode],
                    self.model,
                    self.device,
                    mode,
                    6667,
                    self.arg.work_dir,
                    self.recoder,
                    self.arg.evaluate_tool,
                )
            self.recoder.print_log('Evaluation Done.\n')
        elif self.arg.phase == "features":
            for mode in ["train", "dev", "test"]:
                seq_feature_generation(
                    self.data_loader[mode + "_eval" if mode == "train" else mode],
                    self.model, self.device, mode, self.arg.work_dir, self.recoder
                )

    def save_arg(self):
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            yaml.dump(arg_dict, f)

    def save_model(self, epoch, save_path):
        split_sha256 = None
        if self.arg.split_manifest is not None:
            split_sha256 = load_split_manifest(
                self.arg.split_manifest,
                os.path.dirname(os.path.abspath(self.arg.split_manifest)),
            )["split_sha256"]
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.optimizer.scheduler.state_dict(),
            'rng_state': self.rng.save_rng_state(),
            'split_sha256': split_sha256,
            'checkpoint_selection_split': getattr(
                self.arg,
                "checkpoint_split",
                "dev",
            ),
        }, save_path)

    def loading(self):
        self.device.set_device(self.arg.device)
        print("Loading model")
        model_class = import_class(self.arg.model)
        model = model_class(
            **self.arg.model_args,
            gloss_dict=self.gloss_dict,
            loss_weights=self.arg.loss_weights,
        )
        optimizer = utils.Optimizer(model, self.arg.optimizer_args)

        if self.arg.load_weights:
            self.load_model_weights(model, self.arg.load_weights)
        elif self.arg.load_checkpoints:
            self.load_checkpoint_weights(model, optimizer)
        model = self.model_to_device(model)
        print("Loading model finished.")
        self.load_data()
        return model, optimizer

    def model_to_device(self, model):
        model = model.to(self.device.output_device)
        if len(self.device.gpu_list) > 1:
            model.conv2d = nn.DataParallel(
                model.conv2d,
                device_ids=self.device.gpu_list,
                output_device=self.device.output_device)
        model = convert_model(model)
        model.cuda()
        return model

    def load_model_weights(self, model, weight_path):
        state_dict = torch.load(weight_path)
        if self.arg.split_manifest is not None:
            expected = load_split_manifest(
                self.arg.split_manifest,
                os.path.dirname(os.path.abspath(self.arg.split_manifest)),
            )["split_sha256"]
            if state_dict.get("split_sha256") != expected:
                raise ValueError(
                    "checkpoint split hash does not match split manifest"
                )
        if len(self.arg.ignore_weights):
            for w in self.arg.ignore_weights:
                if state_dict.pop(w, None) is not None:
                    print('Successfully Remove Weights: {}.'.format(w))
                else:
                    print('Can Not Remove Weights: {}.'.format(w))
        weights = self.modified_weights(state_dict['model_state_dict'], False)
        # weights = self.modified_weights(state_dict['model_state_dict'])
        prefix = "multichannel_calibrator."
        fusion_state = {
            name[len(prefix):]: value
            for name, value in weights.items()
            if name.startswith(prefix)
        }
        if fusion_state:
            declared_version = state_dict.get(
                "multichannel_parameterization_version"
            )
            if declared_version != FUSION_PARAMETERIZATION_VERSION:
                raise ValueError(
                    "multichannel checkpoint metadata version is missing or "
                    "incompatible"
                )
            validate_fusion_state(fusion_state)
        model.load_state_dict(weights, strict=True)

    @staticmethod
    def modified_weights(state_dict, modified=False):
        state_dict = OrderedDict([(k.replace('.module', ''), v) for k, v in state_dict.items()])
        if not modified:
            return state_dict
        modified_dict = dict()
        return modified_dict

    def load_checkpoint_weights(self, model, optimizer):
        self.load_model_weights(model, self.arg.load_checkpoints)
        state_dict = torch.load(self.arg.load_checkpoints)

        if len(torch.cuda.get_rng_state_all()) == len(state_dict['rng_state']['cuda']):
            print("Loading random seeds...")
            self.rng.set_rng_state(state_dict['rng_state'])
        if "optimizer_state_dict" in state_dict.keys():
            print("Loading optimizer parameters...")
            optimizer.load_state_dict(state_dict["optimizer_state_dict"])
            optimizer.to(self.device.output_device)
        if "scheduler_state_dict" in state_dict.keys():
            print("Loading scheduler parameters...")
            optimizer.scheduler.load_state_dict(state_dict["scheduler_state_dict"])

        self.arg.optimizer_args['start_epoch'] = state_dict["epoch"] + 1
        self.recoder.print_log("Resuming from checkpoint: epoch {self.arg.optimizer_args['start_epoch']}")

    def load_data(self):
        print("Loading data")
        self.feeder = import_class(self.arg.feeder)
        available = [
            ("train", True),
            ("train_eval", False),
            ("dev", False),
            ("test", False),
        ]
        if self.arg.split_manifest is not None:
            available.append(("cal", False))
        requested = getattr(self.arg, "load_splits", None)
        if requested is None:
            phase = getattr(self.arg, "phase", "train")
            if phase == "train":
                default_views = (
                    "train",
                    "train_eval",
                    getattr(self.arg, "checkpoint_split", "dev"),
                )
            elif phase == "test":
                default_views = tuple(self.arg.eval_splits)
            elif phase == "features":
                default_views = ("train_eval", "dev", "test")
            else:
                default_views = tuple(item[0] for item in available)
            requested_set = set(default_views)
            dataset_list = [
                item for item in available if item[0] in requested_set
            ]
        else:
            requested_set = set(requested)
            dataset_list = [
                item for item in available if item[0] in requested_set
            ]
            missing = requested_set - set(item[0] for item in dataset_list)
            if missing:
                raise ValueError(
                    "requested dataset views are unavailable: {}".format(
                        ", ".join(sorted(missing))
                    )
                )
        split_names = {
            "train": "train",
            "train_eval": "train",
            "cal": "cal",
            "dev": "sel",
            "test": "test",
        }
        source_modes = {
            "train": "train",
            "train_eval": "train",
            "cal": "train",
            "dev": "dev",
            "test": "test",
        }
        for idx, (mode, train_flag) in enumerate(dataset_list):
            arg = dict(self.arg.feeder_args)
            arg["prefix"] = self.arg.dataset_info['dataset_root']
            arg["mode"] = source_modes[mode]
            arg["transform_mode"] = train_flag
            if self.arg.split_manifest is not None:
                arg["split_manifest"] = self.arg.split_manifest
                arg["split_name"] = split_names[mode]
            self.dataset[mode] = self.feeder(gloss_dict=self.gloss_dict, **arg)
            self.data_loader[mode] = self.build_dataloader(self.dataset[mode], mode, train_flag)
        print("Loading data finished.")

    def build_dataloader(self, dataset, mode, train_flag):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.arg.batch_size if mode == "train" else self.arg.test_batch_size,
            shuffle=train_flag,
            drop_last=train_flag,
            num_workers=self.arg.num_worker,  # if train_flag else 0
            collate_fn=self.feeder.collate_fn,
        )


def import_class(name):
    components = name.rsplit('.', 1)
    mod = importlib.import_module(components[0])
    mod = getattr(mod, components[1])
    return mod


if __name__ == '__main__':
    sparser = utils.get_parser()
    p = sparser.parse_args()
    # p.config = "baseline_iter.yaml"
    if p.config is not None:
        with open(p.config, 'r') as f:
            try:
                default_arg = yaml.load(f, Loader=yaml.FullLoader)
            except AttributeError:
                default_arg = yaml.load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        sparser.set_defaults(**default_arg)
    args = sparser.parse_args()
    with open(f"./configs/{args.dataset}.yaml", 'r') as f:
        args.dataset_info = yaml.load(f, Loader=yaml.FullLoader)
    processor = Processor(args)
    utils.pack_code("./", args.work_dir)
    processor.start()
