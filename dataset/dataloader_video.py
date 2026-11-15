import os
import cv2
import sys
import pdb
import re
import six
import glob
import time
import torch
import random
import pandas
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
# import pyarrow as pa
from PIL import Image
import torch.utils.data as data
import matplotlib.pyplot as plt
from dataset.splits import indexed_record_count, load_split_info
from dataset.signer_protocol import load_signer_map
from utils import video_augmentation
from torch.utils.data.sampler import Sampler

sys.path.append("..")


class BaseFeeder(data.Dataset):
    def __init__(self, prefix, gloss_dict, drop_ratio=1, num_gloss=-1, mode="train", transform_mode=True,
                 datatype="lmdb", split_manifest=None, split_name=None,
                 signer_map_path=None):
        self.mode = mode
        self.ng = num_gloss
        self.prefix = prefix
        self.dict = gloss_dict
        self.data_type = datatype
        self.feat_prefix = f"{prefix}/features/fullFrame-256x256px/{mode}"
        self.transform_mode = "train" if transform_mode else "test"
        self.split_manifest = split_manifest
        self.split_name = split_name
        self.signer_map = (
            load_signer_map(signer_map_path)
            if signer_map_path is not None
            else None
        )
        if split_manifest is None:
            self.inputs_list = np.load(
                f"./preprocess/phoenix2014/{mode}_info.npy",
                allow_pickle=True,
            ).item()
        else:
            if split_name is None:
                raise ValueError("split_name is required with split_manifest")
            manifest_path = os.path.abspath(split_manifest)
            self.inputs_list = load_split_info(
                os.path.dirname(manifest_path),
                split_name,
                manifest_path,
            )
        self.num_records = indexed_record_count(self.inputs_list)
        # self.inputs_list = np.load(f"{prefix}/annotations/manual/{mode}.corpus.npy", allow_pickle=True).item()
        # self.inputs_list = np.load(f"{prefix}/annotations/manual/{mode}.corpus.npy", allow_pickle=True).item()
        # self.inputs_list = dict([*filter(lambda x: isinstance(x[0], str) or x[0] < 10, self.inputs_list.items())])
        print(mode, len(self))
        self.data_aug = self.transform()
        print("")

    def __getitem__(self, idx):
        info = self.inputs_list[idx]['original_info']
        info = self._with_signer_override(info, self.inputs_list[idx])
        if self.data_type == "video":
            input_data, label, fi = self.read_video(idx)
            input_data, label = self.normalize(input_data, label)
            # input_data, label = self.normalize(input_data, label, fi['fileid'])
            return input_data, torch.LongTensor(label), info
        elif self.data_type == "lmdb":
            input_data, label, fi = self.read_lmdb(idx)
            input_data, label = self.normalize(input_data, label)
            return input_data, torch.LongTensor(label), info
        else:
            input_data, label = self.read_features(idx)
            return input_data, label, info

    def _with_signer_override(self, info, record):
        if self.signer_map is None:
            return info
        fileid = str(record.get("fileid"))
        if fileid not in self.signer_map:
            raise KeyError(
                "pseudo signer map has no assignment for fileid {}".format(fileid)
            )
        return "{}|__signer_id={}".format(info, self.signer_map[fileid])

    def read_video(self, index, num_glosses=-1):
        # load file info
        fi = self.inputs_list[index]
        img_folder = os.path.join(self.prefix, "features/fullFrame-256x256px/" + fi['folder'])
        img_list = sorted(glob.glob(img_folder))
        label_list = []
        for phase in fi['label'].split(" "):
            if phase == '':
                continue
            if phase in self.dict.keys():
                label_list.append(self.dict[phase][0])
        return [cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) for img_path in img_list], label_list, fi

    def read_features(self, index):
        # load file info
        fi = self.inputs_list[index]
        data = np.load(f"./features/{self.mode}/{fi['fileid']}_features.npy", allow_pickle=True).item()
        return data['features'], data['label']

    def normalize(self, video, label, file_id=None):
        video, label = self.data_aug(video, label, file_id)
        video = video.float() / 127.5 - 1
        return video, label

    def transform(self):
        if self.transform_mode == "train":
            print("Apply training transform.")
            return video_augmentation.Compose([
                # video_augmentation.CenterCrop(224),
                # video_augmentation.WERAugment('/lustre/wangtao/current_exp/exp/baseline/boundary.npy'),
                video_augmentation.RandomCrop(224),
                video_augmentation.RandomHorizontalFlip(0.5),
                video_augmentation.ToTensor(),
                video_augmentation.TemporalRescale(0.2),
                # video_augmentation.Resize(0.5),
            ])
        else:
            print("Apply testing transform.")
            return video_augmentation.Compose([
                video_augmentation.CenterCrop(224),
                # video_augmentation.Resize(0.5),
                video_augmentation.ToTensor(),
            ])

    def byte_to_img(self, byteflow):
        unpacked = pa.deserialize(byteflow)
        imgbuf = unpacked[0]
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')
        return img

    @staticmethod
    def collate_fn(batch):
        batch = [item for item in sorted(batch, key=lambda x: len(x[0]), reverse=True)]
        video, label, info = list(zip(*batch))
        signer_ids = []
        for sample_info in info:
            override = re.search(r"__signer_id=(\d+)", sample_info)
            if override is not None:
                signer_ids.append(int(override.group(1)))
            else:
                fields = sample_info.split("|")
                signer_label = fields[2] if len(fields) > 2 else ""
                digits = "".join(character for character in signer_label if character.isdigit())
                signer_ids.append(min(int(digits), 31) if digits else 0)
        signer_ids = torch.LongTensor(signer_ids)
        if len(video[0].shape) > 3:
            max_len = len(video[0])
            video_length = torch.LongTensor([np.ceil(len(vid) / 4.0) * 4 + 12 for vid in video])
            left_pad = 6
            right_pad = int(np.ceil(max_len / 4.0)) * 4 - max_len + 6
            max_len = max_len + left_pad + right_pad
            padded_video = [torch.cat(
                (
                    vid[0][None].expand(left_pad, -1, -1, -1),
                    vid,
                    vid[-1][None].expand(max_len - len(vid) - left_pad, -1, -1, -1),
                )
                , dim=0)
                for vid in video]
            padded_video = torch.stack(padded_video)
        else:
            max_len = len(video[0])
            video_length = torch.LongTensor([len(vid) for vid in video])
            padded_video = [torch.cat(
                (
                    vid,
                    vid[-1][None].expand(max_len - len(vid), -1),
                )
                , dim=0)
                for vid in video]
            padded_video = torch.stack(padded_video).permute(0, 2, 1)
        label_length = torch.LongTensor([len(lab) for lab in label])
        if max(label_length) == 0:
            return padded_video, video_length, [], [], signer_ids, info
        else:
            padded_label = []
            for lab in label:
                padded_label.extend(lab)
            padded_label = torch.LongTensor(padded_label)
            return (
                padded_video,
                video_length,
                padded_label,
                label_length,
                signer_ids,
                info,
            )

    def __len__(self):
        return self.num_records

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time


if __name__ == "__main__":
    feeder = BaseFeeder()
    dataloader = torch.utils.data.DataLoader(
        dataset=feeder,
        batch_size=1,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    for data in dataloader:
        pdb.set_trace()
