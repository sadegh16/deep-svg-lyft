import torch
from l5kit.dataset import AgentDataset as agent_dataset
from l5kit.rasterization import build_rasterizer as _build_rasterizer

from l5kit.rasterization import SemanticRasterizer, SemBoxRasterizer, BoxRasterizer
from .rasterizer import (
    render_semantic_map, rasterize_semantic, rasterize_sem_box, get_frame, rasterize_box)

import functools
import copy
import types
from deepsvg.svglib.svg import SVG, Bbox
from .utils import apply_colors

from deepsvg.config import _Config
from deepsvg.difflib.tensor import SVGTensor
from deepsvg.svglib.svg import SVG
from deepsvg.svglib.geom import Point

import math
import torch
import torch.utils.data
import random
from typing import List, Union
import pandas as pd
import os
import pickle

import csv






class AgentDataset(torch.utils.data.Dataset):
    def __init__(self, data_cfg: dict, zarr_dataset, rasterizer,
                 model_args, max_num_groups, max_seq_len,
                 perturbation=None, agents_mask=None,
                 min_frame_history=10, min_frame_future=1,
                 max_total_len=None, filter_uni=None, filter_platform=None,
                 filter_category=None, train_ratio=1.0, PAD_VAL=-1,csv_path=None):

        super().__init__()
        print(data_cfg)
        self.svg = True
        self.svg_cmds = True

        self.data = agent_dataset(
            data_cfg, zarr_dataset, rasterizer)

        self.MAX_NUM_GROUPS = max_num_groups
        self.MAX_SEQ_LEN = max_seq_len
        self.MAX_TOTAL_LEN = max_total_len

        if max_total_len is None:
            self.MAX_TOTAL_LEN = max_num_groups * max_seq_len


        self.model_args = model_args

        self.PAD_VAL = PAD_VAL

        # fieldnames = ["idx", "len_path", "max_len_commands"]
        # self.writer = csv.DictWriter(open(csv_path+"/full_result.csv", "w"), fieldnames)
        # self.writer.writeheader()




    @staticmethod
    def _uni_to_label(uni):
        if 48 <= uni <= 57:
            return uni - 48
        elif 65 <= uni <= 90:
            return uni - 65 + 10
        return uni - 97 + 36

    @staticmethod
    def _label_to_uni(label_id):
        if 0 <= label_id <= 9:
            return label_id + 48
        elif 10 <= label_id <= 35:
            return label_id + 65 - 10
        return label_id + 97 - 36

    def _load_tensor(self, icon_id):
        item = self.data[icon_id]
        if self.svg and self.svg_cmds:
            tens = SVG.from_tensor(item['path']).simplify().split_paths().to_tensor(concat_groups=False)
            svg = apply_colors(tens, item['path_type'])
        return svg

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.get(idx, self.model_args)

    @staticmethod
    def _augment(svg, mean=False):
        dx, dy = (0, 0) if mean else (5 * random.random() - 2.5, 5 * random.random() - 2.5)
        factor = 0.7 if mean else 0.2 * random.random() + 0.6

        return svg.zoom(factor).translate(Point(dx, dy))

    @staticmethod
    def simplify(svg, normalize=True):
        svg.canonicalize(normalize=normalize)
        svg = svg.simplify_heuristic()
        return svg.normalize()

    @staticmethod
    def normalize_history(svg, normalize=True):
        svg.canonicalize(normalize=normalize)
        return svg.normalize()

    @staticmethod
    def preprocess(svg, augment=True, numericalize=True, mean=False):
        if augment:
            svg = AgentDataset._augment(svg, mean=mean)
        if numericalize:
            return svg.numericalize(256)
        return svg

    def get(self, idx=0, model_args=None, random_aug=True, id=None, svg: SVG = None):
        item = self.data[idx]
        if self.svg and self.svg_cmds:
            tens_scene = self.simplify(SVG.from_tensor(item['path'])).split_paths().to_tensor(concat_groups=False)
            # tens_path = self.normalize_history(SVG.from_tensor(item['history_agent'])).split_paths().to_tensor(concat_groups=False)
            # tens_scene = apply_colors(tens_scene, item['path_type'])
            # tens_path = apply_colors(tens_path, item['history_agent_type'])
            tens = tens_scene
            del item['path']
            # del item['path_type']
            # del item['history_agent']
            # del item['history_agent_type']
            item['image'],item['valid'] = self.get_data(idx,tens, None, model_args=model_args, label=None)
        return item

    def get_data(self, idx, t_sep, fillings, model_args=None, label=None):
        res = {}
        valid = True
        # max_len_commands = 0
        # len_path = len(t_sep)
        if model_args is None:
            model_args = self.model_args
        if len(t_sep) > self.MAX_NUM_GROUPS:
            t_sep = t_sep[0:self.MAX_NUM_GROUPS]
            valid = False
        pad_len = max(self.MAX_NUM_GROUPS - len(t_sep), 0)

        t_sep.extend([torch.empty(0, 14)] * pad_len)

        t_grouped = [SVGTensor.from_data(torch.cat(t_sep, dim=0), PAD_VAL=self.PAD_VAL).add_eos().add_sos().pad(
            seq_len=self.MAX_TOTAL_LEN + 2)]
        t_normal = []
        for t in t_sep:
            s = SVGTensor.from_data(t, PAD_VAL=self.PAD_VAL)
            # print(s.commands.shape)
            if len(s.commands) > self.MAX_SEQ_LEN:
                s.commands = s.commands[0:self.MAX_SEQ_LEN]
                valid = False
            t_normal.append(s.add_eos().add_sos().pad(
                seq_len=self.MAX_SEQ_LEN + 2))
        # line = {"idx" : idx, "len_path" : len_path, "max_len_commands" : max_len_commands}
        # self.writer.writerow(line)
        # if max_len_commands > self.MAX_SEQ_LEN:
        #     return None
        # if len_path > self.MAX_NUM_GROUPS:
        #     return None

        for arg in set(model_args):
            if "_grouped" in arg:
                arg_ = arg.split("_grouped")[0]
                t_list = t_grouped
            else:
                arg_ = arg
                t_list = t_normal

            if arg_ == "tensor":
                res[arg] = t_list

            if arg_ == "commands":
                res[arg] = torch.stack([t.cmds() for t in t_list])

            if arg_ == "args_rel":
                res[arg] = torch.stack([t.get_relative_args() for t in t_list])
            if arg_ == "args":
                res[arg] = torch.stack([t.args()[0:self.MAX_SEQ_LEN+2] for t in t_list])

        if "filling" in model_args:
            res["filling"] = torch.stack([torch.tensor(t.filling) for t in t_sep]).unsqueeze(-1)

        if "label" in model_args:
            res["label"] = label
        return res,valid

