from src.lyft.data import agent_dataset
from src.lyft.utils import apply_colors

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

from src.argoverse.utils.svg_utils import BaseDataset


import csv

class SVGDataset(torch.utils.data.Dataset):
    def __init__(self, data_type, model_args, max_num_groups, max_seq_len,
                 data_cfg: dict=None, zarr_dataset=None, rasterizer=None,
                 perturbation=None, agents_mask=None,
                 min_frame_history=10, min_frame_future=1,
                 data_dict = None, args=None, mode=None,
                 max_total_len=None, PAD_VAL=-1,csv_path=None,model_type=None):

        super().__init__()
        self.model_type = model_type
        if data_type == "lyft":
            print(data_cfg)
            map_type = data_cfg['raster_params']['map_type']
            self.svg_args = data_cfg['raster_params'].get('svg_args', dict())
            self.svg = map_type.startswith('svg_')
            self.svg_cmds = self.svg_args.get('return_cmds', True)
            print(self.svg,self.svg_cmds)
            self.tensor = map_type.startswith('tensor_')
            self.data = agent_dataset(
                data_cfg, zarr_dataset, rasterizer, perturbation, agents_mask, min_frame_history, min_frame_future)
        elif data_type == "argo":
            self.svg = True
            self.svg_cmds = True
            self.data = BaseDataset(data_dict, args, mode)

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
    def preprocess(svg, augment=True, numericalize=True, mean=False):
        if augment:
            svg = SVGDataset._augment(svg, mean=mean)
        if numericalize:
            return svg.numericalize(256)
        return svg
    @staticmethod
    def normalize_history(svg, normalize=True):
        svg.canonicalize(normalize=normalize)
        return svg.normalize()

    def get(self, idx=0, model_args=None, random_aug=True, id=None, svg: SVG = None):
        item = self.data[idx]
        if self.svg and self.svg_cmds:
            tens_scene=[]
            tens_path=[]
            if len(item['path'])!=0:
                tens_scene = self.simplify(SVG.from_tensor(item['path'])).split_paths().to_tensor(concat_groups=False)
            if len(item['history_agent'])!=0:
                #                 print(item['history_agent'])
                tens_path = self.normalize_history(SVG.from_tensor(item['history_agent'])).split_paths().to_tensor(concat_groups=False)
            if len(item['path_type'])!=0:
                tens_scene = apply_colors(tens_scene, item['path_type'])
            if len(item['history_agent_type'])!=0:
                tens_path = apply_colors(tens_path, item['history_agent_type'])
            del item['path']
            del item['path_type']
            del item['history_agent']
            del item['history_agent_type']
            if self.model_type == 6:
                #                 print(len(tens_path))
                item['image'] = self.get_data(idx,tens_scene, None, model_args=model_args, label=None)
                item['history_svg'] = self.get_data_history(idx,tens_path, None, model_args=model_args, label=None)
            #                 print(item['history_svg']["args"].shape,item['history_svg']["commands"].shape)
            else:
                tens = tens_scene+tens_path
                item['image'] = self.get_data(idx,tens, None, model_args=model_args, label=None)
        if item['image'] is None:
            return None
        return item

    def get_data(self, idx, t_sep, fillings, model_args=None, label=None):
        res = {}
        # max_len_commands = 0
        # len_path = len(t_sep)
        if model_args is None:
            model_args = self.model_args
        if len(t_sep) > self.MAX_NUM_GROUPS:
            return None
        pad_len = max(self.MAX_NUM_GROUPS - len(t_sep), 0)

        t_sep.extend([torch.empty(0, 14)] * pad_len)

        t_grouped = [SVGTensor.from_data(torch.cat(t_sep, dim=0), PAD_VAL=self.PAD_VAL).add_eos().add_sos().pad(
            seq_len=self.MAX_TOTAL_LEN + 2)]
        t_normal = []
        for t in t_sep:
            s = SVGTensor.from_data(t, PAD_VAL=self.PAD_VAL)
            if len(s.commands) > self.MAX_SEQ_LEN:
                return None
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
                res[arg] = torch.stack([t.args() for t in t_list])

        if "filling" in model_args:
            res["filling"] = torch.stack([torch.tensor(t.filling) for t in t_sep]).unsqueeze(-1)

        if "label" in model_args:
            res["label"] = label
        return res


    def get_data_history(self, idx, t_sep, fillings, model_args=None, label=None):
        res = {}
        # max_len_commands = 0
        # len_path = len(t_sep)
        if model_args is None:
            model_args = self.model_args
        pad_len = 0

        t_sep.extend([torch.empty(0, 14)] * pad_len)
        #             print("t_sep",len(t_sep))

        t_grouped = [SVGTensor.from_data(torch.cat(t_sep, dim=0), PAD_VAL=self.PAD_VAL).add_eos().add_sos()]
        t_normal = []
        for t in t_sep:
            s = SVGTensor.from_data(t, PAD_VAL=self.PAD_VAL)
            #                 print(1,len(s.commands))
            j = s.add_eos().add_sos().pad(seq_len=20 + 2)
            #                 print(2,len(s.commands))
            t_normal.append(j)

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
                res[arg] = torch.stack([t.args() for t in t_list])

        if "filling" in model_args:
            res["filling"] = torch.stack([torch.tensor(t.filling) for t in t_sep]).unsqueeze(-1)

        if "label" in model_args:
            res["label"] = label
        return res


