import torch
import functools
import copy
import types
from deepsvg.svglib.svg import SVG, Bbox
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




cmd_codes = dict(m=0, l=1, c=2, a=3, EOS=4, SOS=5, z=6)
COLOR_IDXS = slice(1,6)


def linear_cmd_to_tensor(cmd_index, end_position: tuple, start_position: tuple = None, pad=-1):
    start_pos = start_position if start_position is not None else (0, 0)
    return torch.tensor(
        [cmd_index, *([pad] * 5), start_pos[0], start_pos[1], *([pad] * 4), end_position[0], end_position[1]])


def linear_path_to_tensor(path, pad=-1):
    return torch.stack([linear_cmd_to_tensor(cmd_codes['m'], path[0], pad=pad)] + [
        linear_cmd_to_tensor(cmd_codes['l'], path[i], path[i - 1], pad=pad) for i in range(1, len(path))])


def apply_colors(paths, colors, idxs: slice = COLOR_IDXS):
    colors = colors if colors is not None else [-1] * len(paths)
    for i in range(len(paths)):
        paths[i][:, idxs] = colors[i]
    return paths





class BaseDataset(Dataset):
    def __init__(self, data_dict: Dict[str, Any], args: Any, mode: str):
        """Initialize the Dataset.

        Args:
            data_dict: Dict containing all the data
            args: Arguments passed to the baseline code
            mode: train/val/test mode

        """
        self.data_dict = data_dict
        self.args = args
        self.mode = mode

        # Get input
        self.input_data = data_dict["{}_input".format(mode)]
        if mode != "test":
            self.output_data = data_dict["{}_output".format(mode)]
        self.data_size = self.input_data.shape[0]

        # Get helpers
        self.helpers = self.get_helpers()
        self.helpers = list(zip(*self.helpers))
        
        
        
        from argoverse.map_representation.map_api import ArgoverseMap

        self.avm = ArgoverseMap()
        self.mf=MapFeaturesUtils()
        

    def __len__(self):
        """Get length of dataset.

        Returns:
            Length of dataset

        """
        return self.data_size

    def __getitem__(self, idx: int
                    ) -> Tuple[torch.FloatTensor, Any, Dict[str, np.ndarray]]:
        """Get the element at the given index.

        Args:
            idx: Query index

        Returns:
            A list containing input Tensor, Output Tensor (Empty if test) and viz helpers. 

        """
        helper=self.helpers[idx]
        cnt_lines,img,cnt_lines_norm=self.mf.get_candidate_centerlines_for_trajectory(
                        helper[0] if self.mode != "test"  else  helper[0][:20],
                        yaw_deg=helper[5],centroid=helper[0][0],
                        city_name=helper[1][0],avm=self.avm,
            viz=True,
            seq_len = 80,
            max_candidates=10,
            )
        
        
        res = torch.cat([linear_path_to_tensor(path, -1) for path in cnt_lines_norm], 0)

        return {"history_positions": torch.FloatTensor(self.input_data[idx]),
                "target_positions": torch.empty(1) if self.mode == "test" else torch.FloatTensor(self.output_data[idx]),
                "path":res,
               }
    
    def get_helpers(self) -> Tuple[Any]:
        """Get helpers for running baselines.

        Returns:
            helpers: Tuple in the format specified by LSTM_HELPER_DICT_IDX

        Note: We need a tuple because DataLoader needs to index across all these helpers simultaneously.

        """
        helper_df = self.data_dict[f"{self.mode}_helpers"]
        candidate_centerlines = helper_df["CANDIDATE_CENTERLINES"].values
#         print("ss",candidate_centerlines)
        candidate_nt_distances = helper_df["CANDIDATE_NT_DISTANCES"].values
        xcoord = np.stack(helper_df["FEATURES"].values
                          )[:, :, config.FEATURE_FORMAT["X"]].astype("float")
        ycoord = np.stack(helper_df["FEATURES"].values
                          )[:, :, config.FEATURE_FORMAT["Y"]].astype("float")
        centroids = np.stack((xcoord, ycoord), axis=2)
        _DEFAULT_HELPER_VALUE = np.full((centroids.shape[0]), None)
        city_names = np.stack(helper_df["FEATURES"].values
                              )[:, :, config.FEATURE_FORMAT["CITY_NAME"]]
        seq_paths = helper_df["SEQUENCE"].values
        translation = (helper_df["TRANSLATION"].values
                       if self.args.normalize else _DEFAULT_HELPER_VALUE)
        rotation = (helper_df["ROTATION"].values
                    if self.args.normalize else _DEFAULT_HELPER_VALUE)

        use_candidates = self.args.use_map and self.mode == "test"

        candidate_delta_references = (
            helper_df["CANDIDATE_DELTA_REFERENCES"].values
            if self.args.use_map and use_candidates else _DEFAULT_HELPER_VALUE)
        delta_reference = (helper_df["DELTA_REFERENCE"].values
                           if self.args.use_delta and not use_candidates else
                           _DEFAULT_HELPER_VALUE)

        helpers = [None for i in range(len(config.LSTM_HELPER_DICT_IDX))]

        # Name of the variables should be the same as keys in LSTM_HELPER_DICT_IDX
        for k, v in config.LSTM_HELPER_DICT_IDX.items():
            helpers[v] = locals()[k.lower()]

        return tuple(helpers)





class SVGDataset(torch.utils.data.Dataset):
    def __init__(self,data_dict, args, mode,map_type,svg_args,model_args, max_num_groups, max_seq_len,
                 max_total_len=None, filter_uni=None, filter_platform=None,
                 filter_category=None, train_ratio=1.0, PAD_VAL=-1,csv_path=None):

        super().__init__()
        
        self.svg_args = svg_args
        self.svg = map_type.startswith('svg_')
        self.svg_cmds = self.svg_args.get('return_cmds', True)
        self.tensor = map_type.startswith('tensor_')
        self.data = BaseDataset(data_dict, args, mode)

        self.MAX_NUM_GROUPS = max_num_groups
        self.MAX_SEQ_LEN = max_seq_len
        self.MAX_TOTAL_LEN = max_total_len

        if max_total_len is None:
            self.MAX_TOTAL_LEN = max_num_groups * max_seq_len


        self.model_args = model_args

        self.PAD_VAL = PAD_VAL



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
            svg = AgentDataset._augment(svg, mean=mean)
        if numericalize:
            return svg.numericalize(256)
        return svg

    def get(self, idx=0, model_args=None, random_aug=True, id=None, svg: SVG = None):
        item = self.data[idx]
        if self.svg and self.svg_cmds:
            tens = self.simplify(SVG.from_tensor(item['path'])).split_paths().to_tensor(concat_groups=False)
            # svg = apply_colors(tens, item['path_type'])
            del item['path']
            del item['path_type']
            item['image'] = self.get_data(idx,tens, None, model_args=model_args, label=None)
            if item['image'] is None:
                return
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