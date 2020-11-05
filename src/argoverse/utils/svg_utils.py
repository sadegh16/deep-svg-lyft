from src.lyft.utils import linear_path_to_tensor

import torch
import functools
import copy
import types
from deepsvg.svglib.svg import SVG, Bbox
from deepsvg.config import _Config
from deepsvg.difflib.tensor import SVGTensor
from deepsvg.svglib.svg import SVG
from deepsvg.svglib.geom import Point
import numpy as np
import math
import torch
import torch.utils.data
import random
from typing import List, Union, Dict, Any,Tuple
import pandas as pd
import os
import pickle
from .baseline_config import *
from .map_features_utils import MapFeaturesUtils
from .transform import *
import csv

import math



class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, data_dict: Dict[str, Any], args: Any, mode: str,):
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
        
        middle_dir=mode if mode!="test" else  "test_obs"
        self.root_dir="/work/vita/sadegh/argo/argoverse-api/"+middle_dir+"/data"
        
        from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader

        ##set root_dir to the correct path to your dataset folder
        self.afl = ArgoverseForecastingLoader(self.root_dir)
        
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
        
        
        ############################# find lanes
        cnt_lines,img,cnt_lines_norm,world_to_image_space=self.mf.get_candidate_centerlines_for_trajectory(
                        helper[0] if self.mode != "test"  else  helper[0][:20],
                        yaw_deg=helper[5],
                        city_name=helper[1][0],avm=self.avm,
            viz=True,
            seq_len = 50,
            max_candidates=10,
            )
        #############################
#         print(helper)
        
        traj= helper[0] if self.mode != "test"  else  helper[0][:20]
        traj = transform_points(traj-helper[0][19], yaw_as_rotation33(math.pi*helper[5]/180))
        ############################## find agents history
        agents_history= self.get_agents(idx,world_to_image_space,helper[0][19],)
#         print("agents_history len ",type(agents_history[0]))
        ##############################
        
        ############################## history positions 
        ego_world_history=helper[0][:20]
        history_xy = transform_points(ego_world_history, world_to_image_space)
        history_xy=crop_tensor(history_xy, (224,224))
#         print(len(history_xy))
        ##############################
#         print(len(agents_history))
        path_type=[0]*len(cnt_lines_norm)+[1]
#         print(len(cnt_lines_norm))
        path_stacked_up=cnt_lines_norm+[history_xy]
#         print(len(path_stacked_up))
        
        res = torch.cat([linear_path_to_tensor(path, -1) for path in path_stacked_up], 0)
#         print(len(res))
        

#         res = torch.cat([linear_path_to_tensor(path, -1) for path in cnt_lines_norm], 0)



#         history_positions= torch.FloatTensor(transform_points(helper[0], world_to_image_space)),
#         target_positions= torch.FloatTensor(self.input_data[idx]),
        return {"history_positions": torch.FloatTensor(traj[:self.args.obs_len]),
                "target_positions": torch.empty(1) if self.mode == "test" else torch.FloatTensor(traj[self.args.obs_len:]),
                "path":res,
                "path_type":path_type,
#                 "image":img.transpose(2, 0, 1),
                "base_image":img.transpose(2, 0, 1),
                
                "centroid":helper[0][19],
                "yaw_deg":helper[5],
                "seq_id":helper[8],
#                 "cnt_lines_norm":cnt_lines_norm,
                "world_to_image_space":world_to_image_space,
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
                          )[:, :, FEATURE_FORMAT["X"]].astype("float")
        ycoord = np.stack(helper_df["FEATURES"].values
                          )[:, :, FEATURE_FORMAT["Y"]].astype("float")
        centroids = np.stack((xcoord, ycoord), axis=2)
        _DEFAULT_HELPER_VALUE = np.full((centroids.shape[0]), None)
        city_names = np.stack(helper_df["FEATURES"].values
                              )[:, :, FEATURE_FORMAT["CITY_NAME"]]
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

        helpers = [None for i in range(len(LSTM_HELPER_DICT_IDX))]

        # Name of the variables should be the same as keys in LSTM_HELPER_DICT_IDX
        for k, v in LSTM_HELPER_DICT_IDX.items():
            helpers[v] = locals()[k.lower()]

        return tuple(helpers)

    def get_agents(self,index,world_to_image_space,centroid) :
        """Get agents

        """
        helper_df = self.data_dict[f"{self.mode}_helpers"]
        seq_id=helper_df.iloc[index,0]
        seq_path = f"{self.root_dir}/{seq_id}.csv"
#         print(seq_path)
        df=self.afl.get(seq_path).seq_df
        frames = df.groupby("TRACK_ID")

        res=[]
#         print(len(frames))
        # Plot all the tracks up till current frame
        num_selected=0
        for group_name, group_data in frames:
            object_type = group_data["OBJECT_TYPE"].values[0]
            
            
#             print(group_data[["X","Y"]].values.shape).
            cor_xy = group_data[["X","Y"]].to_numpy()
            if np.linalg.norm(centroid-cor_xy[-1])>40:
                continue
            cor_xy = transform_points(cor_xy, world_to_image_space)
#             print(cor_xy.shape)
            cropped_vector=crop_tensor(cor_xy, (224,224))
#             print(cropped_vector.shape)
            
            if len(cropped_vector)>1:
                res.append(cropped_vector)
                num_selected+=1 
            if num_selected>=15:
                break
#         print(num_selected)
        return res

