import argparse
import datetime
import logging
import time
import random
import pdb
import numpy as np
import torch
import torch.nn.functional as F
import trajnettools
import os
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import torch
import torchvision
from deepsvg.svglib.svg import SVG
from deepsvg.difflib.tensor import SVGTensor
from deepsvg.svglib.geom import Point, Angle
from deepsvg.svg_takeaways import get_data
from deepsvg.config import _Config
import importlib 
import pickle
        
baseadd_load = './svg_imgs/'

baseadd_save = './svg_pkl_train_seq30_simplify0.05_singlepath/'

for i in os.listdir(baseadd_load):
    try:          
        road = SVG.load_svg(baseadd_load+i)
        road2 = road.copy().zoom(np.array([0.1, 0.1]), Point(23.4,23.4)).normalize()
        #To remove all paths except the longest one
        '''g_l = [ii.total_len() for ii in road2.svg_path_groups]
        g_l_max = max(g_l)
        rlist = [pa for ind, pa in enumerate(road2.svg_path_groups) if g_l[ind]<0.9*g_l_max]
        road2.rmv_path_groups(rlist)'''
        road3 = road2.copy().zoom(0.9).simplify_heuristic(tolerance=0.05)
        data_ = {
            "tensors": [SVG.to_tensor(road3, concat_groups=False)],
            "fillings": [road3.to_fillings()],
        }
        data_ = get_data(data_["tensors"], data_["fillings"])
        if(data_==None):
            continue
        output = open( baseadd_save+i.replace('svg','pkl'), 'wb')
        pickle.dump(data_, output)
        output.close()
    except: 
        pass
       