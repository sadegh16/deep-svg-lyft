"""Command line tool to train an LSTM model."""

import argparse
import datetime
import logging
import time
import random
import pdb
import numpy as np
import torch
import torch.nn.functional as F
import os
from . import augmentation
import pickle
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
from torch.utils.tensorboard import SummaryWriter

def train_batch():
    baseadd = '/home/hossein/project/SVG/trajnetbaselines/segmentedImgs/svg_pkl_train_seq30_simplify0.05_singlepath/'    
    self.optimizer.zero_grad()    
    data = {
            "commands" :torch.zeros([self.batch_size, 8, seq_len+2]),
            "args": torch.zeros([self.batch_size, 8, seq_len+2, 11]),
             }

        for i in range(self.batch_size):    
            with open(baseadd+file_name[i]+'_'+str(id[i])+'.pkl', 'rb') as handle: 
                data_ = pickle.load(handle)
            data['commands'][i] = data_['commands']
            data['args'][i] = data_['args']
                
            # pass cases that lead to divergance of the model
            if ((data['args'][i]<-1).any()):
                print("some bugs are seen") 
                return 0
                        
        model_args = [data[arg].to(self.device) for arg in ['commands', 'args', 'commands', 'args']]
        labels = data["label"].to(device) if "label" in data else None
        params_dict = {}
        self.svg_model.train()            
        scene_features = self.svg_model(*model_args, params=params_dict, encode_mode=True).squeeze(0).squeeze(0)

    prediction_v = self.model(obs=observed,  scene_features=scene_features)
    loss = torch.sum(torch.sum(torch.sum(abs(prediction_v - targets[:,-self.n_pred:,]),dim=1),dim=1)*mask)#L1
    loss.backward()
    self.optimizer.step()   
    return loss.item()

def main(epochs=35):
    

    model = MLP_traj_scene(n_obs=args.n_obs, n_pred = args.n_pred, device = args.device)

    
    weight_decay = 1e-2
    num_epochs_reduce_lr = 15
    
    cfg = importlib.import_module('configs.deepsvg.hierarchical_ordered').Config()
    svg_model = cfg.make_model()

    trainer = Trainer(timestamp, model, optimizer=optimizer, device=args.device, n_obs=args.n_obs, n_pred = args.n_pred, lr_scheduler=lr_scheduler, scene_mode=args.scene_mode,svg_model=svg_model)
    
    
    optimizer = torch.optim.Adam(list(model.parameters()) + list(svg_model.parameters()), lr=args.lr, weight_decay=weight_decay) #weight_decay=1e-4
    