from argparse import ArgumentParser
import typing as th
from raster.utils import boolify, CachedDataset
from torch.utils.data import DataLoader, Subset, random_split, Dataset
from pytorch_lightning import LightningDataModule
import pandas as pd
import numpy as np

from l5kit.data import LocalDataManager, ChunkedDataset
import functools

from deepsvg.config import _Config
import torch.nn as nn
from l5kit.rasterization import build_rasterizer,my_build_rasterizer
import argparse
import importlib
from l5kit.configs import load_config_data
from .data import AgentDataset

from deepsvg.utils import Stats, TrainVars, Timer
import torch
from deepsvg import utils
from datetime import datetime
from tensorboardX import SummaryWriter
from deepsvg.utils.stats import SmoothedValue
import os

from torch.utils.data.dataloader import default_collate
from collections import defaultdict
import nonechucks as nc
import pandas as pd

DEFAULT_BATCH_SIZE = 32
DEFAULT_CACHE_SIZE = int(1e9)
DEFAULT_NUM_WORKERS = 4



def my_collate(batch):
    "Puts each data field into a tensor with outer dimension batch size"
    # print(batch)
    # batch = filter(lambda x:x is not None, batch)
    batch = list(filter(None, batch))
    # print(batch)
    if len(batch) > 0:
        return default_collate(batch)
    else:
        return

class IndexedDataset(Dataset):

    def __init__(self, data, indexes, transform=None):
        self.data = data
        self.indexes = indexes

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        return self.data[idx], self.indexes[idx]


class LyftDataModule(LightningDataModule):
    def __init__(
            self,
            data_root: str,
            config: dict,
            model_config,
            # train
            train_split: str = None,
            train_batch_size: str = None,
            train_shuffle: bool = None,
            train_num_workers: int = None,
            train_idxs: th.Any = None,
            # validation
            val_proportion: float = None,
            val_split: str = None,
            val_batch_size: str = None,
            val_shuffle: bool = None,
            val_num_workers: int = None,
            val_idxs: th.Any = None,
            # test
            test_split: str = None,
            test_batch_size: str = None,
            test_shuffle: bool = None,
            test_num_workers: int = None,
            test_idxs: th.Any = None,
            test_mask_path: str = None,
            # overall options
            cache_size: float = 1e9,
            raster_cache_size: float = 0,
            **kwargs,
    ):
        super().__init__()
        self.data_root = data_root
        self.config = config
        self.model_config = model_config
        print('initializing up data module\n\t*root:', data_root, '\t*cache:', cache_size, '\t*raster-cache:',
              raster_cache_size)

        # train
        self.train_split = train_split or config['train_dataloader']['split']
        self.train_batch_size = train_batch_size or config['train_dataloader']['batch_size']
        self.train_shuffle = train_shuffle if train_shuffle is not None else config['train_dataloader'].get(
            'shuffle', True)
        self.train_num_workers = train_num_workers if train_num_workers is not None else config['train_dataloader'].get(
            'num_workers', DEFAULT_NUM_WORKERS)
        self.train_idxs = None if train_idxs is None else pd.read_csv(train_idxs)['idx']
        print('train\n\t*split:', self.train_split, '*batch_size:', self.train_batch_size, '*shuffle:',
              self.train_shuffle, '*num_workers:', self.train_num_workers, '*idxs:', train_idxs)
        # val
        self.val_proportion = val_proportion
        self.val_split = val_split or config.get('val_dataloader', dict()).get('split', None)
        self.val_batch_size = val_batch_size or config.get('val_dataloader', dict()).get(
            'batch_size', self.train_batch_size)
        self.val_shuffle = val_shuffle if val_shuffle is not None else config.get('val_dataloader', dict()).get(
            'shuffle', False)
        self.val_num_workers = val_num_workers if val_num_workers is not None else config.get(
            'val_dataloader', dict()).get('num_workers', DEFAULT_NUM_WORKERS)
        self.val_idxs = None if val_idxs is None else pd.read_csv(val_idxs)['idx']
        assert self.val_split is not None or self.val_proportion is not None, \
            'validation proportion should not be None'
        print('val\n\t*split:', self.val_split, '*batch_size:', self.val_batch_size, '*shuffle:', self.val_shuffle,
              '*num_workers:', self.val_num_workers, '*idxs:', val_idxs, '*proportion:', self.val_proportion)
        # test
        self.test_split = test_split or config.get('test_dataloader', dict()).get('split', None)
        self.test_batch_size = test_batch_size or config.get('test_dataloader', dict()).get(
            'batch_size', None)
        self.test_shuffle = test_shuffle if test_shuffle is not None else config.get('test_dataloader', dict()).get(
            'shuffle', False)
        self.test_num_workers = test_num_workers if test_num_workers is not None else config.get(
            'test_dataloader', dict()).get('num_workers', DEFAULT_NUM_WORKERS)
        self.test_idxs = None if test_idxs is None else pd.read_csv(test_idxs)['idx']
        self.test_mask = None
        if test_mask_path:
            self.test_mask = np.load(test_mask_path)["arr_0"]
        print('test\n\t*split:', self.test_split, '*batch_size:', self.test_batch_size, '*shuffle:', self.test_shuffle,
              '*num_workers:', self.test_num_workers, '*idxs:', test_idxs)

        # attributes
        self.cache_size = int(cache_size)
        self.raster_cache_size = int(raster_cache_size) if raster_cache_size else 0
        self.data_manager = None
        self.rasterizer = None
        self.train_data = None
        self.val_data = None
        self.test_data = None

    def setup(self, stage=None):
        if self.data_manager is None:
            self.data_manager = LocalDataManager(self.data_root)
        if self.rasterizer is None:
            self.rasterizer = build_rasterizer(self.config, self.data_manager)
        if stage == 'fit' or stage is None:
            train_zarr = ChunkedDataset(self.data_manager.require(self.train_split)).open(
                cache_size_bytes=int(self.cache_size))
            train_data = AgentDataset(data_cfg=self.config, zarr_dataset = train_zarr, rasterizer = self.rasterizer,
                                      model_args=self.model_config.model_args, max_num_groups=self.model_config.max_num_groups,
                                      max_seq_len=self.model_config.max_seq_len)

            if self.train_idxs is not None:
                train_data = Subset(train_data, self.train_idxs)
            if self.val_split is None or self.val_split == self.train_split:
                tl = len(train_data)
                vl = int(tl * self.val_proportion)
                self.train_data, self.val_data = random_split(train_data, [tl - vl, vl])
            else:
                self.train_data = train_data
                val_zarr = ChunkedDataset(self.data_manager.require(self.val_split)).open(
                    cache_size_bytes=int(self.cache_size))
                self.val_data = AgentDataset(data_cfg=self.config, zarr_dataset = val_zarr, rasterizer = self.rasterizer,
                                             model_args=self.model_config.model_args, max_num_groups=self.model_config.max_num_groups,
                                             max_seq_len=self.model_config.max_seq_len)
                if self.val_idxs is not None:
                    self.val_data = Subset(self.val_data, self.val_idxs)
            if self.raster_cache_size:
                self.train_data = CachedDataset(self.train_data, self.raster_cache_size)
                self.val_data = CachedDataset(self.val_data, self.raster_cache_size)
        if stage == 'test' or stage is None:
            test_zarr = ChunkedDataset(self.data_manager.require(self.test_split)).open(
                cache_size_bytes=int(self.cache_size))
            if self.test_mask is not None:
                test_data = AgentDataset(data_cfg=self.config, zarr_dataset = test_zarr, rasterizer = self.rasterizer,
                                         model_args=self.model_config.model_args, max_num_groups=self.model_config.max_num_groups,
                                         max_seq_len=self.model_config.max_seq_len,agents_mask=self.test_mask)
            else:
                test_data = AgentDataset(data_cfg=self.config, zarr_dataset = test_zarr, rasterizer = self.rasterizer,
                                         model_args=self.model_config.model_args, max_num_groups=self.model_config.max_num_groups,
                                         max_seq_len=self.model_config.max_seq_len)
            if self.test_idxs is not None:
                test_data = Subset(test_data, self.test_idxs)
            else:
                self.test_idxs = np.arange(start=1, stop=len(test_data) + 1)
            self.test_data = IndexedDataset(test_data, self.test_idxs)

    def _get_dataloader(self, name: str, batch_size=None, num_workers=None, shuffle=None):
        batch_size = batch_size or getattr(self, f'{name}_batch_size')
        num_workers = num_workers or getattr(self, f'{name}_num_workers')
        shuffle = shuffle or getattr(self, f'{name}_shuffle')
        return DataLoader(
            getattr(self, f'{name}_data'), batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)

    def train_dataloader(self, batch_size=None, num_workers=None, shuffle=None):
        return DataLoader(self.train_data, batch_size=self.model_config.train_batch_size, shuffle=True,
                          num_workers=self.model_config.loader_num_workers)

    def val_dataloader(self, batch_size=None, num_workers=None, shuffle=None):
        return DataLoader(self.val_data, batch_size=self.model_config.val_batch_size, shuffle=True,
                          num_workers=self.model_config.loader_num_workers)

    def test_dataloader(self, batch_size=None, num_workers=None, shuffle=False):
        return DataLoader(self.test_data, batch_size=self.model_config.val_batch_size, shuffle=False,
                          num_workers=self.model_config.loader_num_workers)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--data-root', required=True, type=str, help='lyft dataset root folder path')
        parser.add_argument('--cache-size', type=float, default=1e9, help='cache size for each data split')
        parser.add_argument('--raster-cache-size', type=float, default=0., help='cache size for each data split')

        parser.add_argument('--train-split', type=str, default=None, help='train split scenes')
        parser.add_argument('--train-batch-size', type=int, help='train batch size of dataloaders')
        parser.add_argument('--train-shuffle', type=boolify, default=None, help='train dataloader shuffle data')
        parser.add_argument('--train-num-workers', type=int, default=None, help='train dataloader number of workers')
        parser.add_argument('--train-idxs', type=str, default=None, help='train data indexes')

        parser.add_argument('--val-proportion', type=float, default=None, help='validation proportion in data')
        parser.add_argument('--val-split', type=str, default=None, help='validation split scenes')
        parser.add_argument('--val-batch-size', type=int, default=None, help='validation batch size of dataloaders')
        parser.add_argument('--val-shuffle', type=boolify, default=None, help='validation dataloader shuffle data')
        parser.add_argument('--val-num-workers', type=int, default=None, help='validation dataloader number of workers')
        parser.add_argument('--val-idxs', type=str, default=None, help='validation data indexes')

        parser.add_argument('--test-split', type=str, default=None, help='test split scenes')
        parser.add_argument('--test-batch-size', type=int, help='test batch size of dataloaders')
        parser.add_argument('--test-shuffle', type=boolify, default=None, help='test dataloader shuffle data')
        parser.add_argument('--test-num-workers', type=int, default=None, help='test dataloader number of workers')
        parser.add_argument('--test-idxs', type=str, default=None, help='test data indexes')
        return parser
