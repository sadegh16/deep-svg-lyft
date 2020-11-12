from collections import defaultdict, namedtuple
from typing import Any
import numpy as np
import pandas as pd
import bisect
import seaborn as sns
from pathlib import Path

from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import AgentDataset
from l5kit.rasterization import build_rasterizer

from .stats import *
import tqdm

Item = namedtuple('Item', ['value', 'payload'])


class TrajSampler:  # todo remove
    k: int = 500
    p = 0.1

    def __init__(self):
        self.extreme = list()
        self.sample = list()
        self.size = 0

    def __len__(self):
        return min(len(self.extreme), len(self.sample))

    def add(self, trj_idx: int, value: Any):
        self.size += 1
        item = Item(value, trj_idx)
        bisect.insort(self.extreme, item)
        if len(self.extreme) > self.k:
            del self.extreme[0]
        if len(self.sample) < self.k:
            self.sample.append(item)
        elif np.random.rand(1) < self.p:
            self.sample.append(item)
            self.sample.pop(0)

    def stats_dict(self):
        res = defaultdict(list)
        for kind in ['sample', 'extreme']:
            for item in getattr(self, kind):
                res['turn'].append(item.value[0])
                res['speed'].append(item.value[1])
                res['payload'].append(item.payload)
                res['kind'].append(kind)
        return res

    def as_dict(self, dataset, kind=None):
        res = defaultdict(list)
        for que in ['extreme', 'sample']:
            for value, payload in getattr(self, que):
                rastered = dataset[payload]
                del rastered['target_availabilities']
                del rastered['history_availabilities']
                rastered['speed'] = value[1]
                rastered['turn'] = value[0]
                rastered['extreme'] = (que == 'extreme')
                if kind is not None:
                    rastered['kind'] = kind
                res[que].append(rastered)
        return res

    def __repr__(self):
        return f'(s:{len(self.sample)}, e:{len(self.extreme)})'


class DataAnalyser:
    def __init__(self, data_root: str, config_path: str, split: str, show_progress=True, turn_thresh=3.,
                 speed_thresh=0.5, static_thresh=1., output_folder='preprocess', autosave=True, cache_size=1e9):
        self.autosave = autosave
        self.show_progress = show_progress
        self.turn_thresh = turn_thresh
        self.speed_thresh = speed_thresh
        self.static_thresh = static_thresh
        self.split = split
        self.config = load_config_data(config_path)
        self.output_folder = output_folder

        self.data_manager = LocalDataManager(data_root)
        self.rasterizer = build_rasterizer(self.config, self.data_manager)
        self.data_zarr = ChunkedDataset(self.data_manager.require(split)).open(cache_size_bytes=int(cache_size))
        self.dataset = AgentDataset(self.config, self.data_zarr, self.rasterizer)

        self.data = defaultdict(list)
        self.junk = defaultdict(list)

        self.progress = None

    def save(self, folder_name=None):
        folder_name = folder_name or self.output_folder
        Path(folder_name).mkdir(parents=True, exist_ok=True)
        print('saving:')
        for item in ['data', 'junk']:
            print('\t', item)
            data = getattr(self, item)
            # filling last columns if necessary
            max_len = max([len(value) for idx, value in data.items()])
            for idx in data:
                if len(data[idx]) != max_len:
                    data[idx].append(None)
            df = pd.DataFrame(data)
            df.to_csv(f'{self.output_folder}/{self.split.replace("/", "_").replace(".zarr", "")}_{item}.csv')

    def process(self, start=0, end=None, step=10):
        end = end or len(self.dataset)
        idxs = range(start, end, step)
        self.progress = idxs if not self.show_progress else tqdm.tqdm(idxs, ascii=True)
        try:
            for idx in self.progress:
                traj = self.dataset[idx]
                filter_res = filter_traj(traj, self.static_thresh)
                if filter_res:
                    self.junk['type'].append(filter_res[0])
                    self.junk['value'].append(float(filter_res[1]))
                    self.junk['idx'].append(int(idx))
                    continue
                stats = traj_stat(traj)
                self.data['idx'].append(idx)
                for key, value in stats:
                    self.data[key].append(float(value))
        except Exception as e:
            print('stopping', e)
        finally:
            if self.autosave:
                self.save(self.output_folder)
