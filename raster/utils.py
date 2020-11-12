import argparse
from torch.utils.data import Dataset
import functools


class CachedDataset(Dataset):
    def __init__(self, dataset, cache_size=16384):
        self.dataset = dataset
        self.get_item = functools.lru_cache(cache_size)(self.get_item)

    def __len__(self):
        return len(self.dataset)

    def get_item(self, i):
        return self.dataset[i]

    def __getitem__(self, i):
        return self.get_item(i)


def boolify(s):
    if s == 'True' or s == 'true' or s == 'yes' or s == 'Yes':
        return True
    if s == 'False' or s == 'false' or s == 'no' or s == 'No':
        return False
    raise ValueError("cast error")


def dictify(s: str):
    if ':' not in s:
        raise ValueError("cast error")
    else:
        res = dict()
        pairs = s.split(',')
        for pair in pairs:
            key, val = pair.split(':')
            res[key] = auto_cast(val)
        return res


def auto_cast(s):
    for fn in (dictify, boolify, int, float):
        try:
            return fn(s)
        except ValueError:
            pass
    return s


# create a keyvalue class
class KeyValue(argparse.Action):
    # Constructor calling
    def __call__(self, parser, namespace,
                 values, option_string=None):
        setattr(namespace, self.dest, dict())

        for value in values:
            key, value = value.split('=')
            if key in getattr(namespace, self.dest):
                if not isinstance(getattr(namespace, self.dest)[key], list):
                    getattr(namespace, self.dest)[key] = [getattr(namespace, self.dest)[key]]
                getattr(namespace, self.dest)[key].append(auto_cast(value))
            else:
                getattr(namespace, self.dest)[key] = auto_cast(value)
