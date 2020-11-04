from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

from .data import AgentDataset
import src.argoverse.utils.baseline_utils as baseline_utils

DEFAULT_BATCH_SIZE = 32
DEFAULT_CACHE_SIZE = int(1e9)
DEFAULT_NUM_WORKERS = 4


class LyftDataModule(LightningDataModule):
    def __init__(
            self,
            args,
            model_config,
            **kwargs,
    ):
        super().__init__()
        self.args = args
        self.model_config = model_config
        self.data_dict = None
        self.train_data = None
        self.val_data = None
        self.test_data = None

    def setup(self, stage=None):
        if self.args.use_map and self.args.use_social:
            baseline_key = "map_social"
        elif self.args.use_map:
            baseline_key = "map"
        elif self.args.use_social:
            baseline_key = "social"
        else:
            baseline_key = "none"
        self.data_dict = baseline_utils.get_data(self.args, baseline_key)
        if stage == 'fit' or stage is None:
            # # Get PyTorch Dataset
            self.train_data = AgentDataset(model_args=self.model_config.model_args,
                                           max_num_groups=self.model_config.max_num_groups,
                                           max_seq_len=self.model_config.max_seq_len,
                                           data_dict=self.data_dict, args=self.args, mode="train")

            self.val_data = AgentDataset(model_args=self.model_config.model_args,
                                         max_num_groups=self.model_config.max_num_groups,
                                         max_seq_len=self.model_config.max_seq_len,
                                         data_dict=self.data_dict, args=self.args, mode="val")
        elif stage == 'test' or stage is None:
            self.test_data = AgentDataset(model_args=self.model_config.model_args,
                                          max_num_groups=self.model_config.max_num_groups,
                                          max_seq_len=self.model_config.max_seq_len,
                                          data_dict=self.data_dict, args=self.args, mode="test")

    def train_dataloader(self, batch_size=None, num_workers=None, shuffle=None):
        return DataLoader(self.train_data, batch_size=self.model_config.train_batch_size, shuffle=True,
                          num_workers=self.model_config.loader_num_workers)

    def val_dataloader(self, batch_size=None, num_workers=None, shuffle=None):
        return DataLoader(self.val_data, batch_size=self.model_config.val_batch_size, shuffle=True,
                          num_workers=self.model_config.loader_num_workers)

    def test_dataloader(self, batch_size=None, num_workers=None, shuffle=None):
        return DataLoader(self.test_data, batch_size=self.model_config.val_batch_size, shuffle=False,
                          num_workers=self.model_config.loader_num_workers)

