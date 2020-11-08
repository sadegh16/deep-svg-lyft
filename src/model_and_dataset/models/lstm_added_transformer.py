import torch

from .model_trajectory_subset import ModelTrajectory
from .template import RasterModel

class LSTMTransformer(RasterModel):
    def __init__(self,model_config,modes,data_config=None,future_len=None, history_num=None):
        super().__init__(config=data_config, modes=modes, future_len=future_len, in_channels=history_num)
        print("history_num",history_num)
        print("out_dim",self.out_dim)
        self.transformer = ModelTrajectory(model_cfg=model_config)
        self.history_LSTM= torch.nn.Sequential(torch.nn.LSTM(history_num, 32),)
        self.linear = torch.nn.Sequential(torch.nn.Linear(64+128, 128),
                                          torch.nn.ReLU(),
                                          torch.nn.Linear(128, 128),
                                          torch.nn.ReLU(),
                                          torch.nn.Linear(128, self.out_dim),)
        self.final_LSTM = torch.nn.LSTM(128, self.out_dim)

    def _forward(self, x):
        scene,history = x
        s = self.transformer(scene)
        h ,hidden= self.history_LSTM(history.permute(0, 2, 1))
        z = torch.cat((s,h.flatten(start_dim=1)), dim=1)
        l = self.linear(z)
        return l

