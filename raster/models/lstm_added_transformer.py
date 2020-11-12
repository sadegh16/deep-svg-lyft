import torch

from raster.models.model_trajectory import ModelTrajectory
from raster.models.template import RasterModel


class LSTMTransformer(RasterModel):
    def __init__(self,model_config,modes,data_config=None,future_len=None, history_num=None):
        super().__init__(config=data_config, modes=modes, future_len=future_len, in_channels=history_num)
        print("history_num",history_num)
        print("out_dim",self.out_dim)
        self.transformer = ModelTrajectory(model_cfg=model_config)
        self.history_LSTM= torch.nn.Sequential(torch.nn.LSTM(history_num, 32),)
        self.linear = torch.nn.Sequential(torch.nn.Linear(64+128, 128),
                                          torch.nn.ReLU(),)
        self.final_LSTM = torch.nn.LSTM(128, self.out_dim)

    def _forward(self, x):
        scene,history = x
        s = self.transformer(scene)
        print(history.shape)
        print(torch.flatten(history,start_dim=1).unsqueeze(1).shape)
        print(history.permute(0, 2, 1))
        h ,hidden= self.history_LSTM(history.permute(0, 2, 1))
        print(h.shape)
        z = torch.cat((s,h.flatten(start_dim=1)), dim=1)
        l = self.linear(z)
        o,h = self.final_LSTM(l.unsqueeze(1))
        print("ouut",o.shape,o.squeeze(1).shape)
        return o.squeeze(1)

