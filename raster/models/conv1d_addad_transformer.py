import torch

from raster.models.model_trajectory import ModelTrajectory
from raster.models.template import RasterModel

class ConvTransformer(RasterModel):
    def __init__(self,model_config,modes,model_type = 0, data_config=None,future_len=None, history_num=None):
        super().__init__(config=data_config, modes=modes, future_len=future_len, in_channels=history_num)
        #         print("history_num",history_num)
        #         print("out_dim",self.out_dim)
        self.model_type = model_type
        self.transformer = ModelTrajectory(model_cfg=model_config)
        self.history_mlp= torch.nn.Sequential(torch.nn.Conv1d(in_channels=1,out_channels=4,kernel_size=4,stride=2,padding=0),
                                              torch.nn.ReLU(),
                                              torch.nn.Conv1d(in_channels=4,out_channels=8,kernel_size=4,stride=2,padding=0),)
        self.final_mlp = torch.nn.Sequential(torch.nn.Linear(64+128, 128),
                                             torch.nn.ReLU(),
                                             torch.nn.Linear(128, 128),
                                             torch.nn.ReLU(),
                                             torch.nn.Linear(128, self.out_dim),)

    def _forward(self, x):
        scene,history = x
        b = history.shape[0]
        #         print("scene",scene[0].shape,"history",history.shape)
        s = self.transformer(scene)
        #         print(1,s.shape)
        h = self.history_mlp(torch.flatten(history,start_dim=1).unsqueeze(1)).reshape((b,-1))
        print(2,h.shape)
        z = torch.cat((s,h), dim=1)
        #         print(3,z.shape)
        return self.final_mlp(z)
