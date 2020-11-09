import torch
import torchvision as tv
import typing as th
from .template import RasterModel
# from deepsvg.model.modified_model import SVGTransformer
# from deepsvg.model.modified_model_between import SVGTransformer
# class ModelTrajectory(torch.nn.Module):
#     def __init__(self,model_cfg,dim_z=128):
#         super().__init__()
#         self.model_cfg = model_cfg
#         self.model_cfg.model_cfg.dim_z = dim_z
#         self.model_cfg.model_cfg.max_num_groups = self.model_cfg.max_num_groups
#         self.model_cfg.model_cfg.max_seq_len = self.model_cfg.max_seq_len
#         self.model = SVGTransformer(self.model_cfg.model_cfg)
#         print(self.model.encoder)

#     def forward(self,x):
#         commands_enc,args_enc, commands_dec, args_dec,params , encode_mode = x
#         return self.model(commands_enc,args_enc, commands_dec, args_dec,params=params , encode_mode=encode_mode).squeeze(0).squeeze(0)



class ModelTrajectory(RasterModel):
    def __init__(self, model_cfg, data_config, modes=1, future_len=None, in_channels=None, pretrained=True,model_type = None):
        super().__init__(config=None, modes=modes, future_len=future_len, in_channels=in_channels)
        self.model_cfg = model_cfg
        self.model_cfg.model_cfg.args_dim = 1024
        self.model_cfg.model_cfg.dim_z = self.out_dim
        self.model_cfg.model_cfg.max_num_groups = self.model_cfg.max_num_groups
        self.model_cfg.model_cfg.max_seq_len = self.model_cfg.max_seq_len
        self.model_type = model_type
        if model_type == 4:
            from deepsvg.model.modified_model import SVGTransformer
            self.model = SVGTransformer(self.model_cfg.model_cfg)
        if model_type == 5:
            from deepsvg.model.modified_model_between import SVGTransformer
            self.model = SVGTransformer(self.model_cfg.model_cfg)
        if model_type == 1:
            from deepsvg.model.model import SVGTransformer
            self.model = SVGTransformer(self.model_cfg.model_cfg)
        if model_type == 6:
            ##TODO
            # Import Right SVGTransformer Vahid
            self.model = SVGTransformer(self.model_cfg.model_cfg)



    def _forward(self, x):
        if self.model_type == 4 or self.model_type == 5:
            commands_enc, args_enc, commands_dec, args_dec,history,params, encode_mode = x
            return self.model(commands_enc, args_enc, commands_dec, args_dec,history=history, params=params,
                              encode_mode=encode_mode)
        if self.model_type == 6:
            commands_enc, args_enc, commands_dec, args_dec,commands_his,args_his ,params, encode_mode = x
            return self.model(commands_enc, args_enc, commands_dec, args_dec,commands_his=commands_his,args_his=args_his, params=params,
                              encode_mode=encode_mode)
        if self.model_type == 1:
            commands_enc, args_enc, commands_dec, args_dec,params, encode_mode = x
            return self.model(commands_enc, args_enc, commands_dec, args_dec,params=params,
                              encode_mode=encode_mode)