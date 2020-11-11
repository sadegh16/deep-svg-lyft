import torch
import torchvision as tv
import typing as th
from .template import RasterModel
# from deepsvg.model.modified_model_between import SVGTransformer
# from deepsvg.model.model import SVGTransformer



class ModelTrajectory(torch.nn.Module):
    def __init__(self,model_cfg,model_type=0,dim_z=128,mlp=None):
        super().__init__()
        self.model_type = model_type
        self.model_cfg = model_cfg
        self.model_cfg.model_cfg.dim_z = dim_z
        self.model_cfg.model_cfg.max_num_groups = self.model_cfg.max_num_groups
        self.model_cfg.model_cfg.max_seq_len = self.model_cfg.max_seq_len
        if self.model_type==3:
            from deepsvg.model.model import SVGTransformer
            self.model = SVGTransformer(self.model_cfg.model_cfg)
        elif self.model_type==35:
            from deepsvg.model.modified_model_between import SVGTransformer
            self.model = SVGTransformer(self.model_cfg.model_cfg)
        elif self.model_type==345:
            from deepsvg.model.modified_model_residualy import SVGTransformer
            self.model = SVGTransformer(self.model_cfg.model_cfg)
        elif self.model_type==85:
            from deepsvg.model.modified_model_shared_mlp_between import SVGTransformer
            self.model = SVGTransformer(self.model_cfg.model_cfg,mlp)
        elif self.model_type==39:
            from deepsvg.model.modified_model_between_added_agents import SVGTransformer
            self.model = SVGTransformer(self.model_cfg.model_cfg)
        elif self.model_type==1011:
            from deepsvg.model.modified_model_conv_added_agent import SVGTransformer
            self.model = SVGTransformer(self.model_cfg.model_cfg)
        else:
            from deepsvg.model.model import SVGTransformer
            self.model = SVGTransformer(self.model_cfg.model_cfg)
    #         print(self.model.encoder)

    def forward(self,x):
        if self.model_type==3:
            commands_enc,args_enc, commands_dec, args_dec,params, encode_mode = x
            return self.model(commands_enc, args_enc, commands_dec, args_dec,params=params,
                              encode_mode=encode_mode).squeeze(0).squeeze(0)
        elif self.model_type==35 or self.model_type==345 or self.model_type==85:
            commands_enc,args_enc, commands_dec, args_dec,history,params, encode_mode = x
            return self.model(commands_enc, args_enc, commands_dec, args_dec,history=history, params=params,
                              encode_mode=encode_mode)
        elif self.model_type==39 or self.model_type==1011:
            commands_enc,args_enc, commands_dec, args_dec,history,agents,agents_validity,params, encode_mode = x
            return self.model(commands_enc, args_enc, commands_dec, args_dec,history=history,
                              agents= agents,agents_validity=agents_validity, params=params,encode_mode=encode_mode)
        else:
            commands_enc,args_enc, commands_dec, args_dec,params, encode_mode = x
            return self.model(commands_enc, args_enc, commands_dec, args_dec,params=params,
                              encode_mode=encode_mode).squeeze(0).squeeze(0)