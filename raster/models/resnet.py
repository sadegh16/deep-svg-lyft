from abc import ABC
import torch
import torchvision as tv
import typing as th
from raster.models.template import RasterModel


class Resnet(RasterModel):
    def __init__(self, config: dict, modes=1, model_type: str = 'resnet50', pretrained=True,
                 add_fc: th.Optional[th.Union[int, float, list]] = None,
                 add_fc_act: th.Optional[str] = None,
                 add_fc_dropout: float = 0.,
                 add_fc_act_args: th.Optional[dict] = None
                 ):
        super().__init__(config=config, modes=modes)
        self.model = getattr(tv.models, model_type)(pretrained=pretrained)
        self.add_fc, self.add_fc_act, self.add_fc_dropout = add_fc, add_fc_act, add_fc_dropout
        self.add_fc_act_args = add_fc_act_args
        if isinstance(add_fc_act_args, dict):
            act_args_list = []
            act_args_dict = add_fc_act_args
        else:
            act_args_list = add_fc_act_args if isinstance(add_fc_act_args, list) else [add_fc_act_args]
            act_args_dict = dict()
        if self.model.conv1.in_channels != self.in_channels:
            old_weight = self.model.conv1.weight
            old_bias = self.model.conv1.bias
            self.model.conv1 = torch.nn.Conv2d(
                in_channels=self.in_channels, out_channels=self.model.conv1.out_channels,
                stride=self.model.conv1.stride,
                bias=self.model.conv1.bias is not None, kernel_size=self.model.conv1.kernel_size,
                padding=self.model.conv1.padding,
                padding_mode=self.model.conv1.padding_mode, dilation=self.model.conv1.dilation)
            self.model.conv1.weight[:, :min(self.in_channels, old_weight.shape[1]), :, :].data = \
                old_weight[:, :min(self.in_channels, old_weight.shape[1]), :, :].data
            if old_bias is not None:
                self.model.conv1.bias.data = old_bias.data
        if self.model.fc.out_features != self.out_dim:
            if self.add_fc is None:
                fc = torch.nn.Linear(self.model.fc.in_features, self.out_dim, bias=self.model.fc.bias is not None)
            else:
                sizes, last_size, layers = self.add_fc if isinstance(self.add_fc, list) else [
                    self.add_fc], self.model.fc.in_features, []
                act = getattr(torch.nn, self.add_fc_act)(
                    *act_args_list, **act_args_dict) if self.add_fc_act else None
                dropout = torch.nn.Dropout(self.add_fc_dropout, inplace=True) if self.add_fc_dropout else None
                for s in sizes:
                    s = int(s * last_size) if isinstance(s, float) else s
                    if dropout is not None: layers.append(dropout)
                    layers.append(torch.nn.Linear(last_size, s, bias=self.model.fc.bias is not None))
                    if act is not None: layers.append(act)
                    last_size = s
                layers.append(torch.nn.Linear(last_size, self.out_dim, bias=self.model.fc.bias is not None))
                fc = torch.nn.Sequential(*layers)
            self.model.fc = fc
