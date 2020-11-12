import torch


class RasterModel(torch.nn.Module):
    def __init__(self, config: dict, modes: int = 1):
        super().__init__()
        self.modes = modes
        self.future_len = 50
        self.num_preds = self.modes * 2 * self.future_len
        self.out_dim = self.num_preds + (self.modes if self.modes != 1 else 0)

    def _forward(self, x):
        return self.model(x)

    def forward(self, x):
        res = self._forward(x)
        print(res)
        if type(x) is list:
            bs = x[0].shape[0]
        else:
            bs = x.shape[0]
            
        if self.modes != 1:
            pred, conf = torch.split(res, self.num_preds, dim=1)
            pred = pred.view(bs, self.modes, self.future_len, 2)
            conf = torch.softmax(conf, dim=1)
            return pred, conf
        return res.view(bs, 1, self.future_len, 2), res.new_ones((bs, 1))