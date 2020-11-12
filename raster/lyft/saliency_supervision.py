import torch


class SaliencySupervision(torch.nn.Module):
    def __init__(self, config, intrest='simple', **kwargs):
        super().__init__()
        self.intrest = intrest
        self.intrest_func = getattr(self, self.intrest)
        getattr(self, f'setup_{self.intrest}')(config, kwargs)

    def forward(self, grads, *args, **kwargs):
        intrest = self.intrest_func(grads)
        intrest_sum = intrest.abs().sum(dim=[1, 2, 3])
        total = grads.abs().sum(dim=[1, 2, 3])
        return torch.true_divide(intrest_sum, total)

    def simple(self, grads, *args, **kwargs):
        return grads[:, :, self.ys:self.ye, self.xs:self.xe]

    def setup_simple(self, config, kwargs):
        xs, xe = kwargs.get('xs', 0.15), kwargs.get('xe', 0.6)
        ys, ye = kwargs.get('ys', 0.35), kwargs.get('ye', 0.65)
        rs = config['raster_params']['raster_size']
        self.xs, self.xe = int(xs * rs[0]), int(xe * rs[1])
        self.ys, self.ye = int(ys * rs[0]), int(ye * rs[1])

    def boundary(self, grads, inputs, *args, **kwargs):
        pass

    def setup_boundary(self, config, kwargs):
        pass
