import torch
from abc import ABC


class MaskedLinear(torch.nn.Linear, ABC):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, mask_weights=True, mask_bias=True):
        super().__init__(in_features, out_features, bias)
        self.mask_weights = mask_weights
        self.mask_bias = mask_bias

    @staticmethod
    def generate_mask(mask_lens, length):
        idx = torch.arange(length).to(mask_lens.device)
        return (idx.reshape(-1, length) < mask_lens.reshape(-1, 1)).float().to(mask_lens.device)

    @staticmethod
    def generate_bias_mask(mask_lens):
        return (mask_lens != 0).view(-1, 1)

    def forward(self, inputs, mask_lens=None, mask=None, mask_bias=None):
        # linear
        if self.mask_weights:
            mask = mask if mask is not None else self.generate_mask(mask_lens, self.in_features)
            linear = (inputs.view(-1, 1, inputs.shape[1]) @ (self.weight * mask.view(inputs.shape[0], 1, -1)).permute(
                0, 2, 1)).squeeze(1)
        else:
            linear = inputs @ self.weight.permute(1, 0)
        # bias
        if self.mask_bias:
            if mask_bias is None:
                mask_bias = self.generate_bias_mask(mask_lens)
            bias = 0 if self.bias is None else self.bias.view(1, -1).repeat(inputs.shape[0], 1) * mask_bias
        else:
            bias = 0 if self.bias is None else self.bias
        return linear + bias


class PolyLineMLP(torch.nn.Module, ABC):
    def __init__(self, in_features, ff_dim, out_features, num_layers=2, masked=False, bias=True):
        super().__init__()
        self.masked = masked
        self.num_layers = num_layers
        self.act = torch.nn.ReLU(inplace=True)
        self.in_features = in_features
        for i in range(num_layers):
            in_dim = ff_dim if i else in_features
            out_dim = out_features if i == num_layers - 1 else ff_dim
            mask_weights = (not i) and masked
            mask_bias = masked
            layer = MaskedLinear(in_dim, out_dim, bias=bias, mask_weights=mask_weights, mask_bias=mask_bias)
            torch.nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity='relu')
            setattr(self, f'layer{i}', layer)

    def forward(self, x, mask_lens=None, mask=None, mask_bias=None):
        if self.masked:
            mask = mask if mask is not None else MaskedLinear.generate_mask(mask_lens, self.in_features)
            mask_bias = mask_bias if mask_bias is not None else MaskedLinear.generate_bias_mask(mask_lens)

        for i in range(self.num_layers):
            x = getattr(self, f'layer{i}')(x, mask_lens, mask, mask_bias)
            if i != self.num_layers - 1:
                x = self.act(x)
        return x


class Albert(torch.nn.Module, ABC):
    def __init__(self, layer, num_layers, layer_norm=None):
        super().__init__()
        self.num_layers = num_layers
        self.layer = layer
        self.layer_norm = layer_norm

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        for i in range(self.num_layers):
            src = self.layer(src, src_mask, src_key_padding_mask)
        if self.layer_norm is not None:
            src = self.layer_norm(src)
        return src


class MLPResnet(torch.nn.Module, ABC):
    def __init__(self, num_layers, num_features, bias=True, drop_out=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.drop_out = torch.nn.Dropout(drop_out)
        self.act = torch.nn.ReLU()
        for i in range(num_layers):
            layer = torch.nn.Linear(num_features, num_features, bias=bias)
            torch.nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity='relu')
            setattr(self, f'layer{i}', layer)

    def forward(self, inputs):
        for i in range(self.num_layers):
            inputs = self.drop_out(self.act(getattr(self, f'layer{i}')(inputs))) + inputs
        return inputs


class MLP(torch.nn.Module, ABC):
    def __init__(self, num_layers, in_features, num_features, out_features, bias=True):
        super().__init__()
        self.num_layers = num_layers
        self.act = torch.nn.ReLU()
        for i in range(num_layers):
            f_in = in_features if not i else num_features
            f_out = out_features if i == num_layers - 1 else num_features
            layer = torch.nn.Linear(f_in, f_out, bias=bias)
            torch.nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity='relu')
            setattr(self, f'layer{i}', layer)

    def forward(self, inputs):
        for i in range(self.num_layers):
            inputs = self.act(getattr(self, f'layer{i}')(inputs))
        return inputs


class Cheese(torch.nn.Module, ABC):
    def __init__(self, history_num=20, scene_num=20, agents_num=170, modes=1, future_len=30,
                 num_layers=50, d_model=1024, nhead=16, dim_feedforward=4096, dropout=0.1, activation='relu',
                 layer_norm=False, albert=True,
                 traj_ff_dim=768, traj_num_layers=3, scene_ff_dim=512, scene_num_layers=3,
                 mask_hist=False, mask_agents=True, mask_scene=True,
                 cat_transforms=True, res_dim=1024, final_dim=512, final_num_layers=3,
                 resnet_num_layers=5, hist_residual_final=True, hist_residual_transform=True,
                 ):
        super().__init__()
        assert (not hist_residual_final) or d_model == res_dim
        self.history, self.scene, self.agents = history_num is not None, scene_num is not None, agents_num is not None
        self.hist_residual_final = hist_residual_final and self.history
        self.hist_residual_transform = hist_residual_transform and self.history
        self.cat_transforms = cat_transforms

        self.modes = modes
        self.future_len = future_len
        self.num_preds = self.modes * 2 * self.future_len
        self.out_dim = self.num_preds + (self.modes if self.modes != 1 else 0)

        # embedding
        if self.history:
            self.embed_hist = PolyLineMLP(history_num * 2, traj_ff_dim, d_model, traj_num_layers, masked=mask_hist)
        if self.agents:
            self.embed_agents = PolyLineMLP(history_num * 2, traj_ff_dim, d_model, traj_num_layers, masked=mask_agents)
        if self.scene:
            self.embed_scene = PolyLineMLP(scene_num * 2, scene_ff_dim, d_model, scene_num_layers,
                                           masked=mask_scene)
        self.label = torch.nn.Embedding(self.history + self.scene + self.agents, d_model)
        # transformer
        layer_norm = torch.nn.LayerNorm(d_model) if layer_norm else None
        transform_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model, dim_feedforward=dim_feedforward, dropout=dropout,
            nhead=nhead, activation=activation
        )

        self.transform = torch.nn.TransformerEncoder(
            transform_layer, num_layers, layer_norm) if not albert else Albert(transform_layer, num_layers, layer_norm)

        layer = torch.nn.Linear(((self.history + self.scene + self.agents) * d_model if cat_transforms else d_model),
                                res_dim, bias=True)
        torch.nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity='relu')
        self.mix_results = torch.nn.Sequential(layer, torch.nn.ReLU(inplace=True), torch.nn.Dropout(dropout))
        self.resnet = MLPResnet(resnet_num_layers, res_dim, drop_out=dropout)
        self.final = MLP(final_num_layers, res_dim, final_dim, self.out_dim, bias=True)

    def _forward(self, history_positions=None, scene=None, scene_lens=None, agents=None, agents_lens=None):
        # embedding
        viz_mask, poly_lines = [], []
        item_id, idx = 0, 0
        if self.history:
            hist_idx = idx
            idx += 1
            hist = history_positions  # batch['history_positions'] N, HIST_LEN, 2
            # print('hist', hist.shape)
            label = self.label(torch.LongTensor([item_id]).to(hist.device))
            # print(label.shape, 'label')
            # print('out embed hist', self.embed_hist(hist.reshape(-1, hist.shape[-1])).shape)
            hist_embed = self.embed_hist(hist.reshape(-1, hist.shape[-1])).reshape(hist.shape[0], 1, -1).permute(
                1, 0, 2) + label
            # print('hist_embed', hist_embed.shape)
            poly_lines.append(hist_embed)
            viz_mask.append(torch.zeros(hist.shape[0], 1).to(hist.device))
            item_id += 1
            # print('item, idx', item_id, idx)

        if self.scene:
            scene = scene  # batch['padded_cntr_lines']
            # print('scene', scene.shape)
            scene_lens = scene_lens  # batch['available_cntr_size'].reshape(-1)
            # print('scene_lens', scene_lens.shape)
            scene_idx = slice(idx, idx + scene.shape[1])
            idx += scene.shape[1]
            scene_bias_mask = MaskedLinear.generate_bias_mask(scene_lens)
            # print('scene_bias_mask', scene_bias_mask.shape)
            label = self.label(torch.LongTensor([item_id]).to(scene.device))
            scene_embed = self.embed_scene(
                scene.reshape(-1, scene.shape[-1]), scene_lens, mask_bias=scene_bias_mask
            ).reshape(scene.shape[0], scene.shape[1], -1).permute(1, 0, 2) + label
            # print('scene_embed', scene_embed.shape)
            poly_lines.append(scene_embed)
            viz_mask.append((1 - scene_bias_mask.float().reshape(-1, scene.shape[1])).to(scene.device))
            # print('item, idx', item_id, idx)

        if self.agents:
            agents = agents  # batch['padded_cntr_lines']
            # print('agents', agents.shape)
            agents_lens = agents_lens  # batch['available_cntr_size'].reshape(-1)
            # print('agents_lens', agents_lens.shape)
            agents_idx = slice(idx, idx + agents.shape[1])
            idx += agents.shape[1]
            agents_bias_mask = MaskedLinear.generate_bias_mask(agents_lens)
            # print('agents_bias_mask', agents_bias_mask.shape)
            label = self.label(torch.LongTensor([item_id]).to(agents.device))
            agents_embed = self.embed_agents(
                agents.reshape(-1, agents.shape[-1]), agents_lens, mask_bias=agents_bias_mask
            ).reshape(agents.shape[0], agents.shape[1], -1).permute(1, 0, 2) + label
            # print('agents_embed', agents_embed.shape)
            poly_lines.append(agents_embed)
            viz_mask.append((1 - agents_bias_mask.float().reshape(-1, agents.shape[1])).to(agents.device))
            # print('item, idx', item_id, idx)

        viz_mask = torch.cat(viz_mask, dim=1)
        # print('viz_mask', viz_mask.shape)
        poly_lines = torch.cat(poly_lines)
        # print('poly_lines', poly_lines.shape)
        transformed = self.transform(poly_lines, src_key_padding_mask=viz_mask.type(torch.bool)) * (
                1 - viz_mask).permute(1, 0).unsqueeze(-1)
        # print('transformed', transformed.shape)

        results = []
        if self.history:
            transformed_hist = (transformed[hist_idx, ...] + (hist_embed[0] if self.hist_residual_transform else 0)
                                ).reshape(hist.shape[0], -1) / (1 + self.hist_residual_transform)
            results.append(transformed_hist)
            # print('transformed_hist', transformed_hist.shape)
        if self.scene:
            transformed_scene = transformed[scene_idx, ...].sum(0) / (1 - viz_mask.permute(1, 0)[scene_idx, ...]).sum(
                0).reshape(-1, 1)
            results.append(transformed_scene)
            # print('transformed_scene', transformed_scene.shape)
        if self.agents:
            transformed_agents = transformed[agents_idx, ...].sum(0) / (
                    1 - viz_mask.permute(1, 0)[agents_idx, ...]).sum(0).reshape(-1, 1)
            results.append(transformed_agents)
            # print('transformed_agents', transformed_agents.shape)

        results = torch.cat(results, dim=1) if self.cat_transforms else sum(results) / (
                self.history + self.scene + self.agents)
        # print('results averaged/cat', results.shape)
        results = self.resnet(self.mix_results(results)) + (hist_embed[0] if self.hist_residual_final else 0)
        # print('results mixed', results.shape)
        return self.final(results)

    def forward(self, x):
        history_positions, scene, scene_lens, agents, agents_lens = x
        res = self._forward(history_positions, scene, scene_lens, agents, agents_lens)
        bs = history_positions.shape[0]
        if self.modes != 1:
            pred, conf = torch.split(res, self.num_preds, dim=1)
            pred = pred.reshape(bs, self.modes, self.future_len, 2)
            conf = torch.softmax(conf, dim=1)
            return pred, conf
        return res.reshape(bs, 1, self.future_len, 2), res.new_ones((bs, 1))
