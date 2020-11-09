from deepsvg.difflib.tensor import SVGTensor
from deepsvg.utils.utils import _pack_group_batch, _unpack_group_batch, _make_seq_first, _make_batch_first

from .layers.transformer import *
from .layers.improved_transformer import *
from .layers.positional_encoding import *
from .basic_blocks import FCN, HierarchFCN, ResNet
from .config import _DefaultConfig
from .utils import (_get_padding_mask, _get_key_padding_mask, _get_group_mask, _get_visibility_mask,
                    _get_key_visibility_mask, _generate_square_subsequent_mask, _sample_categorical, _threshold_sample)

from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from scipy.optimize import linear_sum_assignment


class SVGEmbedding(nn.Module):
    def __init__(self, cfg: _DefaultConfig, seq_len, rel_args=False, use_group=True, group_len=None):
        super().__init__()

        self.cfg = cfg

        self.command_embed = nn.Embedding(cfg.n_commands, cfg.d_model)

        args_dim = 2 * cfg.args_dim if rel_args else cfg.args_dim + 1
        self.arg_embed = nn.Embedding(args_dim, 64)
        self.embed_fcn = nn.Linear(64 * cfg.n_args, cfg.d_model)

        self.use_group = use_group
        if use_group:
            if group_len is None:
                group_len = cfg.max_num_groups
            self.group_embed = nn.Embedding(group_len + 2, cfg.d_model)

        self.pos_encoding = PositionalEncodingLUT(cfg.d_model, max_len=seq_len + 2)

        self._init_embeddings()

    def _init_embeddings(self):
        nn.init.kaiming_normal_(self.command_embed.weight, mode="fan_in")
        nn.init.kaiming_normal_(self.arg_embed.weight, mode="fan_in")
        nn.init.kaiming_normal_(self.embed_fcn.weight, mode="fan_in")

        if self.use_group:
            nn.init.kaiming_normal_(self.group_embed.weight, mode="fan_in")

    def forward(self, commands, args, groups=None):
        S, GN = commands.shape

        src = self.command_embed(commands.long()) + \
              self.embed_fcn(self.arg_embed((args + 1).long()).view(S, GN, -1))  # shift due to -1 PAD_VAL

        if self.use_group:
            src = src + self.group_embed(groups.long())

        src = self.pos_encoding(src)

        return src


class TrajEmbedding(nn.Module):
    def __init__(self, cfg: _DefaultConfig, seq_len, rel_args=False, use_group=True, group_len=None):
        super().__init__()

        self.cfg = cfg

        self.command_embed = nn.Embedding(cfg.n_commands, cfg.d_model)

        args_dim = 2 * cfg.args_dim if rel_args else cfg.args_dim + 1
        self.arg_embed = nn.Embedding(args_dim, 64)
        self.embed_fcn = nn.Linear(64 * cfg.n_args, cfg.d_model)
        self.embed_tfn = nn.Linear(cfg.n_args, cfg.d_model)

        self.use_group = use_group
        if use_group:
            if group_len is None:
                group_len = cfg.max_num_groups
            self.group_embed = nn.Embedding(group_len + 2, cfg.d_model)

        self.pos_encoding = PositionalEncodingLUT(cfg.d_model, max_len=seq_len + 2)

        self._init_embeddings()

    def _init_embeddings(self):
        nn.init.kaiming_normal_(self.command_embed.weight, mode="fan_in")
        nn.init.kaiming_normal_(self.arg_embed.weight, mode="fan_in")
        nn.init.kaiming_normal_(self.embed_fcn.weight, mode="fan_in")

        if self.use_group:
            nn.init.kaiming_normal_(self.group_embed.weight, mode="fan_in")

    def forward(self, commands, args, groups=None):
        S, GN = commands.shape

        src = self.command_embed(commands.long()) + \
              self.embed_fcn(self.arg_embed((args + 1).long()).view(S, GN, -1))  # shift due to -1 PAD_VAL

        if self.use_group:
            src = src + self.group_embed(groups.long())

        src = self.pos_encoding(src)

        return src


class ConstEmbedding(nn.Module):
    def __init__(self, cfg: _DefaultConfig, seq_len):
        super().__init__()

        self.cfg = cfg

        self.seq_len = seq_len

        self.PE = PositionalEncodingLUT(cfg.d_model, max_len=seq_len)

    def forward(self, z):
        N = z.size(1)
        src = self.PE(z.new_zeros(self.seq_len, N, self.cfg.d_model))
        return src


class Encoder(nn.Module):
    def __init__(self, cfg: _DefaultConfig):
        super().__init__()

        self.cfg = cfg

        seq_len = cfg.max_seq_len if cfg.encode_stages == 2 else cfg.max_total_len
        self.use_group = cfg.encode_stages == 1
        self.embedding = SVGEmbedding(cfg, seq_len, use_group=self.use_group)
        self.traj_embedding = TrajEmbedding(cfg, 20, use_group=self.use_group)  # TODO fix history

        # scene encoder
        encoder_layer = TransformerEncoderLayerImproved(
            cfg.d_model, cfg.n_heads, cfg.dim_feedforward, cfg.dropout, d_global2=None)
        encoder_norm = LayerNorm(cfg.d_model)
        self.encoder = TransformerEncoder(encoder_layer, cfg.n_layers, encoder_norm)

        # traj
        encoder_layer = TransformerEncoderLayerImproved(
            cfg.d_model, cfg.n_heads, cfg.dim_feedforward, cfg.dropout, d_global2=None)
        encoder_norm = LayerNorm(cfg.d_model)
        self.traj_encoder = TransformerEncoder(encoder_layer, cfg.n_layers, encoder_norm)

        if not cfg.self_match:
            self.hierarchical_PE = PositionalEncodingLUT(
                cfg.d_model, max_len=cfg.max_num_groups + 1)  # todo change for agents

        hierarchical_encoder_layer = TransformerEncoderLayerImproved(
            cfg.d_model, cfg.n_heads, cfg.dim_feedforward, cfg.dropout, d_global2=None)
        hierarchical_encoder_norm = LayerNorm(cfg.d_model)
        self.hierarchical_encoder = TransformerEncoder(hierarchical_encoder_layer, cfg.n_layers,
                                                       hierarchical_encoder_norm)

    def forward(self, commands, args, commands_his, args_his, label=None):
        S, G, N = commands.shape
        l = None

        if self.cfg.encode_stages == 2:
            visibility_mask, key_visibility_mask = _get_visibility_mask(commands, seq_dim=0), _get_key_visibility_mask(
                commands, seq_dim=0)
            tvisibility_mask, tkey_visibility_mask = _get_visibility_mask(
                commands, seq_dim=0), _get_key_visibility_mask(commands, seq_dim=0)
            visibility_mask = torch.cat([visibility_mask, tvisibility_mask])
            key_visibility_mask = torch.cat([key_visibility_mask, tkey_visibility_mask], dim=1)

        tcmds, targs, l = _pack_group_batch(commands, args, l)
        commands, args, l = _pack_group_batch(commands, args, l)

        padding_mask, key_padding_mask = _get_padding_mask(commands, seq_dim=0), _get_key_padding_mask(
            commands, seq_dim=0)
        group_mask = _get_group_mask(commands, seq_dim=0) if self.use_group else None

        tpadding_mask, tkey_padding_mask = _get_padding_mask(tcmds, seq_dim=0), _get_key_padding_mask(
            tcmds, seq_dim=0)
        tgroup_mask = _get_group_mask(tcmds, seq_dim=0) if self.use_group else None

        src = self.embedding(commands, args, group_mask)
        tsrc = self.traj_embedding(tcmds, targs, tgroup_mask)

        # transformer
        memory = self.encoder(src, mask=None, src_key_padding_mask=key_padding_mask, memory2=None)
        z = (memory * padding_mask).sum(dim=0, keepdim=True) / padding_mask.sum(dim=0, keepdim=True)
        z = _unpack_group_batch(N, z)
        # traj transform
        memory = self.traj_encoder(tsrc, mask=None, src_key_padding_mask=tkey_padding_mask, memory2=None)
        tz = (memory * tpadding_mask).sum(dim=0, keepdim=True) / tpadding_mask.sum(dim=0, keepdim=True)
        tz = _unpack_group_batch(N, tz)

        z = torch.cat([z, tz], dim=1)

        # second stage
        src = z.transpose(0, 1)
        src = _pack_group_batch(src)
        if not self.cfg.self_match:
            src = self.hierarchical_PE(src)
        memory = self.hierarchical_encoder(src, mask=None, src_key_padding_mask=key_visibility_mask, memory2=None)
        z = (memory * visibility_mask).sum(dim=0, keepdim=True) / visibility_mask.sum(dim=0, keepdim=True)
        z = _unpack_group_batch(N, z)
        return z


class Bottleneck(nn.Module):
    def __init__(self, cfg: _DefaultConfig):
        super(Bottleneck, self).__init__()

        self.bottleneck = nn.Linear(cfg.d_model, cfg.dim_z)

    def forward(self, z):
        return self.bottleneck(z)


class SVGTransformer(nn.Module):
    def __init__(self, cfg: _DefaultConfig):
        super(SVGTransformer, self).__init__()

        self.cfg = cfg
        self.args_dim = 2 * cfg.args_dim if cfg.rel_targets else cfg.args_dim + 1

        if self.cfg.encode_stages > 0:

            self.encoder = Encoder(cfg)

            if cfg.use_resnet:
                self.resnet = ResNet(cfg.d_model)

            self.bottleneck = Bottleneck(cfg)

        self.register_buffer("cmd_args_mask", SVGTensor.CMD_ARGS_MASK)

    def forward(self, commands_enc, args_enc, commands_dec, args_dec, commands_his, args_his, label=None,
                z=None, hierarch_logits=None,
                return_tgt=True, params=None, encode_mode=False, return_hierarch=False):
        commands_enc, args_enc = _make_seq_first(commands_enc, args_enc)  # Possibly None, None
        commands_his, args_his = _make_seq_first(commands_his, args_his)

        if z is None:
            z = self.encoder(commands_enc, args_enc, commands_his, args_his, label)

            if self.cfg.use_resnet:
                z = self.resnet(z)

            z = self.bottleneck(z)
        else:
            z = _make_seq_first(z)

        return z
