import torch
from deepsvg.difflib.tensor import SVGTensor
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
import numpy as np


def _get_key_padding_mask(commands, seq_dim=0):
    """
    Args:
        commands: Shape [S, ...]
    """
    with torch.no_grad():
        key_padding_mask = (commands == SVGTensor.COMMANDS_SIMPLIFIED.index("EOS")).cumsum(dim=seq_dim) > 0
        # print("key_padding_mask",key_padding_mask.shape)
        if seq_dim == 0:
            return key_padding_mask.transpose(0, 1)
        return key_padding_mask


def _get_padding_mask(commands, seq_dim=0, extended=False):
    with torch.no_grad():
        padding_mask = (commands == SVGTensor.COMMANDS_SIMPLIFIED.index("EOS")).cumsum(dim=seq_dim) == 0
        padding_mask = padding_mask.float()
        # print("padding_mask",padding_mask.shape)
        if extended:
            # padding_mask doesn't include the final EOS, extend by 1 position to include it in the loss
            S = commands.size(seq_dim)
            torch.narrow(padding_mask, seq_dim, 3, S-3).add_(torch.narrow(padding_mask, seq_dim, 0, S-3)).clamp_(max=1)

        if seq_dim == 0:
            return padding_mask.unsqueeze(-1)
        return padding_mask


def _get_group_mask(commands, seq_dim=0):
    """
    Args:
        commands: Shape [S, ...]
    """
    with torch.no_grad():
        group_mask = (commands == SVGTensor.COMMANDS_SIMPLIFIED.index("m")).cumsum(dim=seq_dim)
        return group_mask


def _get_visibility_mask(commands, seq_dim=0,modified=False):
    """
    Args:
        commands: Shape [S, ...]
    """
    S = commands.size(seq_dim)
    # print(S)
    # print(commands.permute(2,1,0)[0][0],len(commands[:][0][0]))
    with torch.no_grad():
        # print( SVGTensor.COMMANDS_SIMPLIFIED.index("EOS"),commands == SVGTensor.COMMANDS_SIMPLIFIED.index("EOS"),
        #        (commands == SVGTensor.COMMANDS_SIMPLIFIED.index("EOS")).sum(dim=seq_dim))
        visibility_mask = (commands == SVGTensor.COMMANDS_SIMPLIFIED.index("EOS")).sum(dim=seq_dim) < S - 1
        # print("visibility_mask",visibility_mask.shape)
        if modified == True:
            #             print(visibility_mask.shape[1])
            l =torch.ones((1,visibility_mask.shape[1]),dtype=bool)
            # print("my tensor",l.unsqueeze(0))
            # print(l.shape)
            #             print("visibility_mask",visibility_mask.shape,l.unsqueeze(0).shape,commands.shape)
            visibility_mask = torch.cat((visibility_mask,l.to(device='cuda')))
        # print("visibility_mask",visibility_mask.shape)

        if seq_dim == 0:
            return visibility_mask.unsqueeze(-1)
        return visibility_mask


def _get_key_visibility_mask(commands, seq_dim=0,modified=False):
    S = commands.size(seq_dim)
    with torch.no_grad():
        key_visibility_mask = (commands == SVGTensor.COMMANDS_SIMPLIFIED.index("EOS")).sum(dim=seq_dim) >= S - 1
        # print("key_visibility_mask",key_visibility_mask.shape)
        if modified == True:
            l =torch.zeros((1,key_visibility_mask.shape[1]),dtype=bool)
            #             print(l)
            #             print("my second",l.unsqueeze(0))
            #             print("visibility_mask",key_visibility_mask.shape,l.unsqueeze(0).shape,commands.shape)
            key_visibility_mask = torch.cat((key_visibility_mask,l.to(device='cuda')))
        if seq_dim == 0:
            return key_visibility_mask.transpose(0, 1)
        return key_visibility_mask


def _generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def _sample_categorical(temperature=0.0001, *args_logits):
    if len(args_logits) == 1:
        arg_logits, = args_logits
        return Categorical(logits=arg_logits / temperature).sample()
    return (*(Categorical(logits=arg_logits / temperature).sample() for arg_logits in args_logits),)


def _threshold_sample(arg_logits, threshold=0.5, temperature=1.0):
    scores = F.softmax(arg_logits / temperature, dim=-1)[..., 1]
    return scores > threshold

