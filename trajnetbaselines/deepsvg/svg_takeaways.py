from deepsvg.svglib.svg import SVG
from deepsvg.svglib.geom import Point, Angle
from deepsvg.difflib.tensor import SVGTensor
import torch
import pdb



MAX_NUM_GROUPS = 8 #8
PAD_VAL = -1
MAX_SEQ_LEN = 30 
MAX_TOTAL_LEN = MAX_NUM_GROUPS * MAX_SEQ_LEN

def get_data(t_sep, fillings):
    res = {}
    model_args = ['commands', 'args', 'commands', 'args'] #,'tensor_grouped']
    
    t_sep = t_sep[0]
    fillings = fillings[0]
    
    # to remove all paths with a '-1' 
    '''
    t_sep_copy = t_sep.copy()
    fillings_copy = fillings.copy()
    t_sep = []
    fillings = []
    for tt, ff in zip(t_sep_copy, fillings_copy):
        if ((tt<0)&(tt!=-1)).any():
            pass
        else:
            t_sep.append(tt)
            fillings.append(ff)
    if(len(fillings)==0):
        return None'''
        
    pad_len = max(MAX_NUM_GROUPS - len(t_sep), 0)

    t_sep.extend([torch.empty(0, 14)] * pad_len)
    fillings.extend([0] * pad_len)
    

    t_grouped = [SVGTensor.from_data(torch.cat(t_sep, dim=0), PAD_VAL=PAD_VAL).add_eos().add_sos().pad(
        seq_len=MAX_TOTAL_LEN + 2)]
        
    t_sep = [SVGTensor.from_data(t, PAD_VAL=PAD_VAL, filling=f).add_eos().add_sos().pad(seq_len=MAX_SEQ_LEN + 2) for
             t, f in zip(t_sep, fillings)]
    for arg in set(model_args):
        if "_grouped" in arg:
            arg_ = arg.split("_grouped")[0] 
            t_list = t_grouped 
        else:
            arg_ = arg
            t_list = t_sep

        if arg_ == "tensor":
            res[arg] = t_list

        if arg_ == "commands":
            res[arg] = torch.stack([t.cmds() for t in t_list])

        if arg_ == "args_rel":
            res[arg] = torch.stack([t.get_relative_args() for t in t_list])
        if arg_ == "args":
            res[arg] = torch.stack([t.args() for t in t_list])

    if "filling" in model_args:
        res["filling"] = torch.stack([torch.tensor(t.filling) for t in t_sep]).unsqueeze(-1)

    if "label" in model_args:
        res["label"] = label

    return res
    
'''
# the default code with min changes
MAX_NUM_GROUPS = 8 #8
PAD_VAL = -1
MAX_SEQ_LEN = 30 
MAX_TOTAL_LEN = MAX_NUM_GROUPS * MAX_SEQ_LEN

def get_data(t_sep, fillings):
    res = {}
    model_args = ['commands', 'args', 'commands', 'args'] #,'tensor_grouped']
    
    t_sep = t_sep[0]
    fillings = fillings[0]
    
    pad_len = max(MAX_NUM_GROUPS - len(t_sep), 0)

    t_sep.extend([torch.empty(0, 14)] * pad_len)
    fillings.extend([0] * pad_len)


    t_grouped = [SVGTensor.from_data(torch.cat(t_sep, dim=0), PAD_VAL=PAD_VAL).add_eos().add_sos().pad(
        seq_len=MAX_TOTAL_LEN + 2)]
        
        
        
    t_sep = [SVGTensor.from_data(t, PAD_VAL=PAD_VAL, filling=f).add_eos().add_sos().pad(seq_len=MAX_SEQ_LEN + 2) for
             t, f in zip(t_sep, fillings)]
    for arg in set(model_args):
        if "_grouped" in arg:
            arg_ = arg.split("_grouped")[0] 
            t_list = t_grouped 
        else:
            arg_ = arg
            t_list = t_sep

        if arg_ == "tensor":
            res[arg] = t_list

        if arg_ == "commands":
            res[arg] = torch.stack([t.cmds() for t in t_list])

        if arg_ == "args_rel":
            res[arg] = torch.stack([t.get_relative_args() for t in t_list])
        if arg_ == "args":
            res[arg] = torch.stack([t.args() for t in t_list])

    if "filling" in model_args:
        res["filling"] = torch.stack([torch.tensor(t.filling) for t in t_sep]).unsqueeze(-1)

    if "label" in model_args:
        res["label"] = label

    return res'''