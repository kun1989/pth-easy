from torch import nn
import torch.nn.functional as F

class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.

class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.

def get_act(act, inplace = True):
    if isinstance(act, str):
        if len(act) == 0:
            return None
        act = {
            "relu": nn.ReLU,
            "relu6": nn.ReLU6,
            "hard_swish": Hswish,
            "hard_sigmoid": Hsigmoid,
        }[act]
    return act(inplace=inplace)
