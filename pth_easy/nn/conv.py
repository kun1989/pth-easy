from torch import nn
import torch.nn.functional as F
from .norm import get_norm
from .act import get_act

class Conv2d(nn.Module):
    def __init__(self, in_channels,
                       out_channels,
                       kernel_size=1,
                       stride=1,
                       padding=0,
                       groups=1,
                       norm="",
                       act="",):

        super(Conv2d, self).__init__()
        use_bias = norm == ""
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=use_bias)
        self.norm = get_norm(norm, out_channels)
        self.activation = get_act(act)

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

class Conv2d_1x1(Conv2d):
    def __init__(self, in_channels, out_channels, norm="", act="", stride=1):
        super(Conv2d_1x1, self).__init__(in_channels,
                                         out_channels,
                                         kernel_size=1,
                                         stride=stride,
                                         padding=0,
                                         groups=1,
                                         norm=norm,
                                         act=act)

class DWConv2d(Conv2d):
    def __init__(self, channels, kernel_size, stride, padding, norm="", act=""):
        super(DWConv2d, self).__init__(channels,
                                         channels,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         groups=channels,
                                         norm=norm,
                                         act=act)


