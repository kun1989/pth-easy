import torch
from torch import nn
from ..nn import Conv2d_1x1, DWConv2d, Conv2d, xavier_init, msra_init
import numpy as np

__all__ = ['mobilenet_v2_relu']

def make_divisible(x, divisible_by=8):
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)

class SE(nn.Module):
    def __init__(self, num_in, ratio=4):
        super(SE, self).__init__()
        num_mid = make_divisible(num_in // ratio)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.conv1 = Conv2d_1x1(num_in, num_mid, norm='', act="relu")
        self.conv2 = Conv2d_1x1(num_mid, num_in, norm='', act="hard_sigmoid")

    def forward(self, x):
        w = self.pool(x)
        w = self.conv1(w)
        w = self.conv2(w)
        x = x * w
        return x

class LinearBottleneck(nn.Module):
    def __init__(self, kernel_size, in_channels, channels, t, stride, use_se, act):
        super(LinearBottleneck, self).__init__()
        self.use_se = use_se
        self.use_shortcut = stride == 1 and in_channels == channels
        padding = (kernel_size - 1) // 2
        hidden_dim = in_channels * t

        self.first_conv = (in_channels != hidden_dim)
        if self.first_conv:
            self.expand = Conv2d_1x1(in_channels, hidden_dim, norm='bn', act=act)

        self.conv1 = DWConv2d(hidden_dim, kernel_size, stride, padding, norm='bn', act=act)

        if use_se:
            self.se = SE(hidden_dim)

        self.conv2 = Conv2d_1x1(hidden_dim, channels, norm='bn', act='')

    def forward(self, x):
        out = self.expand(x) if self.first_conv else x
        out = self.conv1(out)
        if self.use_se:
            out = self.se(out)
        out = self.conv2(out)
        if self.use_shortcut:
            out = out + x
        return out

class MobileNetV2(nn.Module):
    def __init__(self, cfg_stages, first_channels, last_channels, classes=1000):
        super(MobileNetV2, self).__init__()

        self.conv1 = Conv2d(3, first_channels, 3, 2, 1, norm = "bn", act = "relu")

        self.stages = []
        in_channels = 0
        for name, params in cfg_stages.items():
            module = []
            for k, ic, oc, s, t, se, act in params:
                module.append(LinearBottleneck(k, ic, oc, t, s, se, act))
                in_channels = oc
            self.add_module(name, nn.Sequential(*module))
            self.stages.append(name)

        self.last_conv = Conv2d_1x1(in_channels, last_channels, norm = "bn", act = "relu")

        self.pool = nn.AvgPool2d(7)

        self.output = nn.Linear(last_channels, classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                msra_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                xavier_init(m)

        # Zero-initialize the last BN in each residual branch,
        for m in self.modules():
            if isinstance(m, LinearBottleneck):
                if m.use_shortcut:
                    nn.init.constant_(m.conv2.norm.weight, 0)

    def forward(self, x):
        x = self.conv1(x)
        for stage_name in self.stages:
            x = getattr(self, stage_name)(x)
        x = self.last_conv(x)
        x = self.pool(x)
        # x = x.flatten(start_dim=1)
        x = x.view(x.size(0), -1)
        x = self.output(x)
        return x

def list2dict(cfg_list, multiplier):
    cfg_stages = {}
    stage_num = 0
    cfg_stages['layer' + str(stage_num)] = []
    for k, ic, oc, s, t, se, act in cfg_list:
        if s == 2:
            stage_num += 1
            cfg_stages['layer' + str(stage_num)] = []

        cfg_stages['layer' + str(stage_num)].append([k, int(ic*multiplier), int(oc*multiplier), s, t, se, act])
    return cfg_stages

def get_mobilenetv2_relu(multiplier=1.0):
    #            k   ic   oc  s  t   se        act
    cfg_list = [[3,  32,  16, 1, 1, False, 'relu'],
                [3,  16,  24, 2, 6, False, 'relu'],
                [3,  24,  24, 1, 6, False, 'relu'],
                [3,  24,  32, 2, 6, False, 'relu'],
                [3,  32,  32, 1, 6, False, 'relu'],
                [3,  32,  32, 1, 6, False, 'relu'],
                [3,  32,  64, 2, 6, False, 'relu'],
                [3,  64,  64, 1, 6, False, 'relu'],
                [3,  64,  64, 1, 6, False, 'relu'],
                [3,  64,  64, 1, 6, False, 'relu'],
                [3,  64,  96, 1, 6, False, 'relu'],
                [3,  96,  96, 1, 6, False, 'relu'],
                [3,  96,  96, 1, 6, False, 'relu'],
                [3,  96, 160, 2, 6, False, 'relu'],
                [3, 160, 160, 1, 6, False, 'relu'],
                [3, 160, 160, 1, 6, False, 'relu'],
                [3, 160, 320, 1, 6, False, 'relu'],
                ]
    last_channels = int(1280 * multiplier) if multiplier > 1.0 else 1280
    cfg_stages = list2dict(cfg_list, multiplier)
    return MobileNetV2(cfg_stages, int(32 * multiplier), last_channels)

def mobilenet_v2_relu():
    return get_mobilenetv2_relu(1.0)

