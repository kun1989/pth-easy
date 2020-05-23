import numpy as np
from torch import nn
import torch.nn.functional as F
from ..nn import Conv2d_1x1, DWConv2d, Conv2d, get_act, xavier_init, msra_init

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

class ResBottleneck(nn.Module): #k, ic, mc, oc, se, act, s
    def __init__(self, kernel_size, num_in, num_mid, num_out, use_se, act, strides):
        super(ResBottleneck, self).__init__()
        self.use_se = use_se
        self.first_conv = (num_in != num_mid)
        self.use_short_cut_conv = True
        if self.first_conv:
            self.expand = Conv2d_1x1(num_in, num_mid, norm='bn', act=act)

        padding = (kernel_size - 1) // 2
        self.conv1 = DWConv2d(num_mid, kernel_size, strides, padding, norm='bn', act=act)

        if use_se:
            self.se = SE(num_mid)

        self.conv2 = Conv2d_1x1(num_mid, num_out, norm='bn', act='')

        if num_in != num_out or strides != 1:
            self.use_short_cut_conv = False

    def forward(self, x):
        out = self.expand(x) if self.first_conv else x
        out = self.conv1(out)
        if self.use_se:
            out = self.se(out)
        out = self.conv2(out)
        if self.use_short_cut_conv:
            return x + out
        else:
            return out

class MobileNetV3(nn.Module):
    def __init__(self, cfg_stages, first_channels, cls_ch_squeeze, cls_ch_expand, classes=1000):
        super(MobileNetV3, self).__init__()

        self.conv1 = Conv2d(3, first_channels, 3, 2, 1, norm="bn", act="hard_swish")

        self.stages = []
        in_channels = 0
        for name, params in cfg_stages.items():
            module = []
            for k, ic, mc, oc, se, act, s in params:
                module.append(ResBottleneck(k, ic, mc, oc, se, act, s))
                in_channels = oc
            self.add_module(name, nn.Sequential(*module))
            self.stages.append(name)

        self.last_conv1 = Conv2d_1x1(in_channels, cls_ch_squeeze, norm="bn", act="hard_swish")
        self.pool = nn.AvgPool2d(7)

        self.last_conv2 = Conv2d_1x1(cls_ch_squeeze, cls_ch_expand, norm="", act="hard_swish")

        self.output = nn.Linear(cls_ch_expand, classes)

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
            if isinstance(m, ResBottleneck):
                if m.use_short_cut_conv:
                    nn.init.constant_(m.conv2.norm.weight, 0)

    def forward(self, x):
        x = self.conv1(x)
        for stage_name in self.stages:
            x = getattr(self, stage_name)(x)
        x = self.last_conv1(x)
        x = self.pool(x)
        x = self.last_conv2(x)
        x = x.flatten(start_dim=1)
        x = self.output(x)
        return x

def list2dict(cfg_list, multiplier):
    cfg_stages = {}
    stage_num = 0
    cfg_stages['layer' + str(stage_num)] = []
    for k, ic, mc, oc, se, act, s in cfg_list:
        if s == 2:
            stage_num += 1
            cfg_stages['layer' + str(stage_num)] = []
        ic = make_divisible(multiplier * ic)
        mc = make_divisible(multiplier * mc)
        oc = make_divisible(multiplier * oc)
        cfg_stages['layer' + str(stage_num)].append([k, ic, mc, oc, se, act, s])
    return cfg_stages

def get_mobilenet_v3(model_name, multiplier=1.):
    if model_name == "large":
        cfg_list = [
            # k   ic   mc   oc   se        act       s
             [3,  16,  16,  16, False,       'relu', 1],
             [3,  16,  64,  24, False,       'relu', 2],
             [3,  24,  72,  24, False,       'relu', 1],
             [5,  24,  72,  40,  True,       'relu', 2],
             [5,  40, 120,  40,  True,       'relu', 1],
             [5,  40, 120,  40,  True,       'relu', 1],
             [3,  40, 240,  80, False, 'hard_swish', 2],
             [3,  80, 200,  80, False, 'hard_swish', 1],
             [3,  80, 184,  80, False, 'hard_swish', 1],
             [3,  80, 184,  80, False, 'hard_swish', 1],
             [3,  80, 480, 112,  True, 'hard_swish', 1],
             [3, 112, 672, 112,  True, 'hard_swish', 1],
             [5, 112, 672, 160,  True, 'hard_swish', 2],
             [5, 160, 960, 160,  True, 'hard_swish', 1],
             [5, 160, 960, 160,  True, 'hard_swish', 1],
            ]
        cls_ch_squeeze = 960
        cls_ch_expand = 1280
    elif model_name == "small":
        cfg_list = [
            # k  ic   mc  oc   se        act       s
             [3, 16,  16, 16,  True,       'relu', 2],
             [3, 16,  72, 24, False,       'relu', 2],
             [3, 24,  88, 24, False,       'relu', 1],
             [5, 24,  96, 40,  True, 'hard_swish', 2],
             [5, 40, 240, 40,  True, 'hard_swish', 1],
             [5, 40, 240, 40,  True, 'hard_swish', 1],
             [5, 40, 120, 48,  True, 'hard_swish', 1],
             [5, 48, 144, 48,  True, 'hard_swish', 1],
             [5, 48, 288, 96,  True, 'hard_swish', 2],
             [5, 96, 576, 96,  True, 'hard_swish', 1],
             [5, 96, 576, 96,  True, 'hard_swish', 1],
            ]
        cls_ch_squeeze = 576
        cls_ch_expand = 1280
    else:
        raise NotImplementedError

    cfg_stages = list2dict(cfg_list, multiplier)
    net = MobileNetV3(cfg_stages, make_divisible(16*multiplier), cls_ch_squeeze, cls_ch_expand)
    return net


def mobilenet_v3_large():
    return get_mobilenet_v3("large")

def mobilenet_v3_small():
    return get_mobilenet_v3("small")
