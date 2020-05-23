import torch
from torch import nn
from ..nn import Conv2d_1x1, DWConv2d, Conv2d, xavier_init, msra_init

__all__ = [
    'MobileNetV2',
    'mobilenet_v2_1_0',
    'mobilenet_v2_0_75',
    'mobilenet_v2_0_5',
    'mobilenet_v2_0_25'
]

class LinearBottleneck(nn.Module):
    def __init__(self, kernel_size, in_channels, channels, t, stride):
        super(LinearBottleneck, self).__init__()
        self.use_shortcut = stride == 1 and in_channels == channels
        padding = (kernel_size - 1) // 2
        hidden_dim = in_channels * t

        if t != 1:
            self.conv = nn.Sequential(
                Conv2d_1x1(in_channels, hidden_dim, norm='bn', act="relu6"),
                DWConv2d(hidden_dim, kernel_size, stride, padding, norm='bn', act="relu6"),
                Conv2d_1x1(hidden_dim, channels, norm='bn', act="")
            )
        else:
            self.conv = nn.Sequential(
                DWConv2d(hidden_dim, kernel_size, stride, padding, norm='bn', act="relu6"),
                Conv2d_1x1(hidden_dim, channels, norm='bn', act="")
            )

    def forward(self, x):
        out = self.conv(x)
        if self.use_shortcut:
            out = out + x
        return out

def list2dict(cfg_list, multiplier):
    cfg_stages = {}
    stage_num = 0
    cfg_stages['layer' + str(stage_num)] = []
    for k, ic, oc, s, t in cfg_list:
        if s == 2:
            stage_num += 1
            cfg_stages['layer' + str(stage_num)] = []

        cfg_stages['layer' + str(stage_num)].append([k, int(ic*multiplier), int(oc*multiplier), s, t])
    return cfg_stages

class MobileNetV2(nn.Module):
    def __init__(self, cfg_stages, first_channels, last_channels, classes=1000):
        super(MobileNetV2, self).__init__()

        self.conv1 = Conv2d(3, first_channels, 3, 2, 1, norm = "bn", act = "relu6")

        self.stages = []
        in_channels = 0
        for name, params in cfg_stages.items():
            module = []
            for k, ic, oc, s, t in params:
                module.append(LinearBottleneck(k, ic, oc, t, s))
                in_channels = oc
            self.add_module(name, nn.Sequential(*module))
            self.stages.append(name)

        self.last_conv = Conv2d_1x1(in_channels, last_channels, norm = "bn", act = "relu6")

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
                   nn.init.constant_(m.conv[2].norm.weight, 0)

    def forward(self, x):
        x = self.conv1(x)
        for stage_name in self.stages:
            x = getattr(self, stage_name)(x)
        x = self.last_conv(x)
        x = self.pool(x)
        x = x.flatten(start_dim=1)
        #x = x.view(x.size(0), -1)
        x = self.output(x)
        return x

def get_mobilenetv2(multiplier=1.0):
    #            k   ic   oc  s  t
    cfg_list = [[3,  32,  16, 1, 1],
                [3,  16,  24, 2, 6],
                [3,  24,  24, 1, 6],
                [3,  24,  32, 2, 6],
                [3,  32,  32, 1, 6],
                [3,  32,  32, 1, 6],
                [3,  32,  64, 2, 6],
                [3,  64,  64, 1, 6],
                [3,  64,  64, 1, 6],
                [3,  64,  64, 1, 6],
                [3,  64,  96, 1, 6],
                [3,  96,  96, 1, 6],
                [3,  96,  96, 1, 6],
                [3,  96, 160, 2, 6],
                [3, 160, 160, 1, 6],
                [3, 160, 160, 1, 6],
                [3, 160, 320, 1, 6],
                ]
    last_channels = int(1280 * multiplier) if multiplier > 1.0 else 1280
    cfg_stages = list2dict(cfg_list, multiplier)
    return MobileNetV2(cfg_stages, int(32 * multiplier), last_channels)

def mobilenet_v2_1_0():
    return get_mobilenetv2(1.0)

def mobilenet_v2_0_75():
    return get_mobilenetv2(0.75)

def mobilenet_v2_0_5():
    return get_mobilenetv2(0.5)

def mobilenet_v2_0_25():
    return get_mobilenetv2(0.25)
