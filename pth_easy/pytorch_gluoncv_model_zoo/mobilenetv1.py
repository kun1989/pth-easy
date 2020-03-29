from torch import nn
from .common import Conv2d_1x1, DWConv2d, Conv2d

__all__ = [
    'MobileNetV1',
    'mobilenet_v1_1_0',
    'mobilenet_v1_0_75',
    'mobilenet_v1_0_5',
    'mobilenet_v1_0_25'
]

def DepthwiseSeparableConv(kernel_size, in_channels, channels, stride):
    padding = (kernel_size - 1) // 2
    conv = nn.Sequential(
        DWConv2d(in_channels, kernel_size, stride, padding, norm='bn', act="relu"),
        Conv2d_1x1(in_channels, channels, norm='bn', act="relu")
    )
    return conv


class MobileNetV1(nn.Module):
    def __init__(self, cfg_stages, first_channels, classes=1000):
        super(MobileNetV1, self).__init__()
        self.conv1 = Conv2d(3, first_channels, 3, 2, 1, norm="bn", act="relu")

        self.stages = []
        in_channels = 0
        for name, params in cfg_stages.items():
            module = []
            for k, ic, oc, s in params:
                module.append(DepthwiseSeparableConv(k, ic, oc, s))
                in_channels = oc
            self.add_module(name, nn.Sequential(*module))
            self.stages.append(name)

        self.pool = nn.AvgPool2d(7)

        self.output = nn.Linear(in_channels, classes)

    def forward(self, x):
        x = self.conv1(x)
        for stage_name in self.stages:
            x = getattr(self, stage_name)(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.output(x)
        return x

def list2dict(cfg_list, multiplier):
    cfg_stages = {}
    stage_num = 0
    cfg_stages['layer' + str(stage_num)] = []
    for k, ic, oc, s in cfg_list:
        if s == 2:
            stage_num += 1
            cfg_stages['layer' + str(stage_num)] = []

        cfg_stages['layer' + str(stage_num)].append([k, int(ic*multiplier), int(oc*multiplier), s])
    return cfg_stages

def get_mobilenetv1(multiplier=1.0):
    #            k    ic    oc  s
    cfg_list = [[3,   32,   64, 1],
                [3,   64,  128, 2],
                [3,  128,  128, 1],
                [3,  128,  256, 2],
                [3,  256,  256, 1],
                [3,  256,  512, 2],
                [3,  512,  512, 1],
                [3,  512,  512, 1],
                [3,  512,  512, 1],
                [3,  512,  512, 1],
                [3,  512,  512, 1],
                [3,  512, 1024, 2],
                [3, 1024, 1024, 1],
                ]
    cfg_stages = list2dict(cfg_list, multiplier)
    return MobileNetV1(cfg_stages, int(32 * multiplier))

def mobilenet_v1_1_0():
    return get_mobilenetv1(1.0)

def mobilenet_v1_0_75():
    return get_mobilenetv1(0.75)

def mobilenet_v1_0_5():
    return get_mobilenetv1(0.5)

def mobilenet_v1_0_25():
    return get_mobilenetv1(0.25)

