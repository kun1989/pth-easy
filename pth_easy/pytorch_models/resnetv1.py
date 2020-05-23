from torch import nn
from ..nn import Conv2d_1x1, Conv2d, xavier_init, msra_init

__all__ = [
    'ResNetV1',
    'resnet50_v1',
]

class BottleneckV1(nn.Module):
    def __init__(self, ic, c, oc, s):
        super(BottleneckV1, self).__init__()
        self.conv1 = Conv2d(ic,  c, 1, s, 0, norm="bn", act="relu")
        self.conv2 = Conv2d( c,  c, 3, 1, 1, norm="bn", act="relu")
        self.conv3 = Conv2d( c, oc, 1, 1, 0, norm="bn", act="")

        if s != 1 or ic != oc:
            self.downsample = Conv2d(ic, oc, 1, s, 0, norm="bn", act="")
        else:
            self.downsample = None

        self.relu3 = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu3(out)

        return out

class BaseStem(nn.Module):
    def __init__(self):
        super(BaseStem, self).__init__()
        self.conv = Conv2d(3, 64, 7, 2, 3, norm="bn", act="relu")
        self.maxpool = nn.MaxPool2d(3, 2, 1)

    def forward(self, x):
        x = self.conv(x)
        x = self.maxpool(x)
        return x

class ResNetV1(nn.Module):
    def __init__(self, cfg_stages, classes=1000):
        super(ResNetV1, self).__init__()
        self.stem = BaseStem()

        self.stages = []
        in_channels = 0
        for name, params in cfg_stages.items():
            module = []
            for ic, c, oc, s in params:
                module.append(BottleneckV1(ic, c, oc, s))
                in_channels = oc
            self.add_module(name, nn.Sequential(*module))
            self.stages.append(name)

        self.pool = nn.AvgPool2d(7)

        self.output = nn.Linear(in_channels, classes)

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
            if isinstance(m, BottleneckV1):
                nn.init.constant_(m.conv3.norm.weight, 0)

    def forward(self, x):
        x = self.stem(x)
        for stage_name in self.stages:
            x = getattr(self, stage_name)(x)
        x = self.pool(x)
        x = x.flatten(start_dim=1)
        # x = x.view(x.size(0), -1)
        x = self.output(x)
        return x

def list2dict(cfg_list):
    cfg_stages = {}
    stage_num = 1
    cfg_stages['layer' + str(stage_num)] = []
    for ic, c, oc, s in cfg_list:
        if s == 2:
            stage_num += 1
            cfg_stages['layer' + str(stage_num)] = []

        cfg_stages['layer' + str(stage_num)].append([ic, c, oc, s])
    return cfg_stages

def resnet50_v1():
    #             ic    c    oc   s
    cfg_list = [[  64,  64,  256, 1],
                [ 256,  64,  256, 1],
                [ 256,  64,  256, 1],
                [ 256, 128,  512, 2],
                [ 512, 128,  512, 1],
                [ 512, 128,  512, 1],
                [ 512, 128,  512, 1],
                [ 512, 256, 1024, 2],
                [1024, 256, 1024, 1],
                [1024, 256, 1024, 1],
                [1024, 256, 1024, 1],
                [1024, 256, 1024, 1],
                [1024, 256, 1024, 1],
                [1024, 512, 2048, 2],
                [2048, 512, 2048, 1],
                [2048, 512, 2048, 1],
                ]
    cfg_stages = list2dict(cfg_list)
    return ResNetV1(cfg_stages)



