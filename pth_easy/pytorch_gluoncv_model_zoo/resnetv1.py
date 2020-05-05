from torch import nn
from ..nn import Conv2d_1x1, Conv2d

__all__ = [
    'ResNetV1',
    'resnet50_v1',
]

class BottleneckV1(nn.Module):
    def __init__(self, ic, c, oc, strides):
        super(BottleneckV1, self).__init__()
        self.conv1 = nn.Conv2d(ic, c, 1, strides, 0, bias=True)
        self.bn1 = nn.BatchNorm2d(c)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(c, c, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(c, oc, 1, bias=True)
        self.bn3 = nn.BatchNorm2d(oc)
        self.relu3 = nn.ReLU()
        if strides != 1 or ic != oc:
            self.downsample = nn.Sequential(
                    nn.Conv2d(ic, oc, 1, strides, bias=False),
                    nn.BatchNorm2d(oc)
                )
        else:
            self.downsample = None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

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

    def forward(self, x):
        x = self.stem(x)
        for stage_name in self.stages:
            x = getattr(self, stage_name)(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
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



