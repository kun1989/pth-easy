from torch import nn

__all__ = [
    'MobileNetV1',
    'mobilenet_v1_1_0',
    'mobilenet_v1_0_75',
    'mobilenet_v1_0_5',
    'mobilenet_v1_0_25'
]

def _add_conv(inp, oup, kernel=1, stride=1, pad=0, num_group=1, active=True):
    out = []
    out.append(nn.Conv2d(inp, oup, kernel, stride, pad, groups=num_group, bias=False))
    out.append(nn.BatchNorm2d(oup))
    if active:
        out.append(nn.ReLU(inplace=True))
    return nn.Sequential(*out)

def _add_conv_dw(dw_channels, channels, stride):
    conv = nn.Sequential(
         _add_conv(inp=dw_channels, oup=dw_channels, kernel=3, stride=stride, pad=1, num_group=dw_channels),
         _add_conv(inp=dw_channels, oup=channels))
    return conv

class MobileNetV1(nn.Module):
    def __init__(self, multiplier=1.0, classes=1000):
        super(MobileNetV1, self).__init__()
        self.features = nn.ModuleList([_add_conv(3, int(32 * multiplier), kernel=3, stride=2, pad=1)])

        dw_channels = [int(x * multiplier) for x in [32, 64] + [128] * 2 +
                       [256] *
                       2 +
                       [512] *
                       6 +
                       [1024]]
        channels = [int(x * multiplier) for x in [64] +
                    [128] * 2 + [256] * 2 + [512] * 6 + [1024] * 2]
        strides = [1, 2] * 3 + [1] * 5 + [2, 1]
        for dwc, c, s in zip(dw_channels, channels, strides):
            self.features.append(_add_conv_dw(dw_channels=dwc, channels=c, stride=s))
        self.features.append(nn.AvgPool2d(7))

        self.output = nn.Linear(int(1024 * multiplier), classes)

    def forward(self, x):
        for i, m in enumerate(self.features):
            x = m(x)
        x = x.view(x.size(0), -1)
        x = self.output(x)
        return x

def mobilenet_v1_1_0():
    return MobileNetV1(1.0)

def mobilenet_v1_0_75():
    return MobileNetV1(0.75)

def mobilenet_v1_0_5():
    return MobileNetV1(0.5)

def mobilenet_v1_0_25():
    return MobileNetV1(0.25)

