from torch import nn

__all__ = [
    'MobileNetV2',
    'mobilenet_v2_1_0',
    'mobilenet_v2_0_75',
    'mobilenet_v2_0_5',
    'mobilenet_v2_0_25'
]

def _add_conv(inp, oup, kernel=1, stride=1, pad=0, num_group=1, active=True):
    out = []
    out.append(nn.Conv2d(inp, oup, kernel, stride, pad, groups=num_group, bias=False))
    out.append(nn.BatchNorm2d(oup))
    if active:
        out.append(nn.ReLU6(inplace=True))
    return nn.Sequential(*out)

class LinearBottleneck(nn.Module):
    def __init__(self, in_channels, channels, t, stride):
        super(LinearBottleneck, self).__init__()
        self.use_shortcut = stride == 1 and in_channels == channels

        hidden_dim = in_channels * t

        if t != 1:
            self.conv = nn.Sequential(
                _add_conv(in_channels, hidden_dim),
                _add_conv(hidden_dim,
                          hidden_dim,
                          kernel=3,
                          stride=stride,
                          pad=1,
                          num_group=hidden_dim),
                _add_conv(hidden_dim, channels, active=False)
            )
        else:
            self.conv = nn.Sequential(
                _add_conv(hidden_dim,
                          hidden_dim,
                          kernel=3,
                          stride=stride,
                          pad=1,
                          num_group=hidden_dim),
                _add_conv(hidden_dim, channels, active=False)
            )

    def forward(self, x):
        out = self.conv(x)
        if self.use_shortcut:
            out = out + x
        return out

class MobileNetV2(nn.Module):
    def __init__(self, multiplier=1.0, classes=1000):
        super(MobileNetV2, self).__init__()
        self.features = nn.ModuleList([_add_conv(3, int(32 * multiplier), kernel=3, stride=2, pad=1)])

        in_channels_group = [int(x * multiplier) for x in [32] + [16] + [24] * 2
                            + [32] * 3 + [64] * 4 + [96] * 3 + [160] * 3]
        channels_group = [int(x * multiplier) for x in [16] + [24] * 2 + [32] * 3
                            + [64] * 4 + [96] * 3 + [160] * 3 + [320]]
        ts = [1] + [6] * 16
        strides = [1, 2] * 2 + [1, 1, 2] + [1] * 6 + [2] + [1] * 3

        for in_c, c, t, s in zip(in_channels_group, channels_group, ts, strides):
            self.features.append(LinearBottleneck(in_c, c, t, s))

        last_channels = int(1280 * multiplier) if multiplier > 1.0 else 1280
        self.features.append(_add_conv(int(320 * multiplier), last_channels))

        self.features.append(nn.AvgPool2d(7))

        self.output = nn.Conv2d(last_channels, classes, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        for i, m in enumerate(self.features):
            x = m(x)
        x = self.output(x)
        x = x.view(x.size(0), -1)
        return x


def mobilenet_v2_1_0():
    return MobileNetV2(1.0)

def mobilenet_v2_0_75():
    return MobileNetV2(0.75)

def mobilenet_v2_0_5():
    return MobileNetV2(0.5)

def mobilenet_v2_0_25():
    return MobileNetV2(0.25)
