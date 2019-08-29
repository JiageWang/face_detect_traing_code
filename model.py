import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv1x1(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )



def conv_dw(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU6(inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True),
    )


class ResBlock(nn.Module):
    def __init__(self, inp, oup, stride):
        super().__init__()
        self.conv1 = conv_dw(inp, oup, stride)
        self.conv2 = conv_dw(inp, oup, stride)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return F.relu6(out + x)


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 3 x 640 x640
        self.base = nn.Sequential(
            conv_dw(3, 32, 2),
            conv_dw(32, 32, 1),
            # 32 x 320 x 320 ==> 1:2
            conv_dw(32, 64, 2),
            # 64 x 160 x 160 ==> 1:4
            conv_dw(64, 64, 1),
            conv_dw(64, 64, 2),
            # 64 x 80 x 80 ==> 1:8
        )
        self.features = nn.Sequential(
            nn.Sequential(
                conv_dw(64, 64, 2),
                ResBlock(64, 64, 1),
                ResBlock(64, 64, 1),
            ),
            # 64 x 40 x 40 ==>> 1:16
            nn.Sequential(
                conv_dw(64, 64, 2),
                ResBlock(64, 64, 1),
                ResBlock(64, 64, 1),
            ),
            # 64 x 20 x 20  ==>> 1:32
            nn.Sequential(
                conv_dw(64, 128, 2),
                ResBlock(128, 128, 1),
                ResBlock(128, 128, 1),
            ),
            # 128 x 10 x 10  ==>> 1:64
            nn.Sequential(
                conv_dw(128, 128, 2),
                ResBlock(128, 128, 1),
                ResBlock(128, 128, 1),
            ),
            # 128 x 5 x 5  ==>> 1:128
            nn.Sequential(
                conv_dw(128, 128, 2),
                ResBlock(128, 128, 1),
                ResBlock(128, 128, 1),
            ),
            # 128 x 3 x 3  ==>> 1:213
            nn.Sequential(
                conv_dw(128, 128, 2),
                ResBlock(128, 128, 1),
                ResBlock(128, 128, 1),
            )
            # 128 x 1 x 1  ==>> 1:640
        )

        self.branchs = nn.Sequential(
            nn.Sequential(
                conv1x1(64, 32),
                conv1x1(32, 5)
            ),
            nn.Sequential(
                conv1x1(64, 32),
                conv1x1(32, 5)
            ),
            nn.Sequential(
                conv1x1(128, 64),
                conv1x1(64, 32),
                conv1x1(32, 5)
            ),
            nn.Sequential(
                conv1x1(128, 64),
                conv1x1(64, 32),
                conv1x1(32, 5)
            ),
            nn.Sequential(
                conv1x1(128, 64),
                conv1x1(64, 32),
                conv1x1(32, 5)
            ),
            nn.Sequential(
                conv1x1(128, 64),
                conv1x1(64, 32),
                conv1x1(32, 5)
            ))
        self.weight_init()

    def forward(self, x):
        x = self.base(x)
        features = []
        for feature in self.features:
            x = feature(x)
            features.append(x)
        out = [branch(feature) for branch, feature in zip(self.branchs, features)]
        return out

    def weight_init(self):
        for layer in self.modules():
            self._layer_init(layer)

    def _layer_init(self, m):
        # 使用isinstance来判断m属于什么类型
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, np.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()

if __name__ == "__main__":
    net = ConvNet().train()
    summary(net.cuda(), input_size=(3, 640, 640), batch_size=1, device='cuda')
