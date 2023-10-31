"""
This code is refer from:
https://github.com/LBH1024/CAN/models/densenet.py

"""


import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    def __init__(self, in_channels, grow_rate, use_dropout):
        super().__init__()
        interChannels = 4 * grow_rate
        self.bn1 = nn.BatchNorm2d(interChannels)
        self.conv1 = nn.Conv2d(
            in_channels, interChannels, kernel_size=1, bias=False
        )  # Xavier initialization
        self.bn2 = nn.BatchNorm2d(grow_rate)
        self.conv2 = nn.Conv2d(
            interChannels, grow_rate, kernel_size=3, padding=1, bias=False
        )  # Xavier initialization
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        if self.use_dropout:
            out = self.dropout(out)
        out = F.relu(self.bn2(self.conv2(out)))
        if self.use_dropout:
            out = self.dropout(out)
        out = torch.concat([x, out], 1)
        return out


class SingleLayer(nn.Module):
    def __init__(self, in_channels, grow_rate, use_dropout):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels, grow_rate, kernel_size=3, padding=1, bias=False
        )

        self.use_dropout = use_dropout
        if use_dropout:
            self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        out = self.conv1(F.relu(x))
        if self.use_dropout:
            out = self.dropout(out)

        out = torch.concat([x, out], 1)
        return out


class Transition(nn.Module):
    def __init__(self, in_channels, out_channels, use_dropout):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.use_dropout = use_dropout
        if use_dropout:
            self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        if self.use_dropout:
            out = self.dropout(out)
        out = F.avg_pool2d(out, 2, ceil_mode=True, exclusive=False)
        return out


class DenseNet(nn.Module):
    def __init__(
        self, grow_rate, reduction, bottleneck, use_dropout, input_channel, **kwargs
    ):
        super().__init__()

        nDenseBlocks = 16
        in_channels = 2 * grow_rate

        self.conv1 = nn.Conv2d(
            input_channel, in_channels, kernel_size=7, padding=3, stride=2, bias=False
        )
        self.dense1 = self._make_dense(
            in_channels, grow_rate, nDenseBlocks, bottleneck, use_dropout
        )
        in_channels += nDenseBlocks * grow_rate
        out_channels = int(math.floor(in_channels * reduction))
        self.trans1 = Transition(in_channels, out_channels, use_dropout)

        in_channels = out_channels
        self.dense2 = self._make_dense(
            in_channels, grow_rate, nDenseBlocks, bottleneck, use_dropout
        )
        in_channels += nDenseBlocks * grow_rate
        out_channels = int(math.floor(in_channels * reduction))
        self.trans2 = Transition(in_channels, out_channels, use_dropout)

        in_channels = out_channels
        self.dense3 = self._make_dense(
            in_channels, grow_rate, nDenseBlocks, bottleneck, use_dropout
        )
        self.out_channels = out_channels

    def _make_dense(
        self, in_channels, grow_rate, nDenseBlocks, bottleneck, use_dropout
    ):
        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.append(Bottleneck(in_channels, grow_rate, use_dropout))
            else:
                layers.append(SingleLayer(in_channels, grow_rate, use_dropout))
            in_channels += grow_rate
        return nn.Sequential(*layers)

    def forward(self, inputs):
        x, x_m, y = inputs
        out = self.conv1(x)
        out = F.relu(out)
        out = F.max_pool2d(out, 2, ceil_mode=True)
        out = self.dense1(out)
        out = self.trans1(out)
        out = self.dense2(out)
        out = self.trans2(out)
        out = self.dense3(out)
        return out, x_m, y
