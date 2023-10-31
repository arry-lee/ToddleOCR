import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

__all__ = ["ResNetFPN"]

from toddleocr.ops import ConvBNLayer


class ResNetFPN(nn.Module):
    def __init__(self, in_channels=1, layers=50, **kwargs):
        super().__init__()
        supported_layers = {
            18: {"depth": [2, 2, 2, 2], "block_class": BasicBlock},
            34: {"depth": [3, 4, 6, 3], "block_class": BasicBlock},
            50: {"depth": [3, 4, 6, 3], "block_class": BottleneckBlock},
            101: {"depth": [3, 4, 23, 3], "block_class": BottleneckBlock},
            152: {"depth": [3, 8, 36, 3], "block_class": BottleneckBlock},
        }
        stride_list = [(2, 2), (2, 2), (1, 1), (1, 1)]
        num_filters = [64, 128, 256, 512]
        self.depth = supported_layers[layers]["depth"]
        self.F = []
        self.conv = ConvBNLayer(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=7,
            stride=2,
            act="relu",
            name="conv1",
        )
        self.block_list = []
        in_ch = 64
        if layers >= 50:
            for block in range(len(self.depth)):
                for i in range(self.depth[block]):
                    if layers in [101, 152] and block == 2:
                        if i == 0:
                            conv_name = "res" + str(block + 2) + "a"
                        else:
                            conv_name = "res" + str(block + 2) + "b" + str(i)
                    else:
                        conv_name = "res" + str(block + 2) + chr(97 + i)
                    block_list = self.add_module(
                        "bottleneckBlock_{}_{}".format(block, i),
                        BottleneckBlock(
                            in_channels=in_ch,
                            out_channels=num_filters[block],
                            stride=stride_list[block] if i == 0 else 1,
                            name=conv_name,
                        ),
                    )
                    in_ch = num_filters[block] * 4
                    self.block_list.append(block_list)
                self.F.append(block_list)
        else:
            for block in range(len(self.depth)):
                for i in range(self.depth[block]):
                    conv_name = "res" + str(block + 2) + chr(97 + i)
                    if i == 0 and block != 0:
                        stride = (2, 1)
                    else:
                        stride = (1, 1)
                    basic_block = self.add_module(
                        conv_name,
                        BasicBlock(
                            in_channels=in_ch,
                            out_channels=num_filters[block],
                            stride=stride_list[block] if i == 0 else 1,
                            is_first=block == i == 0,
                            name=conv_name,
                        ),
                    )
                    in_ch = basic_block.out_channels
                    self.block_list.append(basic_block)
        out_ch_list = [in_ch // 4, in_ch // 2, in_ch]
        self.base_block = []
        self.conv_trans = []
        self.bn_block = []
        for i in [-2, -3]:
            in_channels = out_ch_list[i + 1] + out_ch_list[i]

            self.base_block.append(
                self.add_module(
                    "F_{}_base_block_0".format(i),
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_ch_list[i],
                        kernel_size=1,
                        bias=True,
                    ),
                )
            )
            self.base_block.append(
                self.add_module(
                    "F_{}_base_block_1".format(i),
                    nn.Conv2d(
                        in_channels=out_ch_list[i],
                        out_channels=out_ch_list[i],
                        kernel_size=3,
                        padding=1,
                    ),
                )
            )
            self.base_block.append(
                self.add_module(
                    "F_{}_base_block_2".format(i),
                    nn.BatchNorm2d(out_ch_list[i]),
                )
            )
        self.base_block.append(
            self.add_module(
                "F_{}_base_block_3".format(i),
                nn.Conv2d(in_channels=out_ch_list[i], out_channels=512, kernel_size=1),
            )
        )
        self.out_channels = 512

    def __call__(self, x):
        x = self.conv(x)
        fpn_list = []
        F = []
        for i in range(len(self.depth)):
            fpn_list.append(np.sum(self.depth[: i + 1]))

        block: BasicBlock
        for i, block in enumerate(self.block_list):
            x = block(x)
            for number in fpn_list:
                if i + 1 == number:
                    F.append(x)
        base = F[-1]

        j = 0
        for i, block in enumerate(self.base_block):
            if i % 3 == 0 and i < 6:
                j = j + 1
                b, c, w, h = F[-j - 1].shape
                if [w, h] == list(base.shape[2:]):
                    base = base
                else:
                    base = self.conv_trans[j - 1](base)
                    base = self.bn_block[j - 1](base)
                base = torch.concat([base, F[-j - 1]], dim=1)
            base = block(base)
        return base


class ShortCut(nn.Module):
    def __init__(self, in_channels, out_channels, stride, name, is_first=False):
        super().__init__()
        self.use_conv = True

        if in_channels != out_channels or stride != 1 or is_first == True:
            if stride == (1, 1):
                self.conv = ConvBNLayer(in_channels, out_channels, 1, 1, name=name)
            else:  # stride==(2,2)
                self.conv = ConvBNLayer(in_channels, out_channels, 1, stride, name=name)
        else:
            self.use_conv = False

    def forward(self, x):
        if self.use_conv:
            x = self.conv(x)
        return x


class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, name):
        super().__init__()
        self.conv0 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            act="relu",
            name=name + "_branch2a",
        )
        self.conv1 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            act="relu",
            name=name + "_branch2b",
        )

        self.conv2 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels * 4,
            kernel_size=1,
            act=None,
            name=name + "_branch2c",
        )

        self.short = ShortCut(
            in_channels=in_channels,
            out_channels=out_channels * 4,
            stride=stride,
            is_first=False,
            name=name + "_branch1",
        )
        self.out_channels = out_channels * 4

    def forward(self, x):
        y = self.conv0(x)
        y = self.conv1(y)
        y = self.conv2(y)
        y = y + self.short(x)
        y = F.relu(y)
        return y


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, name, is_first):
        super().__init__()
        self.conv0 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            act="relu",
            stride=stride,
            name=name + "_branch2a",
        )
        self.conv1 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            act=None,
            name=name + "_branch2b",
        )
        self.short = ShortCut(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            is_first=is_first,
            name=name + "_branch1",
        )
        self.out_channels = out_channels

    def forward(self, x):
        y = self.conv0(x)
        y = self.conv1(y)
        y = y + self.short(x)
        return F.relu(y)
