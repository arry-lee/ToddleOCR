from typing import Callable, Optional

import torch

# from torchvision.models.resnet import Bottleneck, BasicBlock, ResNet, resnet50
import torch.nn.functional as F
from torch import nn

from toddleocr.ops import ConvBNLayer


class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        downsample=None,
        name=None,
        is_dcn=False,
    ):
        super().__init__()

        self.conv0 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            act="relu",
            name=name + "_branch2a" if name else None,
        )
        self.conv1 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            act="relu",
            name=name + "_branch2b" if name else None,
            is_dcn=is_dcn,
            dcn_groups=2,
        )
        self.conv2 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels * self.expansion,
            kernel_size=1,
            act=None,
            name=name + "_branch2c" if name else None,
        )
        #
        # if not downsample:
        #     self.short = ConvBNLayer(
        #         in_channels=in_channels,
        #         out_channels=out_channels * 4,
        #         kernel_size=1,
        #         stride=1,
        #         is_vd_mode=False if if_first else True,
        #         name=name + "_branch1" if name else None,
        #     )

        self.downsample = downsample

    def forward(self, inputs):
        y = self.conv0(inputs)
        conv1 = self.conv1(y)
        conv2 = self.conv2(conv1)

        if self.downsample:
            short = self.downsample(inputs)
        else:
            short = inputs
        y = torch.add(short, conv2)
        y = F.relu(y)
        return y


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self, in_channels, out_channels, stride, downsample=None, name=None, **kwargs
    ):
        super().__init__()
        self.stride = stride
        self.conv0 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            act="relu",
            name=name + "_branch2a" if name else None,
        )
        self.conv1 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            act=None,
            name=name + "_branch2b" if name else None,
        )
        self.donwnsample = downsample
        # if not shortcut:
        #     self.short = ConvBNLayer(
        #         in_channels=in_channels,
        #         out_channels=out_channels,
        #         kernel_size=1,
        #         stride=1,
        #         is_vd_mode=False if if_first else True,
        #         name=name + "_branch1" if name else None,
        #     )
        #
        # self.shortcut = shortcut

    def forward(self, inputs):
        y = self.conv0(inputs)
        conv1 = self.conv1(y)
        if self.downsample:
            short = self.downsample(inputs)
        else:
            short = inputs
        # if self.shortcut:
        #     short = inputs
        # else:
        #     short = self.short(inputs)
        y = torch.add(short, conv1)
        y = F.relu(y)
        return y


class ResNet_vd(nn.Module):
    def __init__(
        self,
        in_channels=3,
        layers=50,
        dcn_stage=None,
        out_indices=None,
        **kwargs,
    ):
        super().__init__()
        # self.layers = layers
        supported_layers = {
            18: [2, 2, 2, 2],
            34: [3, 4, 6, 3],
            50: [3, 4, 6, 3],
            101: [3, 4, 23, 3],
            152: [3, 8, 36, 3],
            200: [3, 12, 48, 3],
        }

        depth = supported_layers[layers]

        num_features = [64, 64, 128, 256] if layers < 50 else [64, 256, 512, 1024]

        num_filters = [64, 128, 256, 512]

        self.dcn_stage = dcn_stage or [False] * 4
        self.out_indices = out_indices or [0, 1, 2, 3]

        self.conv1_1 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=32,
            kernel_size=3,
            stride=2,
            act="relu",
        )
        self.conv1_2 = ConvBNLayer(
            in_channels=32, out_channels=32, kernel_size=3, stride=1, act="relu"
        )
        self.conv1_3 = ConvBNLayer(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, act="relu"
        )
        self.pool2d_max = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stages = []
        self.out_channels = []

        # expansion = 4 if layers >= 50 else 1
        block_class = BottleneckBlock if layers >= 50 else BasicBlock

        # if layers >= 50:
        for block in range(len(depth)):
            block_list = []
            # shortcut = False
            is_dcn = self.dcn_stage[block]
            downsample = ConvBNLayer(
                in_channels=in_channels,
                out_channels=num_filters[block],
                kernel_size=1,
                stride=1,
                is_vd_mode=True,
            )
            for i in range(depth[block]):
                if block == i == 0:
                    downsample = None
                _block = block_class(
                    in_channels=num_features[block]
                    if i == 0
                    else num_filters[block] * block_class.expansion,
                    out_channels=num_filters[block],
                    stride=2 if i == 0 and block != 0 else 1,
                    downsample=downsample,
                    is_dcn=is_dcn,
                )

                self.add_module("bb_%d_%d" % (block, i), _block)
                block_list.append(_block)
            if block in self.out_indices:
                self.out_channels.append(num_filters[block] * block_class.expansion)
            self.stages.append(nn.Sequential(*block_list))

    def forward(self, inputs):
        y = self.conv1_1(inputs)
        y = self.conv1_2(y)
        y = self.conv1_3(y)
        y = self.pool2d_max(y)
        out = []
        for i, block in enumerate(self.stages):
            y = block(y)
            if i in self.out_indices:
                out.append(y)
        return out
