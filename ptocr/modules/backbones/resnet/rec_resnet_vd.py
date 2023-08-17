import torch

import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ResNet"]

from ptocr.ops import ConvBNLayer


class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, shortcut=True, if_first=False, name=None):
        super().__init__()

        self.conv0 = ConvBNLayer(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1, act="relu", name=name + "_branch2a"
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
            in_channels=out_channels, out_channels=out_channels * 4, kernel_size=1, act=None, name=name + "_branch2c"
        )

        if not shortcut:
            self.short = ConvBNLayer(
                in_channels=in_channels,
                out_channels=out_channels * 4,
                kernel_size=1,
                stride=stride,
                is_vd_mode=not if_first and stride[0] != 1,
                name=name + "_branch1",
            )

        self.shortcut = shortcut

    def forward(self, inputs):
        y = self.conv0(inputs)

        conv1 = self.conv1(y)
        conv2 = self.conv2(conv1)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)
        y = torch.add(short, conv2)
        y = F.relu(y)
        return y


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, shortcut=True, if_first=False, name=None):
        super().__init__()
        self.stride = stride
        self.conv0 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            act="relu",
            name=name + "_branch2a",
        )
        self.conv1 = ConvBNLayer(
            in_channels=out_channels, out_channels=out_channels, kernel_size=3, act=None, name=name + "_branch2b"
        )

        if not shortcut:
            self.short = ConvBNLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                is_vd_mode=not if_first and stride[0] != 1,
                name=name + "_branch1",
            )

        self.shortcut = shortcut

    def forward(self, inputs):
        y = self.conv0(inputs)
        conv1 = self.conv1(y)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)
        y = torch.add(short, conv1)
        y = F.relu(y)
        return y


class ResNet(nn.Module):
    def __init__(self, in_channels=3, layers=50, **kwargs):
        super().__init__()

        self.layers = layers
        supported_layers = [18, 34, 50, 101, 152, 200]
        assert layers in supported_layers, "supported layers are {} but input layer is {}".format(
            supported_layers, layers
        )

        if layers == 18:
            depth = [2, 2, 2, 2]
        elif layers == 34 or layers == 50:
            depth = [3, 4, 6, 3]
        elif layers == 101:
            depth = [3, 4, 23, 3]
        elif layers == 152:
            depth = [3, 8, 36, 3]
        elif layers == 200:
            depth = [3, 12, 48, 3]
        num_features = [64, 256, 512, 1024] if layers >= 50 else [64, 64, 128, 256]
        num_filters = [64, 128, 256, 512]

        self.conv1_1 = ConvBNLayer(
            in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, act="relu", name="conv1_1"
        )
        self.conv1_2 = ConvBNLayer(in_channels=32, out_channels=32, kernel_size=3, stride=1, act="relu", name="conv1_2")
        self.conv1_3 = ConvBNLayer(in_channels=32, out_channels=64, kernel_size=3, stride=1, act="relu", name="conv1_3")
        self.pool2d_max = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.block_list = []
        if layers >= 50:
            for block in range(len(depth)):
                shortcut = False
                for i in range(depth[block]):
                    if layers in [101, 152, 200] and block == 2:
                        if i == 0:
                            conv_name = "res" + str(block + 2) + "a"
                        else:
                            conv_name = "res" + str(block + 2) + "b" + str(i)
                    else:
                        conv_name = "res" + str(block + 2) + chr(97 + i)

                    if i == 0 and block != 0:
                        stride = (2, 1)
                    else:
                        stride = (1, 1)
                    bottleneck_block = self.add_module(
                        "bb_%d_%d" % (block, i),
                        BottleneckBlock(
                            in_channels=num_features[block] if i == 0 else num_filters[block] * 4,
                            out_channels=num_filters[block],
                            stride=stride,
                            shortcut=shortcut,
                            if_first=block == i == 0,
                            name=conv_name,
                        ),
                    )
                    shortcut = True
                    self.block_list.append(bottleneck_block)
                self.out_channels = num_filters[block] * 4
        else:
            for block in range(len(depth)):
                shortcut = False
                for i in range(depth[block]):
                    conv_name = "res" + str(block + 2) + chr(97 + i)
                    if i == 0 and block != 0:
                        stride = (2, 1)
                    else:
                        stride = (1, 1)

                    basic_block = self.add_module(
                        "bb_%d_%d" % (block, i),
                        BasicBlock(
                            in_channels=num_features[block] if i == 0 else num_filters[block],
                            out_channels=num_filters[block],
                            stride=stride,
                            shortcut=shortcut,
                            if_first=block == i == 0,
                            name=conv_name,
                        ),
                    )
                    shortcut = True
                    self.block_list.append(basic_block)
                self.out_channels = num_filters[block]
        self.out_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, inputs):
        y = self.conv1_1(inputs)
        y = self.conv1_2(y)
        y = self.conv1_3(y)
        y = self.pool2d_max(y)
        for block in self.block_list:
            y = block(y)
        y = self.out_pool(y)
        return y
