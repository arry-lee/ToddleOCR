import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models.resnet import ResNet

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
        self.downsample = downsample

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        if self.downsample:
            short = self.downsample(x)
        else:
            short = x
        x = torch.add(short, x)
        x = F.relu(x)
        return x


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
        self.downsample = downsample

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        if self.downsample:
            short = self.downsample(x)
        else:
            short = x
        x = torch.add(short, x)
        x = F.relu(x)
        return x


class ResNet(nn.Module):
    supported_layers = {
        18: [2, 2, 2, 2],
        34: [3, 4, 6, 3],
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3],
        200: [3, 12, 48, 3],
    }
    num_filters = [64, 128, 256, 512]
    num_features = [64, 256, 512, 1024]
    out_channels = []

    def __init__(self, in_channels=3, layers=50, out_indices=None, dcn_stage=None):
        super().__init__()
        self.layers = layers
        self.in_channels = in_channels
        depth = self.supported_layers[layers]
        num_features = [64, 64, 128, 256] if layers < 50 else self.num_features
        num_filters = self.num_filters
        self.dcn_stage = dcn_stage or [False] * 4
        self.out_indices = out_indices or [0, 1, 2, 3]
        self.conv = self.convs()
        self.pool2d_max = self.pool()
        self.stages = []
        block_class = BottleneckBlock if layers >= 50 else BasicBlock
        for block in range(len(depth)):
            b = self._make_layer(block_class, block, depth, num_features, num_filters)
            self.stages.append(b)

    def _make_layer(self, block_class, bno, depth, num_features, num_filters):
        layers = []
        is_dcn = self.dcn_stage[bno]
        downsample = None
        out_channels = num_filters[bno]
        for i in range(depth[bno]):
            conv_name = self.format_name(bno, i)
            if i == 0:
                in_channels = num_features[bno]
            else:
                in_channels = num_filters[bno] * block_class.expansion
            if i != 0 or bno != 0:
                downsample = self.downsample(in_channels, out_channels)
            if i == 0 and bno != 0:
                stride = 2
            else:
                stride = 1
            _block = block_class(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                downsample=downsample,
                is_dcn=is_dcn,
                name=conv_name,
            )
            self.add_module("bb_%d_%d" % (bno, i), _block)
            layers.append(_block)
        if bno in self.out_indices:
            self.out_channels.append(num_filters[bno] * block_class.expansion)
        return nn.Sequential(*layers)

    def convs(self):
        return ConvBNLayer(
            in_channels=self.in_channels,
            out_channels=64,
            kernel_size=7,
            stride=2,
            act="relu",
        )

    def pool(self):
        return nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def downsample(self, in_channels, out_channels):
        return ConvBNLayer(
            in_channels, out_channels, kernel_size=1, stride=1, is_vd_mode=True
        )

    def format_name(self, *args):
        return None

    def forward(self, x):
        x = self.conv(x)
        x = self.pool2d_max(x)
        out = []
        for i, block in enumerate(self.stages):
            x = block(x)
            if i in self.out_indices:
                out.append(x)
        return out
