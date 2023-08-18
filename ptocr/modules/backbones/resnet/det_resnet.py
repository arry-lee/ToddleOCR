import torch
import torch.nn.functional as F
from torch import nn
from ptocr.ops import ConvBNLayer

class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride, downsample=None, name=None, is_dcn=False):
        super().__init__()
        self.conv0 = ConvBNLayer(in_channels=in_channels, out_channels=out_channels, kernel_size=1, act='relu', name=name + '_branch2a' if name else None)
        self.conv1 = ConvBNLayer(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=stride, act='relu', name=name + '_branch2b' if name else None, is_dcn=is_dcn, dcn_groups=2)
        self.conv2 = ConvBNLayer(in_channels=out_channels, out_channels=out_channels * self.expansion, kernel_size=1, act=None, name=name + '_branch2c' if name else None)
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

    def __init__(self, in_channels, out_channels, stride, downsample=None, name=None, **kwargs):
        super().__init__()
        self.stride = stride
        self.conv0 = ConvBNLayer(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, act='relu', name=name + '_branch2a' if name else None)
        self.conv1 = ConvBNLayer(in_channels=out_channels, out_channels=out_channels, kernel_size=3, act=None, name=name + '_branch2b' if name else None)
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
    supported_layers = {18: [2, 2, 2, 2], 34: [3, 4, 6, 3], 50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3], 200: [3, 12, 48, 3]}
    num_filters = [64, 128, 256, 512]
    num_features = [64, 256, 512, 1024]

    def __init__(self, in_channels=3, layers=50, out_indices=None, dcn_stage=None):
        super().__init__()
        self.layers = layers
        depth = self.supported_layers[layers]
        num_features = [64, 64, 128, 256] if layers < 50 else self.num_features
        num_filters = self.num_filters
        self.dcn_stage = dcn_stage or [False] * 4
        self.out_indices = out_indices or [0, 1, 2, 3]
        self.pool2d_max = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stages = []
        self.out_channels = []
        block_class = BottleneckBlock if layers >= 50 else BasicBlock
        for block in range(len(depth)):
            block_list = []
            is_dcn = self.dcn_stage[block]
            downsample = ConvBNLayer(in_channels=in_channels, out_channels=num_filters[block], kernel_size=1, stride=1, is_vd_mode=True)
            for i in range(depth[block]):
                conv_name = self.format_name(block, i)
                if block == i == 0:
                    downsample = None
                _block = block_class(in_channels=num_features[block] if i == 0 else num_filters[block] * block_class.expansion, out_channels=num_filters[block], stride=2 if i == 0 and block != 0 else 1, downsample=downsample, is_dcn=is_dcn, name=conv_name)
                self.add_module('bb_%d_%d' % (block, i), _block)
                block_list.append(_block)
            if block in self.out_indices:
                self.out_channels.append(num_filters[block] * block_class.expansion)
            self.stages.append(nn.Sequential(*block_list))

        self.conv = ConvBNLayer(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, act='relu')

    def format_name(self, *args):
        return None

    def forward(self, x):
        x = self.conv(x)
        x = self.pool2d_max(x)
        out = []
        for (i, block) in enumerate(self.stages):
            x = block(x)
            if i in self.out_indices:
                out.append(x)
        return out
