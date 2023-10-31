import torch.nn as nn

__all__ = ["ResNet_vd"]

# from toddleocr.modules.backbones.resnet.det_resnet import ResNet
# from toddleocr.ops import ConvBNLayer
#
#
# # class ResNet_vd(nn.Module):
# #
# #     def __init__(self, in_channels=3, layers=50, dcn_stage=None, out_indices=None, **kwargs):
# #         super().__init__()
# #         supported_layers = {18: [2, 2, 2, 2], 34: [3, 4, 6, 3], 50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3], 200: [3, 12, 48, 3]}
# #         depth = supported_layers[layers]
# #         num_features = [64, 256, 512, 1024] if layers >= 50 else [64, 64, 128, 256]
# #         num_filters = [64, 128, 256, 512]
# #         self.dcn_stage = dcn_stage or [False] * 4
# #         self.out_indices = out_indices or [0, 1, 2, 3]
# #         self.conv = nn.Sequential(
# #             ConvBNLayer(in_channels=in_channels, out_channels=32, kernel_size=3, stride=2, act='relu'),
# #             ConvBNLayer(in_channels=32, out_channels=32, kernel_size=3, stride=1, act='relu'),
# #             ConvBNLayer(in_channels=32, out_channels=64, kernel_size=3, stride=1, act='relu'),
# #         )
# #         self.pool2d_max = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
# #         self.stages = []
# #         self.out_channels = []
# #         block_class = BottleneckBlock if layers >= 50 else BasicBlock
# #         for block in range(len(depth)):
# #             block_list = []
# #             is_dcn = self.dcn_stage[block]
# #             downsample = ConvBNLayer(in_channels=in_channels, out_channels=num_filters[block], kernel_size=1, stride=1, is_vd_mode=True)
# #             for i in range(depth[block]):
# #                 if block == i == 0:
# #                     downsample = None
# #                 _block = block_class(in_channels=num_features[block] if i == 0 else num_filters[block] * block_class.expansion, out_channels=num_filters[block], stride=2 if i == 0 and block != 0 else 1, downsample=downsample, is_dcn=is_dcn)
# #                 self.add_module('bb_%d_%d' % (block, i), _block)
# #                 block_list.append(_block)
# #             if block in self.out_indices:
# #                 self.out_channels.append(num_filters[block] * block_class.expansion)
# #             self.stages.append(nn.Sequential(*block_list))
# #
# #     def forward(self, x):
# #         x = self.conv(x)
# #         x = self.pool2d_max(x)
# #         out = []
# #         for (i, block) in enumerate(self.stages):
# #             x = block(x)
# #             if i in self.out_indices:
# #                 out.append(x)
# #         return out
#
# class ResNet_vd(ResNet):
#     def convs(self):
#         return nn.Sequential(
#             ConvBNLayer(in_channels=self.in_channels, out_channels=32, kernel_size=3, stride=2, act='relu'),
#             ConvBNLayer(in_channels=32, out_channels=32, kernel_size=3, stride=1, act='relu'),
#             ConvBNLayer(in_channels=32, out_channels=64, kernel_size=3, stride=1, act='relu'),
#         )
# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d

# __all__ = ["ResNet_vd"]


class DeformableConvV2(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        groups=1,
        bias=None,
        skip_quant=False,
    ):
        super(DeformableConvV2, self).__init__()
        super().__init__()
        self.offset_channel = 2 * kernel_size**2 * groups
        self.mask_channel = kernel_size**2 * groups
        self.conv_dcn = DeformConv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2 * dilation,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.conv_offset = nn.Conv2d(
            in_channels,
            groups * 3 * kernel_size**2,
            kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            bias=bias,
        )
        if skip_quant:
            self.conv_offset.skip_quant = True

    def forward(self, x):
        offset_mask = self.conv_offset(x)
        (offset, mask) = torch.split(
            offset_mask, [self.offset_channel, self.mask_channel], dim=1
        )
        mask = F.sigmoid(mask)
        y = self.conv_dcn(x, offset, mask=mask)
        return y


class ConvBNLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        groups=1,
        dcn_groups=1,
        is_vd_mode=False,
        act=None,
        is_dcn=False,
    ):
        super().__init__()
        self.is_vd_mode = is_vd_mode
        self._pool2d_avg = nn.AvgPool2d(
            kernel_size=2, stride=2, padding=0, ceil_mode=True
        )
        if not is_dcn:
            self._conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size - 1) // 2,
                groups=groups,
                bias=False,
            )
        else:
            self._conv = DeformableConvV2(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                # padding=(kernel_size - 1) // 2,
                groups=dcn_groups,
                bias=False,
            )
        self._batch_norm = nn.BatchNorm2d(out_channels)
        self._act = act

    def forward(self, inputs):
        if self.is_vd_mode:
            inputs = self._pool2d_avg(inputs)
        y = self._conv(inputs)
        y = self._batch_norm(y)
        if self._act == "relu":
            y = F.relu(y)
        return y


class BottleneckBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        shortcut=True,
        if_first=False,
        is_dcn=False,
    ):
        super(BottleneckBlock, self).__init__()
        super().__init__()
        self.conv0 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            act="relu",
        )
        self.conv1 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            act="relu",
            is_dcn=is_dcn,
            dcn_groups=2,
        )
        self.conv2 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels * 4,
            kernel_size=1,
            act=None,
        )
        if not shortcut:
            self.short = ConvBNLayer(
                in_channels=in_channels,
                out_channels=out_channels * 4,
                kernel_size=1,
                stride=1,
                is_vd_mode=False if if_first else True,
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
    def __init__(
        self, in_channels, out_channels, stride, shortcut=True, is_first=False
    ):
        super(BasicBlock, self).__init__()
        super().__init__()
        self.stride = stride
        self.conv0 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            act="relu",
        )
        self.conv1 = ConvBNLayer(
            in_channels=out_channels, out_channels=out_channels, kernel_size=3, act=None
        )
        if not shortcut:
            self.short = ConvBNLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                is_vd_mode=not is_first,
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


class ResNet_vd(nn.Module):
    def __init__(
        self, in_channels=3, layers=50, dcn_stage=None, out_indices=None, **kwargs
    ):
        super().__init__()
        # self.layers = layers
        supported_layers = [18, 34, 50, 101, 152, 200]
        assert (
            layers in supported_layers
        ), "supported layers are {} but input layer is {}".format(
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
        else:
            raise ValueError("Unsupported layers {} for resnet".format(layers))

        num_features = [64, 256, 512, 1024] if layers >= 50 else [64, 64, 128, 256]
        num_filters = [64, 128, 256, 512]
        self.dcn_stage = (
            dcn_stage if dcn_stage is not None else [False, False, False, False]
        )
        self.out_indices = out_indices if out_indices is not None else [0, 1, 2, 3]
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
        if layers >= 50:
            for block in range(len(depth)):
                block_list = []
                shortcut = False
                is_dcn = self.dcn_stage[block]
                for i in range(depth[block]):
                    self.add_module(
                        "bb_%d_%d" % (block, i),
                        bottleneck_block := BottleneckBlock(
                            in_channels=num_features[block]
                            if i == 0
                            else num_filters[block] * 4,
                            out_channels=num_filters[block],
                            stride=2 if i == 0 and block != 0 else 1,
                            shortcut=shortcut,
                            if_first=block == i == 0,
                            is_dcn=is_dcn,
                        ),
                    )
                    shortcut = True
                    block_list.append(bottleneck_block)
                if block in self.out_indices:
                    self.out_channels.append(num_filters[block] * 4)
                self.stages.append(nn.Sequential(*block_list))
        else:
            for block in range(len(depth)):
                block_list = []
                shortcut = False
                for i in range(depth[block]):
                    self.add_module(
                        "bb_%d_%d" % (block, i),
                        basic_block := BasicBlock(
                            in_channels=num_features[block]
                            if i == 0
                            else num_filters[block],
                            out_channels=num_filters[block],
                            stride=2 if i == 0 and block != 0 else 1,
                            shortcut=shortcut,
                            is_first=block == i == 0,
                        ),
                    )
                    shortcut = True
                    block_list.append(basic_block)
                if block in self.out_indices:
                    self.out_channels.append(num_filters[block])
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
