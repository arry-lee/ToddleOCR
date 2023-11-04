# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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

# This code is refer from: https://github.com/PaddlePaddle/PaddleClas/blob/develop/ppcls/arch/backbone/legendary_models/pp_lcnet.py


import torch

import torch.nn as nn
import torch.nn.functional as F
from torch.nn import AdaptiveAvgPool2d, BatchNorm2d, Conv2d
from torch.nn.functional import hardsigmoid

from toddleocr.ops.misc import DeformableConvV2


class Hardswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


class Hardsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        # return (1.2 * x).add_(3.).clamp_(0., 6.).div_(6.)
        return F.relu6(x + 3.0, inplace=True) / 6.0


class ConvBNLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        groups=1,
        bias=False,
        act=None,  # 激活函数名
        name=None,
        **kwargs
        # 不常用的东西放在kwargs里
        # is_dcn=False,  # 是否使用deformable conv network
        # dcn_groups=1,
        # is_vd_mode=False,  # 是否使用平均池化
    ):
        super().__init__()
        self.is_vd_mode = kwargs.get("is_vd_mode", False)

        if self.is_vd_mode:
            stride = 1
            self._pool2d_avg = nn.AvgPool2d(
                kernel_size=2, stride=2, padding=0, ceil_mode=True
            )

        is_dcn = kwargs.get("is_dcn", False)
        if not is_dcn:
            if stride == (1, 1):
                kernel_size = 2
                dilation = 2

            padding = kwargs.get("padding", (kernel_size - 1) // 2)

            self.conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding,
                groups=groups,
                bias=bias,
            )
            if kwargs.get("initializer", None) is not None:
                initializer = kwargs.get("initializer")
                initializer(self.conv.weight)
        else:
            dcn_groups = kwargs.get("dcn_groups", 1)
            self.conv = DeformableConvV2(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                groups=dcn_groups,
                bias=bias,
            )

        self.bn = nn.BatchNorm2d(num_features=out_channels)

        # if name is not None:
        #     if name == "conv1":
        #         bn_name = "bn_" + name
        #     else:
        #         bn_name = "bn" + name[3:]
        #     self.bn.register_buffer(bn_name + "_mean", self.bn.running_mean)
        #     self.bn.register_buffer(bn_name + "_variance", self.bn.running_var)

        if isinstance(act, str):
            self.act = getattr(F, act)
        elif issubclass(act, nn.Module):
            self.act = act()
        elif callable(act):
            self.act = act
        else:
            self.act = None

    def forward(self, x):
        if self.is_vd_mode:
            x = self._pool2d_avg(x)
        x = self.conv(x)
        x = self.bn(x)
        if self.act:
            x = self.act(x)
        return x


class DepthWiseSeparable(nn.Module):
    def __init__(
        self,
        num_features,
        num_filters1,
        num_filters2,
        num_groups,
        stride,
        scale,
        dw_size=3,
        padding=1,
        use_se=False,
    ):
        super().__init__()
        self.use_se = use_se
        self.dw_conv = ConvBNLayer(
            in_channels=num_features,
            out_channels=int(num_filters1 * scale),
            kernel_size=dw_size,
            stride=stride,
            padding=padding,
            groups=int(num_groups * scale),
            act=Hardswish,
        )
        if use_se:
            self.se = SEModule(int(num_filters1 * scale))
        self.pw_conv = ConvBNLayer(
            in_channels=int(num_filters1 * scale),
            out_channels=int(num_filters2 * scale),
            kernel_size=1,
            stride=1,
            padding=0,
            act=Hardswish,
        )

    def forward(self, x):
        x = self.dw_conv(x)
        if self.use_se:
            x = self.se(x)
        x = self.pw_conv(x)
        return x


class MobileNetV1Enhance(nn.Module):
    def __init__(
        self,
        in_channels=3,
        scale=0.5,
        last_conv_stride=1,
        last_pool_type="max",
        **kwargs
    ):
        super().__init__()
        self.scale = scale
        self.block_list = []

        self.conv1 = ConvBNLayer(
            in_channels=3,
            out_channels=int(32 * scale),
            kernel_size=3,
            stride=2,
            padding=1,
            act=Hardswish,
        )

        conv2_1 = DepthWiseSeparable(
            num_features=int(32 * scale),
            num_filters1=32,
            num_filters2=64,
            num_groups=32,
            stride=1,
            scale=scale,
        )
        self.block_list.append(conv2_1)

        conv2_2 = DepthWiseSeparable(
            num_features=int(64 * scale),
            num_filters1=64,
            num_filters2=128,
            num_groups=64,
            stride=1,
            scale=scale,
        )
        self.block_list.append(conv2_2)

        conv3_1 = DepthWiseSeparable(
            num_features=int(128 * scale),
            num_filters1=128,
            num_filters2=128,
            num_groups=128,
            stride=1,
            scale=scale,
        )
        self.block_list.append(conv3_1)

        conv3_2 = DepthWiseSeparable(
            num_features=int(128 * scale),
            num_filters1=128,
            num_filters2=256,
            num_groups=128,
            stride=(2, 1),
            scale=scale,
        )
        self.block_list.append(conv3_2)

        conv4_1 = DepthWiseSeparable(
            num_features=int(256 * scale),
            num_filters1=256,
            num_filters2=256,
            num_groups=256,
            stride=1,
            scale=scale,
        )
        self.block_list.append(conv4_1)

        conv4_2 = DepthWiseSeparable(
            num_features=int(256 * scale),
            num_filters1=256,
            num_filters2=512,
            num_groups=256,
            stride=(2, 1),
            scale=scale,
        )
        self.block_list.append(conv4_2)

        for _ in range(5):
            conv5 = DepthWiseSeparable(
                num_features=int(512 * scale),
                num_filters1=512,
                num_filters2=512,
                num_groups=512,
                stride=1,
                dw_size=5,
                padding=2,
                scale=scale,
                use_se=False,
            )
            self.block_list.append(conv5)

        conv5_6 = DepthWiseSeparable(
            num_features=int(512 * scale),
            num_filters1=512,
            num_filters2=1024,
            num_groups=512,
            stride=(2, 1),
            dw_size=5,
            padding=2,
            scale=scale,
            use_se=True,
        )
        self.block_list.append(conv5_6)

        conv6 = DepthWiseSeparable(
            num_features=int(1024 * scale),
            num_filters1=1024,
            num_filters2=1024,
            num_groups=1024,
            stride=last_conv_stride,
            dw_size=5,
            padding=2,
            use_se=True,
            scale=scale,
        )
        self.block_list.append(conv6)

        self.block_list = nn.Sequential(*self.block_list)
        if last_pool_type == "avg":
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        else:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.out_channels = int(1024 * scale)

    def forward(self, inputs):
        y = self.conv1(inputs)
        y = self.block_list(y)
        y = self.pool(y)
        return y


class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(
            in_channels=channel,
            out_channels=channel // reduction,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=channel // reduction,
            out_channels=channel,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.hardsigmoid = Hardsigmoid()

    def forward(self, x):
        identity = x
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.hardsigmoid(x)
        x = torch.mul(identity, x)
        return x


if __name__ == "__main__":
    from torchsummary import summary

    arr = torch.rand((1, 3, 32, 224))
    model = MobileNetV1Enhance()
    summary(model, input_size=(3, 32, 224), batch_size=1)
    out = model(arr)
    print(out.size())
    # print(model)
