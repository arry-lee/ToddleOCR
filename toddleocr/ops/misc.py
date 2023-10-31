import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import DeformConv2d


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
        elif isinstance(act, type) and issubclass(act, nn.Module):
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


class DeformableConvV2(nn.Module):
    """`Deformable ConvNets v2: More Deformable, Better Results
    <https://arxiv.org/abs/1811.11168>"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        groups=1,
        bias=None,
        skip_quant=False,  # 跳过量化
    ):
        super().__init__()

        self.mask_channel = kernel_size**2 * groups  # 3x3x1=18 每个像素都有一个mask像素
        self.offset_channel = 2 * self.mask_channel  # 2x3x3x1=18 每个像素都有x方向和y方向两个偏移量

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
            3 * self.mask_channel,  # mask_channel + offset_channel
            kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            bias=bias,
        )
        if skip_quant:
            self.conv_offset.skip_quant = True  # note

    def forward(self, x):
        offset_mask = self.conv_offset(x)
        (offset, mask) = torch.split(
            offset_mask, [self.offset_channel, self.mask_channel], dim=1
        )  # 可微
        mask = F.sigmoid(mask)
        return self.conv_dcn(x, offset, mask=mask)
