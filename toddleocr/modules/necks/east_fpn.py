import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["EASTFPN"]

from toddleocr.ops import ConvBNLayer


class DeConvBNLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        groups=1,
        act=None,
        name=None,
    ):
        super().__init__()
        self.act = act
        self.deconv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        # self.bn.register_buffer("bn_" + name + "_mean", self.bn.running_mean)
        # self.bn.register_buffer("bn_" + name + "_variance", self.bn.running_var)
        if act is not None:
            self.act = getattr(F, act)
        else:
            self.act = None

    def forward(self, x):
        x = self.deconv(x)
        x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


class EASTFPN(nn.Module):
    def __init__(self, in_channels, model_name, **kwargs):
        super().__init__()
        self.model_name = model_name
        if self.model_name == "large":
            self.out_channels = 128
        else:
            self.out_channels = 64
        self.in_channels = in_channels[::-1]
        self.h1_conv = ConvBNLayer(
            in_channels=self.out_channels + self.in_channels[1],
            out_channels=self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            act="relu",
            name="unet_h_1",
        )
        self.h2_conv = ConvBNLayer(
            in_channels=self.out_channels + self.in_channels[2],
            out_channels=self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            act="relu",
            name="unet_h_2",
        )
        self.h3_conv = ConvBNLayer(
            in_channels=self.out_channels + self.in_channels[3],
            out_channels=self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            act="relu",
            name="unet_h_3",
        )
        self.g0_deconv = DeConvBNLayer(
            in_channels=self.in_channels[0],
            out_channels=self.out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
            act="relu",
            name="unet_g_0",
        )
        self.g1_deconv = DeConvBNLayer(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
            act="relu",
            name="unet_g_1",
        )
        self.g2_deconv = DeConvBNLayer(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
            act="relu",
            name="unet_g_2",
        )
        self.g3_conv = ConvBNLayer(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            act="relu",
            name="unet_g_3",
        )

    def forward(self, x):
        f = x[::-1]  # 输出层逆序，是个列表
        h = f[0]  # 最后一层的特征
        g = self.g0_deconv(h)  # 上采样
        h = torch.concat([g, f[1]], dim=1)  # 和前一层堆叠
        h = self.h1_conv(h)  # 继续上采样
        g = self.g1_deconv(h)  # 继续上采样输出均为 outchannels
        h = torch.concat([g, f[2]], dim=1)
        h = self.h2_conv(h)
        g = self.g2_deconv(h)
        h = torch.concat([g, f[3]], dim=1)
        h = self.h3_conv(h)
        g = self.g3_conv(h)
        return g
