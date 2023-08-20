import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["MobileNetV3"]
from ptocr.ops import ConvBNLayer


def make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v = new_v+divisor
    return new_v


class MobileNetV3(nn.Module):
    def __init__(
        self,
        in_channels=3,
        model_name="large",
        scale=0.5,
        disable_se=False,
        **kwargs,
    ):
        super().__init__()
        self.disable_se = disable_se
        if model_name == "large":
            cfg = [
                [3, 16, 16, False, "relu", 1],
                [3, 64, 24, False, "relu", 2],
                [3, 72, 24, False, "relu", 1],
                [5, 72, 40, True, "relu", 2],
                [5, 120, 40, True, "relu", 1],
                [5, 120, 40, True, "relu", 1],
                [3, 240, 80, False, "hardswish", 2],
                [3, 200, 80, False, "hardswish", 1],
                [3, 184, 80, False, "hardswish", 1],
                [3, 184, 80, False, "hardswish", 1],
                [3, 480, 112, True, "hardswish", 1],
                [3, 672, 112, True, "hardswish", 1],
                [5, 672, 160, True, "hardswish", 2],
                [5, 960, 160, True, "hardswish", 1],
                [5, 960, 160, True, "hardswish", 1],
            ]
            cls_ch_squeeze = 960
        elif model_name == "small":
            cfg = [
                [3, 16, 16, True, "relu", 2],
                [3, 72, 24, False, "relu", 2],
                [3, 88, 24, False, "relu", 1],
                [5, 96, 40, True, "hardswish", 2],
                [5, 240, 40, True, "hardswish", 1],
                [5, 240, 40, True, "hardswish", 1],
                [5, 120, 48, True, "hardswish", 1],
                [5, 144, 48, True, "hardswish", 1],
                [5, 288, 96, True, "hardswish", 2],
                [5, 576, 96, True, "hardswish", 1],
                [5, 576, 96, True, "hardswish", 1],
            ]
            cls_ch_squeeze = 576
        else:
            raise NotImplementedError(
                f"mode[{model_name}_model] is not implemented!"
            )
        supported_scale = [0.35, 0.5, 0.75, 1.0, 1.25]
        assert (
            scale in supported_scale
        ), "supported scales are {} but input scale is {}".format(
            supported_scale, scale
        )
        inplanes = 16
        self.conv = ConvBNLayer(
            in_channels=in_channels,
            out_channels=make_divisible(inplanes * scale),
            kernel_size=3,
            stride=2,
            padding=1,
            groups=1,
            act="hardswish",
        )
        self.stages = []
        self.out_channels = []
        block_list = []
        i = 0
        inplanes = make_divisible(inplanes * scale)
        for k, exp, c, se, nl, s in cfg:
            se = se and (not self.disable_se)
            start_idx = 2 if model_name == "large" else 0
            if s == 2 and i > start_idx:
                self.out_channels.append(inplanes)
                self.stages.append(nn.Sequential(*block_list))
                block_list = []
            block_list.append(
                InvertedResidual(
                    in_channels=inplanes,
                    mid_channels=make_divisible(scale * exp),
                    out_channels=make_divisible(scale * c),
                    kernel_size=k,
                    stride=s,
                    use_se=se,
                    act=nl,
                )
            )
            inplanes = make_divisible(scale * c)
            i += 1
        block_list.append(
            ConvBNLayer(
                in_channels=inplanes,
                out_channels=make_divisible(scale * cls_ch_squeeze),
                kernel_size=1,
                stride=1,
                padding=0,
                groups=1,
                act="hardswish",
            )
        )
        self.stages.append(nn.Sequential(*block_list))
        self.out_channels.append(make_divisible(scale * cls_ch_squeeze))
        for i, stage in enumerate(self.stages):
            self.add_module("stage{}".format(i), stage)

    def forward(self, x):
        x = self.conv(x)
        out_list = []
        for stage in self.stages:
            x = stage(x)
            out_list.append(x)
        return out_list


class InvertedResidual(nn.Module):
    def __init__(
        self,
        in_channels,
        mid_channels,
        out_channels,
        kernel_size,
        stride,
        use_se,
        act=None,
    ):
        super().__init__()
        self.use_shortcut = stride == 1 and in_channels == out_channels
        self.use_se = use_se
        self.expand_conv = ConvBNLayer(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            act=act,
        )
        self.bottleneck_conv = ConvBNLayer(
            in_channels=mid_channels,
            out_channels=mid_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=int((kernel_size - 1) // 2),
            groups=mid_channels,
            act=act,
        )
        if self.use_se:
            self.mid_se = SqueezeExcitation(mid_channels)
        self.linear_conv = ConvBNLayer(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            act=None,
        )

    def forward(self, inputs):
        x = self.expand_conv(inputs)
        x = self.bottleneck_conv(x)
        if self.use_se:
            x = self.mid_se(x)
        x = self.linear_conv(x)
        if self.use_shortcut:
            x = torch.add(inputs, x)
        return x


class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels // reduction,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.conv2 = nn.Conv2d(
            in_channels=in_channels // reduction,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def hardsigmoid(self, x, slope=0.2, offset=0.5):
        return torch.clamp((x * slope + offset).clamp(min=0, max=1), 0, 1)

    def forward(self, inputs):
        outputs = self.avg_pool(inputs)
        outputs = self.conv1(outputs)
        outputs = F.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.hardsigmoid(outputs, slope=0.2, offset=0.5)
        return inputs * outputs
