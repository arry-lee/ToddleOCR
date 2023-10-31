import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["CSPPAN"]

from toddleocr.ops import ConvBNLayer


class DPModule(nn.Module):
    """
    Depth-wise and point-wise module.
     Args:
        in_channels (int): The input channels of this Module.
        out_channels (int): The output channels of this Module.
        kernel_size (int): The conv2d kernel size of this Module.
        stride (int): The conv2d's stride of this Module.
        act (str): The activation function of this Module,
                   Now support `leaky_relu` and `hard_swish`.
    """

    def __init__(
        self, in_channels=96, out_channels=96, kernel_size=3, stride=1, act="leaky_relu"
    ):
        super().__init__()
        # initializer = nn.init.KaimingUniform()
        self.act = act
        self.dwconv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=out_channels,
            padding=(kernel_size - 1) // 2,
            stride=stride,
            bias=False,
        )
        nn.init.kaiming_uniform_(self.dwconv.weight)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.pwconv = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
            groups=1,
            padding=0,
            bias=False,
        )
        nn.init.kaiming_uniform_(self.pwconv.weight)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def act_func(self, x):
        if self.act == "leaky_relu":
            x = F.leaky_relu(x)
        elif self.act == "hardswish":
            x = F.hardswish(x)
        return x

    def forward(self, x):
        x = self.act_func(self.bn1(self.dwconv(x)))
        x = self.act_func(self.bn2(self.pwconv(x)))
        return x


class DarknetBottleneck(nn.Module):
    """The basic bottleneck block used in Darknet.
    Each Block consists of two ConvModules and the input is added to the
    final output. Each ConvModule is composed of Conv, BN, and act.
    The first convLayer has filter size of 1x1 and the second one has the
    filter size of 3x3.
    Args:
        in_channels (int): The input channels of this Module.
        out_channels (int): The output channels of this Module.
        expansion (int): The kernel size of the convolution. Default: 0.5
        add_identity (bool): Whether to add identity to the out.
            Default: True
        use_depthwise (bool): Whether to use depthwise separable convolution.
            Default: False
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        expansion=0.5,
        add_identity=True,
        use_depthwise=False,
        act="leaky_relu",
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        conv_func = DPModule if use_depthwise else ConvBNLayer
        self.conv1 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=1,
            act=act,
        )
        self.conv2 = conv_func(
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            act=act,
        )
        self.add_identity = add_identity and in_channels == out_channels

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)

        if self.add_identity:
            return out + identity
        else:
            return out


class CSPLayer(nn.Module):
    """Cross Stage Partial Layer.
    Args:
        in_channels (int): The input channels of the CSP layer.
        out_channels (int): The output channels of the CSP layer.
        expand_ratio (float): Ratio to adjust the number of channels of the
            hidden layer. Default: 0.5
        num_blocks (int): Number of blocks. Default: 1
        add_identity (bool): Whether to add identity in blocks.
            Default: True
        use_depthwise (bool): Whether to depthwise separable convolution in
            blocks. Default: False
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        expand_ratio=0.5,
        num_blocks=1,
        add_identity=True,
        use_depthwise=False,
        act="leaky_relu",
    ):
        super().__init__()
        mid_channels = int(out_channels * expand_ratio)
        self.main_conv = ConvBNLayer(in_channels, mid_channels, 1, act=act)
        self.short_conv = ConvBNLayer(in_channels, mid_channels, 1, act=act)
        self.final_conv = ConvBNLayer(2 * mid_channels, out_channels, 1, act=act)

        self.blocks = nn.Sequential(
            *[
                DarknetBottleneck(
                    mid_channels,
                    mid_channels,
                    kernel_size,
                    1.0,
                    add_identity,
                    use_depthwise,
                    act=act,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x):
        x_short = self.short_conv(x)

        x_main = self.main_conv(x)
        x_main = self.blocks(x_main)

        x_final = torch.concat((x_main, x_short), dim=1)
        return self.final_conv(x_final)


class Channel_T(nn.Module):
    def __init__(self, in_channels=(116, 232, 464), out_channels=96, act="leaky_relu"):
        super().__init__()
        self.convs = nn.ModuleList()
        for i in range(len(in_channels)):
            self.convs.append(ConvBNLayer(in_channels[i], out_channels, 1, act=act))

    def forward(self, x):
        outs = [self.convs[i](x[i]) for i in range(len(x))]
        return outs


class CSPPAN(nn.Module):
    """Path Aggregation Network with CSP module.
    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        kernel_size (int): The conv2d kernel size of this Module.
        num_csp_blocks (int): Number of bottlenecks in CSPLayer. Default: 1
        use_depthwise (bool): Whether to depthwise separable convolution in
            blocks. Default: True
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=5,
        num_csp_blocks=1,
        use_depthwise=True,
        act="hardswish",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = [out_channels] * len(in_channels)
        conv_func = DPModule if use_depthwise else ConvBNLayer

        self.conv_t = Channel_T(in_channels, out_channels, act=act)

        # build top-down blocks
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.top_down_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1, 0, -1):
            self.top_down_blocks.append(
                CSPLayer(
                    out_channels * 2,
                    out_channels,
                    kernel_size=kernel_size,
                    num_blocks=num_csp_blocks,
                    add_identity=False,
                    use_depthwise=use_depthwise,
                    act=act,
                )
            )

        # build bottom-up blocks
        self.downsamples = nn.ModuleList()
        self.bottom_up_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1):
            self.downsamples.append(
                conv_func(
                    out_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=2,
                    act=act,
                )
            )
            self.bottom_up_blocks.append(
                CSPLayer(
                    out_channels * 2,
                    out_channels,
                    kernel_size=kernel_size,
                    num_blocks=num_csp_blocks,
                    add_identity=False,
                    use_depthwise=use_depthwise,
                    act=act,
                )
            )

    def forward(self, inputs):
        """
        Args:
            inputs (tuple[Tensor]): input features.
        Returns:
            tuple[Tensor]: CSPPAN features.
        """
        assert len(inputs) == len(self.in_channels)
        inputs = self.conv_t(inputs)

        # top-down path
        inner_outs = [inputs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = inputs[idx - 1]
            upsample_feat = F.upsample(
                feat_heigh, size=feat_low.shape[2:4], mode="nearest"
            )

            inner_out = self.top_down_blocks[len(self.in_channels) - 1 - idx](
                torch.concat([upsample_feat, feat_low], 1)
            )
            inner_outs.insert(0, inner_out)

        # bottom-up path
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsamples[idx](feat_low)
            out = self.bottom_up_blocks[idx](
                torch.concat([downsample_feat, feat_height], 1)
            )
            outs.append(out)

        return tuple(outs)
