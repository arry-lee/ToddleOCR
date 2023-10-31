import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["MobileNetV3"]

from toddleocr.ops import ConvBNLayer


def make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class MobileNetV3(nn.Module):
    """
    挤压和激励以及swish非线性都使用了sigmoid，它的计算效率很低，
    而且很难在定点算法中保持精度，因此我们将其替换为硬sigmoid

    之前的工作

    MobileNetV1： 深度可分离卷积+1×1卷积
    MobileNetV2： 线性瓶颈＋反向残差
    MnasNet： 建立在 MobileNetV2 结构上，将基于SE的注意力模块引入到bottleneck结构中
    本文的方法

    （1）上述三种结构相结合构建块来实现更高效的模型

    （2）引入了改进的swish作为非线性函数

    （3）将Sigmoid替换为hard Sigmoid

    Swish的简单性及其与ReLU的相似性使从业者可以轻松地在任何神经网络中用Swish单元替换ReLU。
    swish(x) = x*sigmoid(x)
    hardswish(x) = 0 if x<-3 else (x if x>3 else x*(x+3)/6)
    """

    def __init__(self, in_channels=3, model_name="large", scale=0.5, disable_se=False):
        super().__init__()
        self.disable_se = disable_se
        if model_name == "large":
            cfg = [
                [3, 16, 16, False, "relu", 1],
                [3, 64, 24, False, "relu", 2],
                [3, 72, 24, False, "relu", 1],
                [5, 72, 40, True, "relu", 2],  # stage0
                [5, 120, 40, True, "relu", 1],
                [5, 120, 40, True, "relu", 1],
                [3, 240, 80, False, "hardswish", 2],  # stage1
                [3, 200, 80, False, "hardswish", 1],
                [3, 184, 80, False, "hardswish", 1],
                [3, 184, 80, False, "hardswish", 1],
                [3, 480, 112, True, "hardswish", 1],
                [3, 672, 112, True, "hardswish", 1],
                [5, 672, 160, True, "hardswish", 2],  # stage2
                [5, 960, 160, True, "hardswish", 1],
                [5, 960, 160, True, "hardswish", 1],  # stage3
            ]
            cls_ch_squeeze = 960
        elif model_name == "small":
            cfg = [
                [3, 16, 16, True, "relu", 2],
                [3, 72, 24, False, "relu", 2],  # stage0
                [3, 88, 24, False, "relu", 1],
                [5, 96, 40, True, "hardswish", 2],  # stage1
                [5, 240, 40, True, "hardswish", 1],
                [5, 240, 40, True, "hardswish", 1],
                [5, 120, 48, True, "hardswish", 1],
                [5, 144, 48, True, "hardswish", 1],
                [5, 288, 96, True, "hardswish", 2],  # stage2
                [5, 576, 96, True, "hardswish", 1],
                [5, 576, 96, True, "hardswish", 1],  # stage3
            ]
            cls_ch_squeeze = 576
        else:
            raise NotImplementedError(f"mode[{model_name}_model] is not implemented!")
        supported_scale = [0.35, 0.5, 0.75, 1.0, 1.25]
        assert (
            scale in supported_scale
        ), "supported scales are {} but input scale is {}".format(
            supported_scale, scale
        )
        inplanes = 16
        self.conv = ConvBNLayer(
            in_channels=in_channels,
            out_channels=make_divisible(inplanes * scale),  # 输出通道数的缩放，取8倍数
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
        start_idx = 2 if model_name == "large" else 0
        inplanes = make_divisible(inplanes * scale)
        for k, exp, c, se, nl, s in cfg:
            se = se and (not self.disable_se)
            if s == 2 and i > start_idx:  # stride==2时输出
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
        out_channels = make_divisible(scale * cls_ch_squeeze)
        block_list.append(
            ConvBNLayer(
                in_channels=inplanes,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=1,
                act="hardswish",
            )
        )
        self.stages.append(nn.Sequential(*block_list))
        self.out_channels.append(out_channels)
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
    """倒残差块
    InvertedResidual模块由三个主要部分组成：扩张卷积（Expansion Convolution）、
    深度可分离卷积（Depthwise Separable Convolution）和线性投影（Linear Projection）。

    扩张卷积（Expansion Convolution）：该步骤通过使用1x1卷积来扩展输入特征图的通道数。
    这样做的目的是为了增加网络的表达能力，使得后续的卷积操作可以更好地捕捉特征。
        具体来说，对于每个输出通道，1x1卷积会使用一个1x1的卷积核，该卷积核的深度与输入通道数相同（即3）。
        对于输入的每个像素位置，1x1卷积会将该位置上的3个通道分别与卷积核的对应通道进行逐元素相乘，
        并将结果相加，得到该位置上的输出通道值。

        举个例子，假设输入图像的尺寸为HxW（高度x宽度），输入通道数为3，输出通道数为8。那么对于输出图像的某个像素位置(i, j)，
        1x1卷积会对输入图像在该位置上的3个通道的值与卷积核的3个通道的值进行逐元素相乘并相加，得到该位置上的8个输出通道的值。
        这样，输出图像的尺寸仍然是HxW，但通道数变为8。

    深度可分离卷积（Depthwise Separable Convolution）：在扩张卷积之后，使用深度可分离卷积来减少计算复杂度。
    深度可分离卷积将标准卷积分解为深度卷积和逐点卷积两步。首先，深度卷积在每个输入通道上进行卷积操作，
    然后逐点卷积在通道间进行卷积操作。这种操作方式大大减少了参数数量和计算量，同时保持了较好的特征表示能力。
    具体来说，深度可分离卷积的操作过程如下：

        深度卷积（Depthwise Convolution）：对输入的每个通道分别进行卷积操作，使用大小为kxk的卷积核，
        得到与输入通道数相同的输出通道数。也就是说，每个输入通道都会生成一个输出通道。

        逐点卷积（Pointwise Convolution）：对深度卷积的结果进行逐点卷积，即使用大小为1x1的卷积核，
        将每个输入通道与对应的权重进行线性组合得到最终的输出通道。这里的1x1卷积核相当于对每个通道进行全连接操作。

        因此，深度可分离卷积的总参数量为C×k×k + C×M，其中C×k×k表示深度卷积的参数量，C×M表示逐点卷积的参数量。

    线性投影（Linear Projection）：最后，为了保持特征图的维度一致性，InvertedResidual模块使用1x1卷积进行线性投影。
    这样可以将特征图的通道数调整为与输入相同，以便与网络的其他部分连接和融合。

    InvertedResidual模块的作用是在保持较高精度的同时，显著减少了计算复杂度。
    通过扩张卷积和深度可分离卷积的结合，InvertedResidual模块能够在轻量级网络中提供更强大的特征表示能力。
    这使得MobileNetV3在资源受限的设备上具有更好的实时性能和计算效率。

    扩张卷积（Expansion Convolution）：使用1x1卷积操作，将输入通道数扩展到一个较大的值，以增加网络的表达能力。
    深度可分离卷积（Depthwise Convolution）：对扩张卷积的结果进行深度可分离卷积操作，即先对每个输入通道进行卷积，再对输出通道进行逐点卷积。
    线性投影（Linear Projection）：在深度可分离卷积的输出上再次使用1x1卷积操作，将输出通道数降低到一个较小的值。
    线性投影的作用是为了减少最终的输出通道数，从而控制模型的复杂度，并提取更加紧凑的特征表示。这样可以在保持一定性能的同时，减少计算量和参数量。

    """

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
        # 没有下采样，输入通道数等于输出通道数的时候使用短路连接
        self.use_shortcut = stride == 1 and in_channels == out_channels
        self.use_se = use_se
        # 扩张卷积
        self.expand_conv = ConvBNLayer(
            in_channels=in_channels,
            out_channels=mid_channels,  # 卷积层数，相当于对全图进行一个数值缩放，线性组合，参数量为in*out
            kernel_size=1,  # 1x1
            stride=1,
            padding=0,
            act=act,
        )
        # 深度可分卷积
        self.bottleneck_conv = ConvBNLayer(
            in_channels=mid_channels,
            out_channels=mid_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=int((kernel_size - 1) // 2),
            groups=mid_channels,  # 分组卷积，且分了所有层，在每一层上单独卷积然后叠加
            act=act,
        )
        if self.use_se:
            self.mid_se = SqueezeExcitation(mid_channels)
        # 线性投影
        self.linear_conv = ConvBNLayer(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            act=None,  # 线性层与扩张卷积区别在于不用激活
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
    """
    SE模块的主要思想是，对于每个通道，先使用全局平均池化操作将其压缩成一个标量，
        然后通过两个全连接层，分别学习通道间的关系和权重，最终将通道进行重新加权。具体来说，SE模块包括以下几个步骤：

    Squeeze：使用全局平均池化操作将每个通道的特征图压缩成一个标量，得到一个通道数为1的特征图。
    Excitation：将上一步得到的通道数为1的特征图通过两个全连接层，学习通道间的关系和权重，得到一个通道数为C的向量。
    Scale：将上一步得到的向量通过sigmoid函数进行归一化，得到一个通道数为C的向量，表示每个通道的重要性。
    Recalibration：将上一步得到的向量与输入特征图相乘，重新加权每个通道的特征图。

    SE模块的作用是，通过学习通道间的关系和权重，自适应地调整每个通道的重要性，
    使得网络更加关注重要的通道，抑制不重要的通道。这样可以提高网络的表达能力，增强对特征的区分能力，从而提高模型的性能。
    Squeeze-and-Excitation (SE)是MobileNetV3中的一种模块，用于增强网络的表达能力。SE模块可以在不增加计算量的情况下提高网络的性能

    为什么说它没有增加计算量？

    SE模块的计算量主要与通道数相关，而通道数相对于输入特征图的空间维度来说通常是比较小的。相比于卷积操作，SE模块的计算量可以被忽略不计。
    SE模块的确没有增加计算量，主要是因为其操作都是在特征图的通道维度上进行的，而不是空间维度上。具体来说，SE模块包括以下几个步骤：

    Squeeze：使用全局平均池化操作将每个通道的特征图压缩成一个标量，得到一个通道数为1的特征图。
    这一步的计算量非常小，只需要对每个通道的特征图进行一次平均池化操作。

    Excitation：将上一步得到的通道数为1的特征图通过两个全连接层，学习通道间的关系和权重，
    得到一个通道数为C的向量。这一步的计算量与输入特征图的大小无关，只与通道数有关，因此计算量非常小。

    Scale：将上一步得到的向量通过sigmoid函数进行归一化，得到一个通道数为C的向量，
    表示每个通道的重要性。这一步的计算量也非常小，只需要对每个通道的向量进行一次sigmoid操作。

    Recalibration：将上一步得到的向量与输入特征图相乘，重新加权每个通道的特征图。
    这一步的计算量与输入特征图的大小和通道数有关，但是由于输入特征图已经被压缩成了通道数为1的特征图，因此计算量也非常小。

    """

    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 平均到只有一个值
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels // reduction,
            kernel_size=1,
            stride=1,
            padding=0,
        )  # 使用卷积来代替全连接
        self.conv2 = nn.Conv2d(
            in_channels=in_channels // reduction,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    @staticmethod
    def hardsigmoid(x, slope=0.2, offset=0.5):
        return torch.clamp(x * slope + offset, 0, 1)

    def forward(self, inputs):
        outputs = self.avg_pool(inputs)
        outputs = self.conv1(outputs)
        outputs = F.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.hardsigmoid(outputs)  # Cx1x1
        return inputs * outputs
