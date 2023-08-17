import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d

__all__ = ["ResNet_vd", "DeformableConvV2"]

from ptocr.ops import ConvBNLayer


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

        self.mask_channel = kernel_size ** 2 * groups  # 3x3x1=18 每个像素都有一个mask像素
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
        (offset, mask) = torch.split(offset_mask, [self.offset_channel, self.mask_channel], dim=1)  # 可微
        mask = F.sigmoid(mask)
        return self.conv_dcn(x, offset, mask=mask)




class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, shortcut=True, if_first=False, is_dcn=False):
        super().__init__()
        self.conv0 = ConvBNLayer(in_channels=in_channels, out_channels=out_channels, kernel_size=1, act="relu")
        self.conv1 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            act="relu",
            is_dcn=is_dcn,
            dcn_groups=2,
        )
        self.conv2 = ConvBNLayer(in_channels=out_channels, out_channels=out_channels * 4, kernel_size=1, act=None)
        if not shortcut:
            self.short = ConvBNLayer(
                in_channels=in_channels,
                out_channels=out_channels * 4,
                kernel_size=1,
                stride=1,
                is_vd_mode=not if_first,
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
    def __init__(self, in_channels, out_channels, stride, shortcut=True, is_first=False, **kwargs):
        super().__init__()
        self.stride = stride  # maybe unused
        self.conv0 = ConvBNLayer(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, act="relu"
        )
        self.conv1 = ConvBNLayer(in_channels=out_channels, out_channels=out_channels, kernel_size=3, act=None)
        if not shortcut:
            self.short = ConvBNLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                is_vd_mode=not is_first,  # 不是第一个降采样
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


from torchvision.models.resnet import ResNet, Bottleneck


class ResNet_vd(nn.Module):
    def __init__(self, in_channels=3, layers=50, dcn_stage=None, out_indices=None, **kwargs):
        super().__init__()
        # self.layers = layers
        supported_layers = {18: [2, 2, 2, 2],
                            34: [3, 4, 6, 3],
                            50: [3, 4, 6, 3],
                            101: [3, 4, 23, 3],
                            152: [3, 8, 36, 3],
                            200: [3, 12, 48, 3]}

        assert layers in supported_layers, f"supported layers are {list(supported_layers.keys())} but input layer is {layers}"
        depth = supported_layers[layers]

        num_features = [64, 256, 512, 1024] if layers >= 50 else [64, 64, 128, 256]
        num_filters = [64, 128, 256, 512]

        self.dcn_stage = dcn_stage if dcn_stage is not None else [False, False, False, False]
        self.out_indices = out_indices if out_indices is not None else [0, 1, 2, 3]
        self.conv1_1 = ConvBNLayer(in_channels=in_channels, out_channels=32, kernel_size=3, stride=2, act="relu")
        self.conv1_2 = ConvBNLayer(in_channels=32, out_channels=32, kernel_size=3, stride=1, act="relu")
        self.conv1_3 = ConvBNLayer(in_channels=32, out_channels=64, kernel_size=3, stride=1, act="relu")
        self.pool2d_max = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stages = []
        self.out_channels = []

        expansion = 4 if layers >= 50 else 1
        block_class = BottleneckBlock if layers >= 50 else BasicBlock

        # if layers >= 50:
        for block in range(len(depth)):
            block_list = []
            shortcut = False
            is_dcn = self.dcn_stage[block]
            for i in range(depth[block]):
                bottleneck_block = block_class(
                    in_channels=num_features[block] if i == 0 else num_filters[block] * expansion,
                    out_channels=num_filters[block],
                    stride=2 if i == 0 and block != 0 else 1,
                    shortcut=shortcut,
                    if_first=block == i == 0,
                    is_dcn=is_dcn,
                )
                # block_list["bb_%d_%d" % (block, i)] = bottleneck_block
                self.add_module("bb_%d_%d" % (block, i), bottleneck_block)
                shortcut = True
                block_list.append(bottleneck_block)
            if block in self.out_indices:
                self.out_channels.append(num_filters[block] * expansion)
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


from torchvision.models.resnet import ResNet, Bottleneck

if __name__ == "__main__":
    # c = ConvBNLayer(3, 32, 3, stride=2, groups=1)
    # print(c)
    # x = torch.randn(1, 3, 224, 224)
    # y = c(x)
    # print(y.shape)
    res = ResNet_vd()
    print(res)
    # for name, model in res.named_children():
    #     if name == "bb_0_0":
    #         print(model)
    #         print(res.stages[0][0])
    #         assert model is res.stages[0][0]
