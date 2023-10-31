import torch.nn as nn

__all__ = ["ResNet_SAST"]
from toddleocr.modules.backbones.resnet.det_resnet import ResNet
from toddleocr.ops import ConvBNLayer


class ResNet_SAST(ResNet):
    num_features = [64, 256, 512, 1024, 2048]
    num_filters = [64, 128, 256, 512, 512]
    out_channels = [3, 64]

    def convs(self):
        return nn.Sequential(
            ConvBNLayer(
                in_channels=self.in_channels,
                out_channels=32,
                kernel_size=3,
                stride=2,
                act="relu",
                name="conv1_1",
            ),
            ConvBNLayer(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=1,
                act="relu",
                name="conv1_2",
            ),
            ConvBNLayer(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                act="relu",
                name="conv1_3",
            ),
        )

    def format_name(self, block, i):
        if self.layers in [101, 152] and block == 2:
            if i == 0:
                conv_name = "res" + str(block + 2) + "a"
            else:
                conv_name = "res" + str(block + 2) + "b" + str(i)
        else:
            conv_name = "res" + str(block + 2) + chr(97 + i)
        return conv_name

    def forward(self, x):
        out = [x]
        y = self.conv(x)
        out.append(y)
        y = self.pool2d_max(y)
        for block in self.stages:
            y = block(y)
            out.append(y)
        return out
