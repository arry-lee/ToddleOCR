import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ResNet_E2E"]

from toddleocr.modules.backbones.resnet.det_resnet_vd_sast import ResNet_SAST
from toddleocr.ops import ConvBNLayer


class ResNet_E2E(ResNet_SAST):
    def convs(self):
        return ConvBNLayer(
            in_channels=self.in_channels,
            out_channels=64,
            kernel_size=7,
            stride=2,
            act="relu",
            name="conv1_1",
        )
