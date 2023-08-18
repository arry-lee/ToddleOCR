import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ResNet_E2E"]

from ptocr.modules.backbones import ResNet_SAST
from ptocr.ops import ConvBNLayer


class ResNet_E2E(ResNet_SAST):

    def convs(self):
        return ConvBNLayer(
            in_channels=self.in_channels, out_channels=64, kernel_size=7, stride=2, act="relu", name="conv1_1"
        )
