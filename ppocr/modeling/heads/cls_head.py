import math
import torch
from torch import nn
import torch.nn.functional as F


class ClsHead(nn.Module):
    """
    Class orientation

    Args:

        params(dict): super parameters for build Class network
    """

    def __init__(self, in_channels, class_dim, **kwargs):
        super(ClsHead, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        stdv = 1.0 / math.sqrt(in_channels * 1.0)
        self.fc = nn.Linear(
            in_channels,
            class_dim,
            bias=True,
        )

    def forward(self, x, targets=None):
        x = self.pool(x)
        x = torch.reshape(x, shape=[x.shape[0], x.shape[1]])
        x = self.fc(x)
        if not self.training:
            x = F.softmax(x, dim=1)
        return x
