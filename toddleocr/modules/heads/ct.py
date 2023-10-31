import math

import torch.nn.functional as F
from torch import nn
from torch.nn.init import normal_, ones_, zeros_


class CTHead(nn.Module):
    def __init__(
        self, in_channels, hidden_dim, num_classes, loss_kernel=None, loss_loc=None
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, hidden_dim, kernel_size=3, stride=1, padding=1
        )
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(
            hidden_dim, num_classes, kernel_size=1, stride=1, padding=0
        )

        for m in self.children():
            if isinstance(m, nn.Conv2d):
                n = m._kernel_size[0] * m._kernel_size[1] * m._out_channels
                normal_(m.weight, mean=0.0, std=math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                zeros_(m.bias)
                ones_(m.weight)

    def _upsample(self, x, scale=1):
        return F.upsample(x, scale_factor=scale, mode="bilinear")

    def forward(self, f, targets=None):
        out = self.conv1(f)
        out = self.relu1(self.bn1(out))
        out = self.conv2(out)

        if self.training:
            out = self._upsample(out, scale=4)
            return {"maps": out}
        else:
            score = F.sigmoid(out[:, 0, :, :])
            return {"maps": out, "score": score}
