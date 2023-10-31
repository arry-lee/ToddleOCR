from torch import nn
from torch.nn import functional as F

__all__ = ["PRENHead"]


class PRENHead(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x, targets=None):
        predicts = self.linear(x)

        if not self.training:
            predicts = F.softmax(predicts, dim=2)

        return predicts
