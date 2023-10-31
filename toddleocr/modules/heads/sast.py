import torch.nn.functional as F
from torch import nn

__all__ = ["SASTHead"]

from toddleocr.ops import ConvBNLayer


class SASTHeader1(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super().__init__()
        out_channels = [64, 64, 128]
        self.score_conv = nn.Sequential(
            ConvBNLayer(
                in_channels, out_channels[0], 1, 1, act="relu", name="f_score1"
            ),
            ConvBNLayer(
                out_channels[0], out_channels[1], 3, 1, act="relu", name="f_score2"
            ),
            ConvBNLayer(
                out_channels[1], out_channels[2], 1, 1, act="relu", name="f_score3"
            ),
            ConvBNLayer(out_channels[2], 1, 3, 1, act=None, name="f_score4"),
        )
        self.border_conv = nn.Sequential(
            ConvBNLayer(
                in_channels, out_channels[0], 1, 1, act="relu", name="f_border1"
            ),
            ConvBNLayer(
                out_channels[0], out_channels[1], 3, 1, act="relu", name="f_border2"
            ),
            ConvBNLayer(
                out_channels[1], out_channels[2], 1, 1, act="relu", name="f_border3"
            ),
            ConvBNLayer(out_channels[2], 4, 3, 1, act=None, name="f_border4"),
        )

    def forward(self, x):
        f_score = self.score_conv(x)
        f_score = F.sigmoid(f_score)
        f_border = self.border_conv(x)
        return (f_score, f_border)


class SASTHeader2(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super().__init__()
        out_channels = [64, 64, 128]
        self.tvo_conv = nn.Sequential(
            ConvBNLayer(in_channels, out_channels[0], 1, 1, act="relu", name="f_tvo1"),
            ConvBNLayer(
                out_channels[0], out_channels[1], 3, 1, act="relu", name="f_tvo2"
            ),
            ConvBNLayer(
                out_channels[1], out_channels[2], 1, 1, act="relu", name="f_tvo3"
            ),
            ConvBNLayer(out_channels[2], 8, 3, 1, act=None, name="f_tvo4"),
        )
        self.tco_conv = nn.Sequential(
            ConvBNLayer(in_channels, out_channels[0], 1, 1, act="relu", name="f_tco1"),
            ConvBNLayer(
                out_channels[0], out_channels[1], 3, 1, act="relu", name="f_tco2"
            ),
            ConvBNLayer(
                out_channels[1], out_channels[2], 1, 1, act="relu", name="f_tco3"
            ),
            ConvBNLayer(out_channels[2], 2, 3, 1, act=None, name="f_tco4"),
        )

    def forward(self, x):
        f_tvo = self.tvo_conv(x)
        f_tco = self.tco_conv(x)
        return (f_tvo, f_tco)


class SASTHead(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super().__init__()
        self.head1 = SASTHeader1(in_channels)
        self.head2 = SASTHeader2(in_channels)

    def forward(self, x, targets=None):
        (f_score, f_border) = self.head1(x)
        (f_tvo, f_tco) = self.head2(x)
        predicts = {}
        predicts["f_score"] = f_score
        predicts["f_border"] = f_border
        predicts["f_tvo"] = f_tvo
        predicts["f_tco"] = f_tco
        return predicts
