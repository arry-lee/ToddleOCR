import torch.nn.functional as F
from torch import nn
__all__ = ['EASTHead']


class ConvBNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1, act=None, name=None):
        super().__init__()
        # self.if_act = if_act
        self.act = act
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )

        self.bn = nn.BatchNorm2d(
            num_features=out_channels,
            # act=act,
            # bias=True,
            # moving_mean_name="bn_" + name + "_mean",
            # moving_variance_name="bn_" + name + "_variance",
        )
        self.bn.register_buffer("bn_" + name + "_mean", self.bn.running_mean)
        self.bn.register_buffer("bn_" + name + "_variance", self.bn.running_var)
        if act is not None:
            self.act = getattr(F, act)
        else:
            self.act = None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


class EASTHead(nn.Module):
    def __init__(self, in_channels, model_name, **kwargs):
        super().__init__()
        self.model_name = model_name
        if self.model_name == "large":
            num_outputs = [128, 64, 1, 8]
        else:
            num_outputs = [64, 32, 1, 8]

        self.det_conv1 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=num_outputs[0],
            kernel_size=3,
            stride=1,
            padding=1,
            act="relu",
            name="det_head1",
        )
        self.det_conv2 = ConvBNLayer(
            in_channels=num_outputs[0],
            out_channels=num_outputs[1],
            kernel_size=3,
            stride=1,
            padding=1,
            act="relu",
            name="det_head2",
        )
        self.score_conv = ConvBNLayer(
            in_channels=num_outputs[1],
            out_channels=num_outputs[2],
            kernel_size=1,
            stride=1,
            padding=0,
            act=None,
            name="f_score",
        )
        self.geo_conv = ConvBNLayer(
            in_channels=num_outputs[1],
            out_channels=num_outputs[3],
            kernel_size=1,
            stride=1,
            padding=0,
            act=None,
            name="f_geo",
        )

    def forward(self, x, targets=None):
        f_det = self.det_conv1(x)
        f_det = self.det_conv2(f_det)
        f_score = self.score_conv(f_det)
        f_score = F.sigmoid(f_score)
        f_geo = self.geo_conv(f_det)
        f_geo = (F.sigmoid(f_geo) - 0.5) * 2 * 800

        pred = {"f_score": f_score, "f_geo": f_geo}
        return pred
