"""
This code is refer from: 
https://github.com/hikopensource/DAVAR-Lab-OCR/blob/main/davarocr/davar_rcg/models/connects/single_block/RFAdaptor.py
"""
import torch.nn as nn
from torch.nn.init import kaiming_normal_, ones_, zeros_

__all__ = ["RFAdaptor"]


class S2VAdaptor(nn.Module):
    def __init__(self, in_channels=512):
        super().__init__()
        self.in_channels = in_channels
        self.channel_inter = nn.Linear(self.in_channels, self.in_channels, bias=False)
        self.channel_bn = nn.BatchNorm1d(self.in_channels)
        self.channel_act = nn.ReLU()
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            kaiming_normal_(m.weight)
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm2d, nn.BatchNorm1d)):
            zeros_(m.bias)
            ones_(m.weight)

    def forward(self, semantic):
        semantic_source = semantic
        semantic = semantic.squeeze(2).transpose([0, 2, 1])
        channel_att = self.channel_inter(semantic)
        channel_att = channel_att.transpose([0, 2, 1])
        channel_bn = self.channel_bn(channel_att)
        channel_att = self.channel_act(channel_bn)
        channel_output = semantic_source * channel_att.unsqueeze(-2)
        return channel_output


class V2SAdaptor(nn.Module):
    def __init__(self, in_channels=512, return_mask=False):
        super().__init__()
        self.in_channels = in_channels
        self.return_mask = return_mask
        self.channel_inter = nn.Linear(self.in_channels, self.in_channels, bias=False)
        self.channel_bn = nn.BatchNorm1d(self.in_channels)
        self.channel_act = nn.ReLU()

    def forward(self, visual):
        visual = visual.squeeze(2).transpose([0, 2, 1])
        channel_att = self.channel_inter(visual)
        channel_att = channel_att.transpose([0, 2, 1])
        channel_bn = self.channel_bn(channel_att)
        channel_att = self.channel_act(channel_bn)
        channel_output = channel_att.unsqueeze(-2)
        if self.return_mask:
            return (channel_output, channel_att)
        return channel_output


class RFAdaptor(nn.Module):
    def __init__(self, in_channels=512, use_v2s=True, use_s2v=True, **kwargs):
        super().__init__()
        if use_v2s is True:
            self.neck_v2s = V2SAdaptor(in_channels=in_channels, **kwargs)
        else:
            self.neck_v2s = None
        if use_s2v is True:
            self.neck_s2v = S2VAdaptor(in_channels=in_channels, **kwargs)
        else:
            self.neck_s2v = None
        self.out_channels = in_channels

    def forward(self, x):
        (visual_feature, rcg_feature) = x
        if visual_feature is not None:
            (
                batch,
                source_channels,
                v_source_height,
                v_source_width,
            ) = visual_feature.shape
            visual_feature = visual_feature.reshape(
                [batch, source_channels, 1, v_source_height * v_source_width]
            )
        if self.neck_v2s is not None:
            v_rcg_feature = rcg_feature * self.neck_v2s(visual_feature)
        else:
            v_rcg_feature = rcg_feature
        if self.neck_s2v is not None:
            v_visual_feature = visual_feature + self.neck_s2v(rcg_feature)
        else:
            v_visual_feature = visual_feature
        if v_rcg_feature is not None:
            (batch, source_channels, source_height, source_width) = v_rcg_feature.shape
            v_rcg_feature = v_rcg_feature.reshape(
                [batch, source_channels, 1, source_height * source_width]
            )
            v_rcg_feature = v_rcg_feature.squeeze(2).transpose([0, 2, 1])
        return (v_visual_feature, v_rcg_feature)
