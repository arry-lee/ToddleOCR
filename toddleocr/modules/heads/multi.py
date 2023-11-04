import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from toddleocr.modules.necks.rnn import (
    EncoderWithFC,
    EncoderWithRNN,
    EncoderWithSVTR,
    Im2Seq,
    SequenceEncoder,
)

from .ctc import CTCHead
from .sar import SARHead

__all__ = ["MultiHead"]


class MultiHead(nn.Module):
    def __init__(self, in_channels, out_channels_list, **kwargs):
        super().__init__()
        self.head_list = kwargs.pop("head_list")
        self.gtc_head = "sar"
        assert len(self.head_list) >= 2
        for idx, head_name in enumerate(self.head_list):
            name = head_name.pop("class")
            if name == "SARHead":
                sar_args = head_name
                self.sar_head = eval(name)(
                    in_channels=in_channels,
                    out_channels=out_channels_list["SARLabelDecode"],
                    **sar_args
                )
            elif name == "CTCHead":
                self.encoder_reshape = Im2Seq(in_channels)
                neck_args = head_name["Neck"]
                encoder_type = neck_args.pop("name")
                self.encoder = encoder_type
                self.ctc_encoder = SequenceEncoder(
                    in_channels=in_channels, encoder_type=encoder_type, **neck_args
                )
                head_args = head_name["Head"]
                self.ctc_head = eval(name)(
                    in_channels=self.ctc_encoder.out_channels,
                    out_channels=out_channels_list["CTCLabelDecode"],
                    **head_args
                )
            else:
                raise NotImplementedError(
                    "{} is not supported in MultiHead yet".format(name)
                )

    def forward(self, x, targets=None):
        ctc_encoder = self.ctc_encoder(x)
        ctc_out = self.ctc_head(ctc_encoder, targets)
        head_out = dict()
        head_out["ctc"] = ctc_out
        head_out["ctc_neck"] = ctc_encoder
        if not self.training:
            return ctc_out
        if self.gtc_head == "sar":
            sar_out = self.sar_head(x, targets[1:])
            head_out["sar"] = sar_out
            return head_out
        else:
            return head_out
