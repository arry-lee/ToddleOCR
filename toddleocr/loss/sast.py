# SAST是一种文本检测和识别的技术，全称为Spatial Attention-based Scene Text Detection and Recognition。它是用于自然场景文本检测和识别的端到端模型，可以同时完成文本行的检测和字符识别任务。
#
# SAST模型基于空间注意力机制，能够有效地解决自然场景中的复杂背景、低分辨率、遮挡等问题，实现准确、鲁棒的文本检测和识别。该模型通过融合多尺度特征、设计多通道自适应卷积模块和使用超参数预测器等技术，达到了优秀的检测和识别性能。
#
# SAST可应用于多个场景，如自动驾驶、图像翻译、图像检索、文档分析等领域，可以提供高效、准确的文本检测和识别能力，对于自然场景中的文字信息提取具有重要的应用价值。


import numpy as np
import torch
from torch import nn


class SASTLoss(nn.Module):
    """ """

    def __init__(self, eps=1e-6, **kwargs):
        super().__init__()
        # self.dice_loss = DiceLoss(eps=eps)

    def forward(self, predicts, labels):
        """
        tcl_pos: N x 128 x 3
        tcl_mask: N x 128 x 1
        tcl_label: N x X list or LoDTensor
        """

        f_score = predicts["f_score"]
        f_border = predicts["f_border"]
        f_tvo = predicts["f_tvo"]
        f_tco = predicts["f_tco"]

        l_score, l_border, l_mask, l_tvo, l_tco = labels[1:]

        # score_loss
        intersection = torch.sum(f_score * l_score * l_mask)
        union = torch.sum(f_score * l_mask) + torch.sum(l_score * l_mask)
        score_loss = 1.0 - 2 * intersection / (union + 1e-5)

        # border loss
        l_border_split, l_border_norm = torch.split(l_border, [4, 1], dim=1)
        f_border_split = f_border
        border_ex_shape = l_border_norm.shape * np.array([1, 4, 1, 1])
        l_border_norm_split = torch.unsqueeze(l_border_norm, 0).repeat(border_ex_shape)
        l_border_score = torch.unsqueeze(l_score, 0).repeat(border_ex_shape)
        l_border_mask = torch.unsqueeze(l_mask, 0).repeat(border_ex_shape)

        border_diff = l_border_split - f_border_split
        abs_border_diff = torch.abs(border_diff)
        border_sign = abs_border_diff < 1.0
        border_sign = border_sign.type(dtype=torch.float32)
        border_sign.stop_gradient = True
        border_in_loss = 0.5 * abs_border_diff * abs_border_diff * border_sign + (
            abs_border_diff - 0.5
        ) * (1.0 - border_sign)
        border_out_loss = l_border_norm_split * border_in_loss
        border_loss = torch.sum(border_out_loss * l_border_score * l_border_mask) / (
            torch.sum(l_border_score * l_border_mask) + 1e-5
        )

        # tvo_loss
        l_tvo_split, l_tvo_norm = torch.split(l_tvo, [8, 1], dim=1)
        f_tvo_split = f_tvo
        tvo_ex_shape = l_tvo_norm.shape * np.array([1, 8, 1, 1])
        l_tvo_norm_split = torch.unsqueeze(l_tvo_norm, 0).repeat(tvo_ex_shape)
        l_tvo_score = torch.unsqueeze(l_score, 0).repeat(tvo_ex_shape)
        l_tvo_mask = torch.unsqueeze(l_mask, 0).repeat(tvo_ex_shape)
        #
        tvo_geo_diff = l_tvo_split - f_tvo_split
        abs_tvo_geo_diff = torch.abs(tvo_geo_diff)
        tvo_sign = abs_tvo_geo_diff < 1.0
        tvo_sign = tvo_sign.type(dtype=torch.float32)
        tvo_sign.stop_gradient = True
        tvo_in_loss = 0.5 * abs_tvo_geo_diff * abs_tvo_geo_diff * tvo_sign + (
            abs_tvo_geo_diff - 0.5
        ) * (1.0 - tvo_sign)
        tvo_out_loss = l_tvo_norm_split * tvo_in_loss
        tvo_loss = torch.sum(tvo_out_loss * l_tvo_score * l_tvo_mask) / (
            torch.sum(l_tvo_score * l_tvo_mask) + 1e-5
        )

        # tco_loss
        l_tco_split, l_tco_norm = torch.split(l_tco, [2, 1], dim=1)
        f_tco_split = f_tco
        tco_ex_shape = l_tco_norm.shape * np.array([1, 2, 1, 1])
        l_tco_norm_split = torch.unsqueeze(l_tco_norm, 0).repeat(tco_ex_shape)
        l_tco_score = torch.unsqueeze(l_score, 0).repeat(tco_ex_shape)
        l_tco_mask = torch.unsqueeze(l_mask, 0).repeat(tco_ex_shape)

        tco_geo_diff = l_tco_split - f_tco_split
        abs_tco_geo_diff = torch.abs(tco_geo_diff)
        tco_sign = abs_tco_geo_diff < 1.0
        tco_sign = tco_sign.type(dtype=torch.float32)
        tco_sign.stop_gradient = True
        tco_in_loss = 0.5 * abs_tco_geo_diff * abs_tco_geo_diff * tco_sign + (
            abs_tco_geo_diff - 0.5
        ) * (1.0 - tco_sign)
        tco_out_loss = l_tco_norm_split * tco_in_loss
        tco_loss = torch.sum(tco_out_loss * l_tco_score * l_tco_mask) / (
            torch.sum(l_tco_score * l_tco_mask) + 1e-5
        )

        # total loss
        tvo_lw, tco_lw = 1.5, 1.5
        score_lw, border_lw = 1.0, 1.0
        total_loss = (
            score_loss * score_lw
            + border_loss * border_lw
            + tvo_loss * tvo_lw
            + tco_loss * tco_lw
        )

        losses = {
            "loss": total_loss,
            "score_loss": score_loss,
            "border_loss": border_loss,
            "tvo_loss": tvo_loss,
            "tco_loss": tco_loss,
        }
        return losses
