# copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
from torch import nn

from .basic import DiceLoss


class EASTLoss(nn.Module):
    """ """

    def __init__(self, eps=1e-6, **kwargs):
        super().__init__()
        self.dice_loss = DiceLoss(eps=eps)

    def forward(self, predicts, labels):
        l_score, l_geo, l_mask = labels[1:]
        f_score = predicts["f_score"]
        f_geo = predicts["f_geo"]

        dice_loss = self.dice_loss(f_score, l_score, l_mask)

        # smoooth_l1_loss
        channels = 8
        l_geo_split = torch.split(l_geo, 1, dim=1)
        f_geo_split = torch.split(f_geo, 1, dim=1)
        smooth_l1 = 0
        for i in range(0, channels):
            geo_diff = l_geo_split[i] - f_geo_split[i]
            abs_geo_diff = torch.abs(geo_diff)
            smooth_l1_sign = torch.lt(abs_geo_diff, l_score)
            smooth_l1_sign = smooth_l1_sign.type(dtype=torch.float32)
            in_loss = abs_geo_diff * abs_geo_diff * smooth_l1_sign + (
                abs_geo_diff - 0.5
            ) * (1.0 - smooth_l1_sign)
            out_loss = l_geo_split[-1] / channels * in_loss * l_score
            smooth_l1 += out_loss
        smooth_l1_loss = torch.mean(smooth_l1 * l_score)

        dice_loss = dice_loss * 0.01
        total_loss = dice_loss + smooth_l1_loss
        losses = {
            "loss": total_loss,
            "dice_loss": dice_loss,
            "smooth_l1_loss": smooth_l1_loss,
        }
        return losses


# class EASTLoss(nn.Module):
#     def __init__(self, eps=1e-6):
#         super().__init__()
#         self.eps = eps
#
#     def forward(self, preds, labels):
#         f_score = preds['f_score']
#         f_geo = preds['f_geo']
#         g_score, g_geo, g_mask = labels[1:]
#
#         # Dice loss
#         intersection = torch.sum(f_score * g_score * g_mask)
#         union = torch.sum(f_score * g_mask) + torch.sum(g_score * g_mask) + self.eps
#         dice_loss = 1 - (2 * intersection / union)
#
#         # Smooth L1 loss
#         smooth_l1_loss = torch.sum(torch.abs(f_geo - g_geo) * g_score * g_mask) / (
#                     torch.sum(g_score * g_mask) + self.eps)
#
#         total_loss = dice_loss + smooth_l1_loss
#
#         losses = {
#             'loss': total_loss,
#             'dice_loss': dice_loss,
#             'smooth_l1_loss': smooth_l1_loss
#         }
#
#         return losses
