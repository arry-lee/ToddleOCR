# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch import nn
import torch

from .det_basic_loss import DiceLoss
from ppocr.utils.e2e_utils.extract_batchsize import pre_process


class PGLoss(nn.Module):
    def __init__(self, tcl_bs, max_text_length, max_text_nums, pad_num, eps=1e-6, **kwargs):
        super(PGLoss, self).__init__()
        self.tcl_bs = tcl_bs
        self.max_text_nums = max_text_nums
        self.max_text_length = max_text_length
        self.pad_num = pad_num
        self.dice_loss = DiceLoss(eps=eps)

    def border_loss(self, f_border, l_border, l_score, l_mask):
        l_border_split, l_border_norm = torch.tensor.split(l_border, [4, 1], dim=1)
        f_border_split = f_border
        b, c, h, w = l_border_norm.shape
        l_border_norm_split = torch.unsqueeze(l_border_norm, 0).repeat(b, 4 * c, h, w)
        b, c, h, w = l_score.shape
        l_border_score = torch.unsqueeze(l_score, 0).repeat(b, 4 * c, h, w)
        b, c, h, w = l_mask.shape
        l_border_mask = torch.unsqueeze(l_mask, 0).repeat(b, 4 * c, h, w)
        border_diff = l_border_split - f_border_split
        abs_border_diff = torch.abs(border_diff)
        border_sign = abs_border_diff < 1.0
        border_sign = border_sign.type(dtype=torch.float32)
        border_sign.stop_gradient = True
        border_in_loss = 0.5 * abs_border_diff * abs_border_diff * border_sign + (abs_border_diff - 0.5) * (1.0 - border_sign)
        border_out_loss = l_border_norm_split * border_in_loss
        border_loss = torch.sum(border_out_loss * l_border_score * l_border_mask) / (torch.sum(l_border_score * l_border_mask) + 1e-5)
        return border_loss

    def direction_loss(self, f_direction, l_direction, l_score, l_mask):
        l_direction_split, l_direction_norm = torch.tensor.split(l_direction, [2, 1], dim=1)
        f_direction_split = f_direction
        b, c, h, w = l_direction_norm.shape
        l_direction_norm_split = torch.unsqueeze(l_direction_norm, 0).repeat(b, 2 * c, h, w)
        b, c, h, w = l_score.shape
        l_direction_score = torch.unsqueeze(l_score, 0).repeat(b, 2 * c, h, w)
        b, c, h, w = l_mask.shape
        l_direction_mask = torch.unsqueeze(l_mask, 0).repeat(b, 2 * c, h, w)
        direction_diff = l_direction_split - f_direction_split
        abs_direction_diff = torch.abs(direction_diff)
        direction_sign = abs_direction_diff < 1.0
        direction_sign = direction_sign.type(dtype=torch.float32)
        direction_sign.stop_gradient = True
        direction_in_loss = 0.5 * abs_direction_diff * abs_direction_diff * direction_sign + (abs_direction_diff - 0.5) * (1.0 - direction_sign)
        direction_out_loss = l_direction_norm_split * direction_in_loss
        direction_loss = torch.sum(direction_out_loss * l_direction_score * l_direction_mask) / (torch.sum(l_direction_score * l_direction_mask) + 1e-5)
        return direction_loss

    def ctcloss(self, f_char, tcl_pos, tcl_mask, tcl_label, label_t):
        f_char = f_char.permute(0, 2, 3, 1)
        tcl_pos = torch.reshape(tcl_pos, [-1, 3])
        tcl_pos = tcl_pos.type(torch.int)
        f_tcl_char = torch.gather_nd(f_char, tcl_pos)
        f_tcl_char = torch.reshape(f_tcl_char, [-1, 64, self.pad_num + 1])  # len(Lexicon_Table)+1
        f_tcl_char_fg, f_tcl_char_bg = torch.split(f_tcl_char, [self.pad_num, 1], dim=2)
        f_tcl_char_bg = f_tcl_char_bg * tcl_mask + (1.0 - tcl_mask) * 20.0
        b, c, l = tcl_mask.shape
        tcl_mask_fg = torch.unsqueeze(tcl_mask, 0).repeat(b, c, self.pad_num * l)
        tcl_mask_fg.stop_gradient = True
        f_tcl_char_fg = f_tcl_char_fg * tcl_mask_fg + (1.0 - tcl_mask_fg) * (-20.0)
        f_tcl_char_mask = torch.concat([f_tcl_char_fg, f_tcl_char_bg], dim=2)
        f_tcl_char_ld = f_tcl_char_mask.permute(1, 0, 2)
        N, B, _ = f_tcl_char_ld.shape
        input_lengths = torch.Tensor([N] * B, dtype="int64")
        cost = torch.nn.functional.ctc_loss(log_probs=f_tcl_char_ld, labels=tcl_label, input_lengths=input_lengths, label_lengths=label_t, blank=self.pad_num, reduction="none")
        cost = cost.mean()
        return cost

    def forward(self, predicts, labels):
        images, tcl_maps, tcl_label_maps, border_maps, direction_maps, training_masks, label_list, pos_list, pos_mask = labels
        # for all the batch_size
        pos_list, pos_mask, label_list, label_t = pre_process(label_list, pos_list, pos_mask, self.max_text_length, self.max_text_nums, self.pad_num, self.tcl_bs)

        f_score, f_border, f_direction, f_char = predicts["f_score"], predicts["f_border"], predicts["f_direction"], predicts["f_char"]
        score_loss = self.dice_loss(f_score, tcl_maps, training_masks)
        border_loss = self.border_loss(f_border, border_maps, tcl_maps, training_masks)
        direction_loss = self.direction_loss(f_direction, direction_maps, tcl_maps, training_masks)
        ctc_loss = self.ctcloss(f_char, pos_list, pos_mask, label_list, label_t)
        loss_all = score_loss + border_loss + direction_loss + 5 * ctc_loss

        losses = {"loss": loss_all, "score_loss": score_loss, "border_loss": border_loss, "direction_loss": direction_loss, "ctc_loss": ctc_loss}
        return losses
