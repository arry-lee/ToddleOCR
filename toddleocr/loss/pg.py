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
import copy

import numpy as np
import torch

from torch import nn

from .basic import DiceLoss


class PGLoss(nn.Module):
    def __init__(
        self, tcl_bs, max_text_length, max_text_nums, pad_num, eps=1e-6, **kwargs
    ):
        super().__init__()
        self.tcl_bs = tcl_bs
        self.max_text_nums = max_text_nums
        self.max_text_length = max_text_length
        self.pad_num = pad_num
        self.dice_loss = DiceLoss(eps=eps)

    def border_loss(self, f_border, l_border, l_score, l_mask):
        l_border_split, l_border_norm = torch.split(l_border, [4, 1], dim=1)
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
        border_in_loss = 0.5 * abs_border_diff * abs_border_diff * border_sign + (
            abs_border_diff - 0.5
        ) * (1.0 - border_sign)
        border_out_loss = l_border_norm_split * border_in_loss
        border_loss = torch.sum(border_out_loss * l_border_score * l_border_mask) / (
            torch.sum(l_border_score * l_border_mask) + 1e-5
        )
        return border_loss

    def direction_loss(self, f_direction, l_direction, l_score, l_mask):
        l_direction_split, l_direction_norm = torch.split(l_direction, [2, 1], dim=1)
        f_direction_split = f_direction
        b, c, h, w = l_direction_norm.shape
        l_direction_norm_split = torch.unsqueeze(l_direction_norm, 0).repeat(
            b, 2 * c, h, w
        )
        b, c, h, w = l_score.shape
        l_direction_score = torch.unsqueeze(l_score, 0).repeat(b, 2 * c, h, w)
        b, c, h, w = l_mask.shape
        l_direction_mask = torch.unsqueeze(l_mask, 0).repeat(b, 2 * c, h, w)
        direction_diff = l_direction_split - f_direction_split
        abs_direction_diff = torch.abs(direction_diff)
        direction_sign = abs_direction_diff < 1.0
        direction_sign = direction_sign.type(dtype=torch.float32)
        direction_sign.stop_gradient = True
        direction_in_loss = (
            0.5 * abs_direction_diff * abs_direction_diff * direction_sign
            + (abs_direction_diff - 0.5) * (1.0 - direction_sign)
        )
        direction_out_loss = l_direction_norm_split * direction_in_loss
        direction_loss = torch.sum(
            direction_out_loss * l_direction_score * l_direction_mask
        ) / (torch.sum(l_direction_score * l_direction_mask) + 1e-5)
        return direction_loss

    def ctcloss(self, f_char, tcl_pos, tcl_mask, tcl_label, label_t):
        f_char = f_char.permute(0, 2, 3, 1)
        tcl_pos = torch.reshape(tcl_pos, [-1, 3])
        tcl_pos = tcl_pos.type(torch.int)
        f_tcl_char = torch.gather(f_char, None, tcl_pos)
        f_tcl_char = torch.reshape(
            f_tcl_char, [-1, 64, self.pad_num + 1]
        )  # len(Lexicon_Table)+1
        f_tcl_char_fg, f_tcl_char_bg = torch.split(f_tcl_char, [self.pad_num, 1], dim=2)
        f_tcl_char_bg = f_tcl_char_bg * tcl_mask + (1.0 - tcl_mask) * 20.0
        b, c, l = tcl_mask.shape
        tcl_mask_fg = torch.unsqueeze(tcl_mask, 0).repeat(b, c, self.pad_num * l)
        tcl_mask_fg.stop_gradient = True
        f_tcl_char_fg = f_tcl_char_fg * tcl_mask_fg + (1.0 - tcl_mask_fg) * (-20.0)
        f_tcl_char_mask = torch.concat([f_tcl_char_fg, f_tcl_char_bg], dim=2)
        f_tcl_char_ld = f_tcl_char_mask.permute(1, 0, 2)
        N, B, _ = f_tcl_char_ld.shape
        input_lengths = torch.Tensor([N] * B, dtype=torch.int64)
        cost = torch.nn.functional.ctc_loss(
            log_probs=f_tcl_char_ld,
            targets=tcl_label,
            input_lengths=input_lengths,
            target_lengths=label_t,
            blank=self.pad_num,
            reduction="none",
        )
        cost = cost.mean()
        return cost

    def forward(self, predicts, labels):
        (
            images,
            tcl_maps,
            tcl_label_maps,
            border_maps,
            direction_maps,
            training_masks,
            label_list,
            pos_list,
            pos_mask,
        ) = labels
        # for all the batch_size
        pos_list, pos_mask, label_list, label_t = pre_process(
            label_list,
            pos_list,
            pos_mask,
            self.max_text_length,
            self.max_text_nums,
            self.pad_num,
            self.tcl_bs,
        )

        f_score, f_border, f_direction, f_char = (
            predicts["f_score"],
            predicts["f_border"],
            predicts["f_direction"],
            predicts["f_char"],
        )
        score_loss = self.dice_loss(f_score, tcl_maps, training_masks)
        border_loss = self.border_loss(f_border, border_maps, tcl_maps, training_masks)
        direction_loss = self.direction_loss(
            f_direction, direction_maps, tcl_maps, training_masks
        )
        ctc_loss = self.ctcloss(f_char, pos_list, pos_mask, label_list, label_t)
        loss_all = score_loss + border_loss + direction_loss + 5 * ctc_loss

        losses = {
            "loss": loss_all,
            "score_loss": score_loss,
            "border_loss": border_loss,
            "direction_loss": direction_loss,
            "ctc_loss": ctc_loss,
        }
        return losses


def org_tcl_rois(batch_size, pos_lists, pos_masks, label_lists, tcl_bs):
    """ """
    pos_lists_, pos_masks_, label_lists_ = [], [], []
    img_bs = batch_size
    ngpu = int(batch_size / img_bs)
    img_ids = np.array(pos_lists, dtype=np.int32)[:, 0, 0].copy()
    pos_lists_split, pos_masks_split, label_lists_split = [], [], []
    for i in range(ngpu):
        pos_lists_split.append([])
        pos_masks_split.append([])
        label_lists_split.append([])

    for i in range(img_ids.shape[0]):
        img_id = img_ids[i]
        gpu_id = int(img_id / img_bs)
        img_id = img_id % img_bs
        pos_list = pos_lists[i].copy()
        pos_list[:, 0] = img_id
        pos_lists_split[gpu_id].append(pos_list)
        pos_masks_split[gpu_id].append(pos_masks[i].copy())
        label_lists_split[gpu_id].append(copy.deepcopy(label_lists[i]))
    # repeat or delete
    for i in range(ngpu):
        vp_len = len(pos_lists_split[i])
        if vp_len <= tcl_bs:
            for j in range(0, tcl_bs - vp_len):
                pos_list = pos_lists_split[i][j].copy()
                pos_lists_split[i].append(pos_list)
                pos_mask = pos_masks_split[i][j].copy()
                pos_masks_split[i].append(pos_mask)
                label_list = copy.deepcopy(label_lists_split[i][j])
                label_lists_split[i].append(label_list)
        else:
            for j in range(0, vp_len - tcl_bs):
                c_len = len(pos_lists_split[i])
                pop_id = np.random.permutation(c_len)[0]
                pos_lists_split[i].pop(pop_id)
                pos_masks_split[i].pop(pop_id)
                label_lists_split[i].pop(pop_id)
    # merge
    for i in range(ngpu):
        pos_lists_.extend(pos_lists_split[i])
        pos_masks_.extend(pos_masks_split[i])
        label_lists_.extend(label_lists_split[i])
    return pos_lists_, pos_masks_, label_lists_


def pre_process(
    label_list, pos_list, pos_mask, max_text_length, max_text_nums, pad_num, tcl_bs
):
    label_list = label_list.numpy()
    batch, _, _, _ = label_list.shape
    pos_list = pos_list.numpy()
    pos_mask = pos_mask.numpy()
    pos_list_t = []
    pos_mask_t = []
    label_list_t = []
    for i in range(batch):
        for j in range(max_text_nums):
            if pos_mask[i, j].any():
                pos_list_t.append(pos_list[i][j])
                pos_mask_t.append(pos_mask[i][j])
                label_list_t.append(label_list[i][j])
    pos_list, pos_mask, label_list = org_tcl_rois(
        batch, pos_list_t, pos_mask_t, label_list_t, tcl_bs
    )
    label = []
    tt = [l.tolist() for l in label_list]
    for i in range(tcl_bs):
        k = 0
        for j in range(max_text_length):
            if tt[i][j][0] != pad_num:
                k += 1
            else:
                break
        label.append(k)
    label = torch.Tensor(label)
    label = label.type(dtype=torch.int64)
    pos_list = torch.Tensor(pos_list)
    pos_mask = torch.Tensor(pos_mask)
    label_list = torch.squeeze(torch.Tensor(label_list), dim=2)
    label_list = label_list.type(dtype=torch.int32)
    return pos_list, pos_mask, label_list, label
