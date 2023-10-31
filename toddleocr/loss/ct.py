# CT代表Connectionist Temporal Classification，是一种用于序列标注任务的算法。它常被用于语音识别和光学字符识别（OCR）等领域。
#
# 在传统的序列标注任务中，通常使用隐马尔可夫模型（HMM）进行建模和推断。然而，HMM对于输入和输出序列长度不匹配的情况处理起来并不方便。
#
# CT算法通过使用一个中间的"blank"标签来进行扩展，将输入序列映射到输出序列的对应位置上。这样，可以实现输入和输出序列长度不一致的情况下的标注。具体而言，CT算法会对输入序列的每个时间步进行预测，并将结果映射到输出序列中的对应位置。同时，它还会考虑到"blank"标签的存在，以处理多个连续的相同字符或连续空白的情况。
#
# CT算法的训练过程通常使用梯度下降优化算法进行，目标是最大化正确标签的概率，并最小化其它非正确标签的概率。这样可以提高模型在序列标注任务中的表现。
#
# 总而言之，CT算法是一种用于序列标注任务的算法，通过引入"blank"标签和特定的映射方式，解决了输入和输出序列长度不一致的问题。它在语音识别和OCR等领域有着广泛的应用
"""
This code is refer from:
https://github.com/shengtao96/CentripetalText/tree/main/models/loss
"""


import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def ohem_single(score, gt_text, training_mask):
    # online hard example mining

    pos_num = int(torch.sum(gt_text > 0.5)) - int(
        torch.sum((gt_text > 0.5) & (training_mask <= 0.5))
    )

    if pos_num == 0:
        # selected_mask = gt_text.copy() * 0 # may be not good
        selected_mask = training_mask
        selected_mask = selected_mask.reshape(
            (1, selected_mask.shape[0], selected_mask.shape[1])
        ).type(dtype=torch.float32)
        return selected_mask

    neg_num = int(torch.sum((gt_text <= 0.5) & (training_mask > 0.5)))
    neg_num = int(min(pos_num * 3, neg_num))

    if neg_num == 0:
        selected_mask = training_mask
        selected_mask = selected_mask.reshape(
            (1, selected_mask.shape[0], selected_mask.shape[1])
        ).type(dtype=torch.float32)
        return selected_mask

    # hard example
    neg_score = score[(gt_text <= 0.5) & (training_mask > 0.5)]
    neg_score_sorted = torch.sort(-neg_score)
    threshold = -neg_score_sorted[neg_num - 1]

    selected_mask = ((score >= threshold) | (gt_text > 0.5)) & (training_mask > 0.5)
    selected_mask = selected_mask.reshape(
        (1, selected_mask.shape[0], selected_mask.shape[1])
    ).type(dtype=torch.float32)
    return selected_mask


def ohem_batch(scores, gt_texts, training_masks):
    selected_masks = []
    for i in range(scores.shape[0]):
        selected_masks.append(
            ohem_single(scores[i, :, :], gt_texts[i, :, :], training_masks[i, :, :])
        )

    selected_masks = torch.concat(selected_masks, 0).type(dtype=torch.float32)
    return selected_masks


def iou_single(a, b, mask, n_class):
    EPS = 1e-6
    valid = mask == 1
    a = a[valid]
    b = b[valid]
    miou = []

    # iou of each class
    for i in range(n_class):
        inter = ((a == i) & (b == i)).type(dtype=torch.float32)
        union = ((a == i) | (b == i)).type(dtype=torch.float32)

        miou.append(torch.sum(inter) / (torch.sum(union) + EPS))
    miou = sum(miou) / len(miou)
    return miou


def iou(a, b, mask, n_class=2, reduce=True):
    batch_size = a.shape[0]

    a = a.reshape((batch_size, -1))
    b = b.reshape((batch_size, -1))
    mask = mask.reshape((batch_size, -1))

    iou = torch.zeros((batch_size,), dtype=torch.float32)
    for i in range(batch_size):
        iou[i] = iou_single(a[i], b[i], mask[i], n_class)

    if reduce:
        iou = torch.mean(iou)
    return iou


class DiceLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, input, target, mask, reduce=True):
        batch_size = input.shape[0]
        input = F.sigmoid(input)  # scale to 0-1

        input = input.reshape((batch_size, -1))
        target = target.reshape((batch_size, -1)).type(dtype=torch.float32)
        mask = mask.reshape((batch_size, -1)).type(dtype=torch.float32)

        input = input * mask
        target = target * mask

        a = torch.sum(input * target, dim=1)
        b = torch.sum(input * input, dim=1) + 0.001
        c = torch.sum(target * target, dim=1) + 0.001
        d = (2 * a) / (b + c)
        loss = 1 - d

        loss = self.loss_weight * loss

        if reduce:
            loss = torch.mean(loss)

        return loss


class SmoothL1Loss(nn.Module):
    def __init__(self, beta=1.0, loss_weight=1.0):
        super().__init__()
        self.beta = beta
        self.loss_weight = loss_weight

        np_coord = np.zeros(shape=[640, 640, 2], dtype=np.int64)
        for i in range(640):
            for j in range(640):
                np_coord[i, j, 0] = j
                np_coord[i, j, 1] = i
        np_coord = np_coord.reshape((-1, 2))

        self.coord = nn.Parameter(
            torch.tensor(np_coord, dtype=torch.int32), requires_grad=False
        )  # Note: In PyTorch, int32 type is supported by default
        self.register_buffer("eps", torch.tensor(1e-6))

    def forward_single(self, input, target, mask, beta=1.0):
        batch_size = input.shape[0]

        diff = torch.abs(input - target) * mask.unsqueeze(1)
        loss = torch.where(diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta)
        loss = loss.reshape((batch_size, -1)).float()
        mask = mask.reshape((batch_size, -1)).float()
        loss = torch.sum(loss, dim=-1)
        loss = loss / (mask.sum(dim=-1) + self.eps)

        return loss

    def select_single(self, distance, gt_instance, gt_kernel_instance, training_mask):
        with torch.no_grad():
            select_distance_list = []
            for i in range(2):
                tmp1 = distance[i, :]
                tmp2 = tmp1[self.coord[:, 1], self.coord[:, 0]]
                select_distance_list.append(tmp2.unsqueeze(0))
            select_distance = torch.cat(select_distance_list, dim=0)

            off_points = self.coord.float() + 10 * select_distance.transpose(1, 0)
            off_points = off_points.long().clamp(0, distance.shape[-1] - 1)

            selected_mask = (
                gt_instance[self.coord[:, 1], self.coord[:, 0]]
                != gt_kernel_instance[off_points[:, 1], off_points[:, 0]]
            )
            selected_mask = selected_mask.reshape((1, -1, distance.shape[-1])).long()
            selected_training_mask = selected_mask * training_mask

            return selected_training_mask

    def forward(
        self,
        distances,
        gt_instances,
        gt_kernel_instances,
        training_masks,
        gt_distances,
        reduce=True,
    ):
        selected_training_masks = []
        for i in range(distances.shape[0]):
            selected_training_masks.append(
                self.select_single(
                    distances[i, :, :, :],
                    gt_instances[i, :, :],
                    gt_kernel_instances[i, :, :],
                    training_masks[i, :, :],
                )
            )
        selected_training_masks = torch.cat(selected_training_masks, 0).float()

        loss = self.forward_single(
            distances, gt_distances, selected_training_masks, self.beta
        )
        loss = self.loss_weight * loss

        with torch.no_grad():
            batch_size = distances.shape[0]
            false_num = selected_training_masks.reshape((batch_size, -1)).sum(dim=-1)
            total_num = training_masks.reshape((batch_size, -1)).float().sum(dim=-1)
            iou_text = (total_num - false_num) / (total_num + self.eps)

        if reduce:
            loss = torch.mean(loss)

        return loss, iou_text


class CTLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.kernel_loss = DiceLoss()
        self.loc_loss = SmoothL1Loss(beta=0.1, loss_weight=0.05)

    def forward(self, preds, batch):
        imgs = batch[0]
        out = preds["maps"]
        (
            gt_kernels,
            training_masks,
            gt_instances,
            gt_kernel_instances,
            training_mask_distances,
            gt_distances,
        ) = batch[1:]

        kernels = out[:, 0, :, :]
        distances = out[:, 1:, :, :]

        # kernel loss
        selected_masks = ohem_batch(kernels, gt_kernels, training_masks)

        loss_kernel = self.kernel_loss(
            kernels, gt_kernels, selected_masks, reduce=False
        )

        iou_kernel = iou(
            (kernels > 0).type(dtype=torch.int64),
            gt_kernels,
            training_masks,
            reduce=False,
        )
        losses = dict(
            loss_kernels=loss_kernel,
        )

        # loc loss
        loss_loc, iou_text = self.loc_loss(
            distances,
            gt_instances,
            gt_kernel_instances,
            training_mask_distances,
            gt_distances,
            reduce=False,
        )
        losses.update(
            dict(
                loss_loc=loss_loc,
            )
        )

        loss_all = loss_kernel + loss_loc
        losses = {"loss": loss_all}

        return losses
