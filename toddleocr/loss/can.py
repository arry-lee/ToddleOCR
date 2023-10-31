"""
This code is refer from:
https://github.com/LBH1024/CAN/models/can.py
"""

import numpy as np
import torch
import torch.nn as nn

__all__ = ["CANLoss"]


class CANLoss(nn.Module):
    """
    CANLoss 包含两部分损失函数：

    word_average_loss：用于计算符号（symbol）的平均准确率损失。它使用交叉熵损失函数（CrossEntropyLoss）。
    平均准确率损失衡量了模型对每个符号的分类准确性。

    counting_loss：用于计算每个符号的计数损失。它使用平滑的 L1 损失函数（SmoothL1Loss）。
    计数损失衡量了模型对每个符号数量的预测准确性。

    """

    def __init__(self):
        super().__init__()

        self.use_label_mask = False
        self.out_channel = 111
        self.cross = (
            nn.CrossEntropyLoss(reduction="none")
            if self.use_label_mask
            else nn.CrossEntropyLoss()
        )
        self.counting_loss = nn.SmoothL1Loss(reduction="mean")
        self.ratio = 16

    def forward(self, preds, batch):
        word_probs = preds[0]
        counting_preds = preds[1]
        counting_preds1 = preds[2]
        counting_preds2 = preds[3]
        labels = batch[2]
        labels_mask = batch[3]
        counting_labels = gen_counting_label(labels, self.out_channel, True)
        counting_loss = (
            self.counting_loss(counting_preds1, counting_labels)
            + self.counting_loss(counting_preds2, counting_labels)
            + self.counting_loss(counting_preds, counting_labels)
        )

        word_loss = self.cross(
            torch.reshape(word_probs, [-1, word_probs.shape[-1]]),
            torch.reshape(labels, [-1]),
        )
        word_average_loss = (
            torch.sum(torch.reshape(word_loss * labels_mask, [-1]))
            / (torch.sum(labels_mask) + 1e-10)
            if self.use_label_mask
            else word_loss
        )
        loss = word_average_loss + counting_loss
        return {"loss": loss}


def gen_counting_label(labels, channel, tag):
    b, t = labels.shape
    counting_labels = np.zeros([b, channel])

    if tag:
        ignore = [0, 1, 107, 108, 109, 110]
    else:
        ignore = []
    for i in range(b):
        for j in range(t):
            k = labels[i][j]
            if k in ignore:
                continue
            else:
                counting_labels[i][k] += 1
    counting_labels = torch.tensor(counting_labels, dtype=torch.float32)
    return counting_labels
