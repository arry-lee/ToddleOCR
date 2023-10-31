import torch
from torch import nn

"""This code is refer from:
https://github.com/hikopensource/DAVAR-Lab-OCR
"""


class SPINAttentionLoss(nn.Module):
    def __init__(self, reduction="mean", ignore_index=-100, **kwargs):
        super().__init__()
        self.loss_func = nn.CrossEntropyLoss(
            reduction=reduction, ignore_index=ignore_index
        )

    def forward(self, predicts, batch):
        targets = batch[1].type(torch.int64)
        targets = targets[:, 1:]  # remove [eos] in label

        # label_lengths = batch[2].astype("int64")
        # batch_size, num_steps, num_classes = predicts.shape[0], predicts.shape[1], predicts.shape[2]
        assert (
            len(targets.shape) == len(list(predicts.shape)) - 1
        ), "The target's shape and inputs's shape is [N, d] and [N, num_steps]"

        inputs = torch.reshape(predicts, [-1, predicts.shape[-1]])
        targets = torch.reshape(targets, [-1])

        return {"loss": self.loss_func(inputs, targets)}
