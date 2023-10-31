import torch
from torch import nn


class ClsLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.loss_func = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, predicts, batch):
        label = batch[1].type(torch.int64)
        loss = self.loss_func(input=predicts, target=label)
        return {"loss": loss}
