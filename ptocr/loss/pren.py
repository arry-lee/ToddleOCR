import torch
from torch import nn


class PRENLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        # note: 0 is padding idx
        self.loss_func = nn.CrossEntropyLoss(ignore_index=0)

    def forward(self, predicts, batch):
        loss = self.loss_func(predicts, batch[1].type(torch.int64))
        return {"loss": loss}
