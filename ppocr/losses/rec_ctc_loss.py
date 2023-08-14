import torch
import torch.nn as nn

__all__ = ['CTCLoss']

class CTCLoss(nn.Module):
    """The Connectionist Temporal Classification loss"""
    def __init__(self, use_focal_loss=False):
        super(CTCLoss, self).__init__()
        self.loss_func = nn.CTCLoss(blank=0, reduction="none")
        self.use_focal_loss = use_focal_loss

    def forward(self, predicts, batch):
        if isinstance(predicts, (list, tuple)):
            predicts = predicts[-1]
        predicts = predicts.transpose((1, 0, 2))
        N, B, _ = predicts.shape
        preds_lengths = torch.Tensor([N] * B, dtype=torch.int64, device=predicts.device)
        labels = batch[1].to(torch.int32)
        label_lengths = batch[2].to(torch.int64)
        loss = self.loss_func(predicts, labels, preds_lengths, label_lengths)

        if self.use_focal_loss:
            weight = torch.exp(-loss)
            weight = torch.subtract(torch.Tensor([1.0], device=loss.device), weight)
            weight = torch.square(weight)
            loss = torch.multiply(loss, weight)

        loss = loss.mean()
        return {"loss": loss}
