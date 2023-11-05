import torch

__all__ = ["CTCLoss"]

from torch import nn

from .ace import ACELoss
from .center import CenterLoss


class CTCLoss(nn.Module):
    """The Connectionist Temporal Classification loss"""

    def __init__(self, use_focal_loss=False):
        super().__init__()
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


class EnhancedCTCLoss(nn.Module):
    """引入了focal_loss ace_loss 和 center_loss的增强ctc_loss"""
    def __init__(
        self,
        use_focal_loss=False,
        use_ace_loss=False,
        ace_loss_weight=0.1,
        use_center_loss=False,
        center_loss_weight=0.05,
        num_classes=6625,
        feat_dim=96,
        center_file_path=None,
        **kwargs
    ):
        super().__init__()
        self.ctc_loss_func = CTCLoss(use_focal_loss=use_focal_loss)

        self.use_ace_loss = use_ace_loss
        self.use_center_loss = use_center_loss

        if use_ace_loss:
            self.ace_loss_func = ACELoss()
            self.ace_loss_weight = ace_loss_weight

        if use_center_loss:
            self.use_center_loss = use_center_loss
            self.center_loss_func = CenterLoss(
                num_classes=num_classes,
                feat_dim=feat_dim,
                center_file_path=center_file_path,
            )
            self.center_loss_weight = center_loss_weight

    def __call__(self, predicts, batch):
        loss = self.ctc_loss_func(predicts, batch)["loss"]

        if self.use_center_loss:
            center_loss = (
                self.center_loss_func(predicts, batch)["loss_center"]
                * self.center_loss_weight
            )
            loss = loss + center_loss

        if self.use_ace_loss:
            ace_loss = (
                self.ace_loss_func(predicts, batch)["loss_ace"] * self.ace_loss_weight
            )
            loss = loss + ace_loss

        return {"enhanced_ctc_loss": loss}
