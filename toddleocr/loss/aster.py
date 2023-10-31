import torch
from torch import nn

__all__ = ["AsterLoss"]


class CosineEmbeddingLoss(nn.Module):
    """余弦嵌入损失函数"""

    def __init__(self, margin=0.0):
        super().__init__()
        self.margin = margin
        self.eps = 1e-12

    def forward(self, x1, x2, target):
        similarity = torch.sum(x1 * x2, dim=-1) / (
            torch.norm(x1, dim=-1) * torch.norm(x2, dim=-1) + self.eps
        )
        one_list = torch.full_like(target, fill_value=1)
        out = torch.mean(
            torch.where(
                torch.eq(target, one_list),
                1.0 - similarity,
                torch.maximum(torch.zeros_like(similarity), similarity - self.margin),
            )
        )

        return out


class AsterLoss(nn.Module):
    """AsterLoss 是一个自定义的损失函数，用于目标检测或文本识别任务中。
    AsterLoss 通过结合两个子损失函数来计算总体损失：semantic loss（语义损失）和recognition loss（识别损失）。

    该损失函数的主要目的是在训练过程中通过最小化损失来优化模型的性能。下面是 AsterLoss 类的主要属性和方法：

    属性：

    weight：权重参数，用于控制不同损失之间的相对重要性。
    size_average：布尔值，确定是否将损失归一化为标量。
    ignore_index：指定要忽略的目标索引。
    sequence_normalize：布尔值，确定是否对序列进行归一化。
    sample_normalize：布尔值，确定是否对样本进行归一化。"""

    def __init__(
        self,
        weight=None,
        size_average=True,
        ignore_index=-100,
        sequence_normalize=False,
        sample_normalize=True,
        **kwargs
    ):
        super().__init__()
        self.weight = weight
        self.size_average = size_average
        self.ignore_index = ignore_index
        self.sequence_normalize = sequence_normalize
        self.sample_normalize = sample_normalize
        self.loss_sem = CosineEmbeddingLoss()
        self.is_cosine_loss = True
        self.loss_func_rec = nn.CrossEntropyLoss(reduction="none")

    def forward(self, predicts, batch):
        targets = batch[1].type(torch.int64)
        label_lengths = batch[2].type(torch.int64)
        sem_target = batch[3].type(torch.float32)
        embedding_vectors = predicts["embedding_vectors"]
        rec_pred = predicts["rec_pred"]

        if not self.is_cosine_loss:
            sem_loss = torch.sum(self.loss_sem(embedding_vectors, sem_target))
        else:
            label_target = torch.ones([embedding_vectors.shape[0]])
            sem_loss = torch.sum(
                self.loss_sem(embedding_vectors, sem_target, label_target)
            )

        # rec loss
        batch_size, def_max_length = targets.shape[0], targets.shape[1]

        mask = torch.zeros([batch_size, def_max_length])
        for i in range(batch_size):
            mask[i, : label_lengths[i]] = 1
        mask = mask.type(dtype=torch.float32)
        max_length = max(label_lengths)
        assert max_length == rec_pred.shape[1]
        targets = targets[:, :max_length]
        mask = mask[:, :max_length]
        rec_pred = torch.reshape(rec_pred, [-1, rec_pred.shape[2]])
        input = nn.functional.log_softmax(rec_pred, dim=1)
        targets = torch.reshape(targets, [-1, 1])
        mask = torch.reshape(mask, [-1, 1])
        output = -torch.gather(input, 1, index=targets) * mask
        output = torch.sum(output)
        if self.sequence_normalize:
            output = output / torch.sum(mask)
        if self.sample_normalize:
            output = output / batch_size

        loss = output + sem_loss * 0.1
        return {"loss": loss}
