import torch
from torch import nn


class AttentionLoss(nn.Module):
    """AttentionLoss 是一个自定义的损失函数，它通常与注意力机制（Attention Mechanism）相关的模型一起使用。
    在注意力机制中，模型尝试通过对输入的不同部分分配不同的权重来聚焦于相关的信息。

    AttentionLoss 的作用是通过最小化损失来优化注意力机制相关模型的性能。
    它根据模型的输出和目标之间的差异计算损失值，并将其用于反向传播优化模型参数。

    具体实现中，AttentionLoss 常常使用交叉熵损失（CrossEntropyLoss）作为基本损失函数。
    交叉熵损失用于比较模型的预测结果和真实标签之间的差异，可以用于分类任务。

    在 AttentionLoss 的前向传播方法中，通常会根据模型的预测结果和真实标签计算交叉熵损失，并将其返回。
    这样，在训练过程中，通过最小化 AttentionLoss 来优化模型的参数，以改善模型对关注的准确性和表达能力。

    需要注意的是，AttentionLoss 可能在不同的应用场景中有不同的具体实现方式，具体实现可能会因模型而异。
    因此，对于 AttentionLoss 的详细信息，最好参考具体的模型或文献中的描述。"""

    def __init__(self):
        super().__init__()
        self.loss_func = nn.CrossEntropyLoss(reduction="none")

    def forward(self, predicts, batch):
        targets = batch[1].type(torch.int64)
        # label_lengths = batch[2].astype("int64")
        batch_size, num_steps, num_classes = (
            predicts.shape[0],
            predicts.shape[1],
            predicts.shape[2],
        )
        assert (
            len(targets.shape) == len(list(predicts.shape)) - 1
        ), "The target's shape and inputs's shape is [N, d] and [N, num_steps]"

        inputs = predicts.view(-1, num_classes)
        targets = targets.view(-1)

        return {"loss": torch.sum(self.loss_func(inputs, targets))}
