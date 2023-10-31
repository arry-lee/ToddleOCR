import torch
import torch.nn as nn


class ACELoss(nn.Module):
    """聚合交叉熵 ACE（Aggregated Cross-Entropy）损失"""

    def __init__(self, **kwargs):
        super().__init__()
        self.loss_func = nn.CrossEntropyLoss(
            weight=None, ignore_index=0, reduction="none"
        )

    def forward(self, predicts, batch):
        if isinstance(predicts, (list, tuple)):
            predicts = predicts[-1]

        B, N = predicts.shape[:2]  # 获取输入 predicts 的形状，B 表示 batch size，N 表示序列长度
        div = torch.tensor([N], dtype=torch.float32)  # 用于进行归一化的除数

        predicts = nn.functional.softmax(predicts, dim=-1)  # 对输入的预测进行 softmax 归一化
        aggregation_preds = torch.sum(predicts, dim=1)  # 对 softmax 归一化后的预测结果进行求和，得到聚合预测
        aggregation_preds = torch.divide(aggregation_preds, div)  # 将聚合预测结果除以除数进行归一化操作

        length = batch[2].float()  # 获取 batch 中的长度信息，转换为 float 类型
        batch = batch[3].float()  # 获取 batch 数据的一部分，转换为 float 类型
        batch[:, 0] = torch.sub(div, length)  # 将除数减去长度信息，并赋值给 batch 的第一列
        batch = torch.divide(batch, div)  # 对 batch 进行归一化操作

        loss = self.loss_func(aggregation_preds, batch)  # 计算 ACE 损失
        return {"loss_ace": loss}  # 返回包含 ACE 损失的字典形式的输出
