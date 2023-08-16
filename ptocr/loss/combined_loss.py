import torch
import torch.nn as nn

from .rec_ctc_loss import CTCLoss
from .center_loss import CenterLoss
from .ace_loss import ACELoss
from .rec_sar_loss import SARLoss

from .distillation_loss import DistillationCTCLoss
from .distillation_loss import DistillationSARLoss
from .distillation_loss import DistillationDMLLoss
from .distillation_loss import DistillationDistanceLoss, DistillationDBLoss, DistillationDilaDBLoss
from .distillation_loss import DistillationVQASerTokenLayoutLMLoss, DistillationSERDMLLoss
from .distillation_loss import DistillationLossFromOutput
from .distillation_loss import DistillationVQADistanceLoss


import torch
import torch.nn as nn

from .rec_ctc_loss import CTCLoss
from .center_loss import CenterLoss
from .ace_loss import ACELoss
from .rec_sar_loss import SARLoss

from .distillation_loss import DistillationCTCLoss
from .distillation_loss import DistillationSARLoss
from .distillation_loss import DistillationDMLLoss
from .distillation_loss import DistillationDistanceLoss, DistillationDBLoss, DistillationDilaDBLoss
from .distillation_loss import DistillationVQASerTokenLayoutLMLoss, DistillationSERDMLLoss
from .distillation_loss import DistillationLossFromOutput
from .distillation_loss import DistillationVQADistanceLoss


class CombinedLoss(nn.Module):
    """
    CombinedLoss:
        组合多个损失函数的类
    """

    def __init__(self, loss_config_list=None):
        super().__init__()
        self.loss_func = []  # 存储损失函数实例的列表
        self.loss_weight = []  # 存储每个损失函数的权重

        assert isinstance(loss_config_list, list), 'operator config should be a list'
        for config in loss_config_list:
            assert isinstance(config, dict) and len(config) == 1, "yaml format error"
            name = list(config)[0]  # 获取损失函数的名称
            param = config[name]  # 获取损失函数的参数配置
            assert "weight" in param, "weight must be in param, but param just contains {}".format(param.keys())
            self.loss_weight.append(param.pop("weight"))  # 获取并移除权重参数
            self.loss_func.append(eval(name)(**param))  # 根据名称和参数创建损失函数实例 fixme 有注入可能

    def forward(self, input, batch, **kwargs):
        loss_dict = {}  # 存储每个损失函数的计算结果
        loss_all = 0.  # 总损失值

        for idx, loss_func in enumerate(self.loss_func):
            loss = loss_func(input, batch, **kwargs)  # 计算损失
            if isinstance(loss, torch.Tensor):
                loss = {"loss_{}_{}".format(str(loss), idx): loss}  # 如果损失是Tensor，则将其包装为字典形式

            weight = self.loss_weight[idx]  # 获取当前损失函数的权重

            loss = {key: loss[key] * weight for key in loss}  # 根据权重对损失进行加权

            if "loss" in loss:
                loss_all += loss["loss"]  # 将损失累加到总损失值中
            else:
                loss_all += sum(loss.values())

            loss_dict.update(loss)  # 更新损失字典

        loss_dict["loss"] = loss_all  # 将总损失值存入损失字典

        return loss_dict
