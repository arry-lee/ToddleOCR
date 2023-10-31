import torch
import torch.nn as nn
from torch import nn

from .ace import ACELoss
from .center import CenterLoss

from .ctc import CTCLoss

from .distillation import (
    DistillationCTCLoss,
    DistillationDBLoss,
    DistillationDilaDBLoss,
    DistillationDistanceLoss,
    DistillationDMLLoss,
    DistillationLossFromOutput,
    DistillationSARLoss,
    DistillationSERDMLLoss,
    DistillationVQADistanceLoss,
    DistillationVQASerTokenLayoutLMLoss,
)
from .sar import SARLoss


class CombinedLoss(nn.Module):
    """
    CombinedLoss:
        组合多个损失函数的类
    """

    def __init__(self, loss_config_list=None):
        super().__init__()
        self.loss_func = []  # 存储损失函数实例的列表
        self.loss_weight = []  # 存储每个损失函数的权重

        assert isinstance(loss_config_list, list), "operator config should be a list"
        for config in loss_config_list:
            assert isinstance(config, dict) and len(config) == 1, "yaml format error"
            name = list(config)[0]  # 获取损失函数的名称
            param = config[name]  # 获取损失函数的参数配置
            assert (
                "weight" in param
            ), "weight must be in param, but param just contains {}".format(
                param.keys()
            )
            self.loss_weight.append(param.pop("weight"))  # 获取并移除权重参数
            self.loss_func.append(eval(name)(**param))  # 根据名称和参数创建损失函数实例 fixme 有注入可能

    def forward(self, input, batch, **kwargs):
        loss_dict = {}  # 存储每个损失函数的计算结果
        loss_all = 0.0  # 总损失值

        for idx, loss_func in enumerate(self.loss_func):
            loss = loss_func(input, batch, **kwargs)  # 计算损失
            if isinstance(loss, torch.Tensor):
                loss = {
                    "loss_{}_{}".format(str(loss), idx): loss
                }  # 如果损失是Tensor，则将其包装为字典形式

            weight = self.loss_weight[idx]  # 获取当前损失函数的权重

            loss = {key: loss[key] * weight for key in loss}  # 根据权重对损失进行加权

            if "loss" in loss:
                loss_all = loss_all + loss["loss"]  # 将损失累加到总损失值中
            else:
                loss_all = loss_all + sum(loss.values())

            loss_dict.update(loss)  # 更新损失字典

        loss_dict["loss"] = loss_all  # 将总损失值存入损失字典

        return loss_dict


class MultiLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.loss_funcs = {}
        self.loss_list = kwargs.pop("loss_config_list")
        self.weight_1 = kwargs.get("weight_1", 1.0)
        self.weight_2 = kwargs.get("weight_2", 1.0)
        self.gtc_loss = kwargs.get("gtc_loss", "sar")
        for loss_info in self.loss_list:
            for name, param in loss_info.items():
                if param is not None:
                    kwargs.update(param)
                loss = eval(name)(**kwargs)
                self.loss_funcs[name] = loss

    def forward(self, predicts, batch):
        self.total_loss = {}
        total_loss = 0.0
        # batch [image, label_ctc, label_sar, length, valid_ratio]
        for name, loss_func in self.loss_funcs.items():
            if name == "CTCLoss":
                loss = (
                    loss_func(predicts["ctc"], batch[:2] + batch[3:])["loss"]
                    * self.weight_1
                )
            elif name == "SARLoss":
                loss = (
                    loss_func(predicts["sar"], batch[:1] + batch[2:])["loss"]
                    * self.weight_2
                )
            else:
                raise NotImplementedError(
                    "{} is not supported in MultiLoss yet".format(name)
                )
            self.total_loss[name] = loss
            total_loss += loss
        self.total_loss["loss"] = total_loss
        return self.total_loss
