# copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This code is refer from:
https://github.com/WenmuZhou/DBNet.pytorch/blob/master/models/losses/basic_loss.py
"""




import torch
import torch.nn.functional as F
from torch import nn


class BalanceLoss(nn.Module):
    def __init__(
        self, balance_loss=True, main_loss_type="DiceLoss", negative_ratio=3, return_origin=False, eps=1e-6, **kwargs
    ):
        """
        The BalanceLoss for Differentiable Binarization text detection
        args:
            balance_loss (bool): whether balance loss or not, default is True
            main_loss_type (str): can only be one of ['CrossEntropy','DiceLoss',
                'Euclidean','BCELoss', 'MaskL1Loss'], default is  'DiceLoss'.
            negative_ratio (int|float): float, default is 3.
            return_origin (bool): whether return unbalanced loss or not, default is False.
            eps (float): default is 1e-6.
        """
        super(BalanceLoss, self).__init__()
        self.balance_loss = balance_loss
        self.main_loss_type = main_loss_type
        self.negative_ratio = negative_ratio
        self.return_origin = return_origin
        self.eps = eps

        if self.main_loss_type == "CrossEntropy":
            self.loss = nn.CrossEntropyLoss()
        elif self.main_loss_type == "Euclidean":
            self.loss = nn.MSELoss()
        elif self.main_loss_type == "DiceLoss":
            self.loss = DiceLoss(self.eps)
        elif self.main_loss_type == "BCELoss":
            self.loss = BCELoss(reduction="none")
        elif self.main_loss_type == "MaskL1Loss":
            self.loss = MaskL1Loss(self.eps)
        else:
            loss_type = ["CrossEntropy", "DiceLoss", "Euclidean", "BCELoss", "MaskL1Loss"]
            raise Exception("main_loss_type in BalanceLoss() can only be one of {}".format(loss_type))

    def forward(self, pred, gt, mask=None):
        """
        The BalanceLoss for Differentiable Binarization text detection
        args:
            pred (variable): predicted feature maps.
            gt (variable): ground truth feature maps.
            mask (variable): masked maps.
        return: (variable) balanced loss
        """
        positive = gt * mask
        negative = (1 - gt) * mask

        positive_count = int(positive.sum())
        negative_count = int(min(negative.sum(), positive_count * self.negative_ratio))
        loss = self.loss(pred, gt, mask=mask)

        if not self.balance_loss:
            return loss

        positive_loss = positive * loss
        negative_loss = negative * loss
        negative_loss = torch.reshape(negative_loss, shape=[-1])
        if negative_count > 0:
            sort_loss = negative_loss.sort(descending=True)
            negative_loss = sort_loss[:negative_count]
            # negative_loss, _ = torch.topk(negative_loss, k=negative_count_int)
            balance_loss = (positive_loss.sum() + negative_loss.sum()) / (positive_count + negative_count + self.eps)
        else:
            balance_loss = positive_loss.sum() / (positive_count + self.eps)
        if self.return_origin:
            return balance_loss, loss

        return balance_loss


class DiceLoss(nn.Module):
    """
    Dice Loss是一种用于图像分割任务的损失函数，它衡量了预测分割结果与真实分割结果之间的重叠程度。
    Dice Loss的全称是Sørensen–Dice系数损失（Sørensen–Dice Coefficient Loss），也被称为F1 Score Loss。
    Dice Loss的背景可以追溯到医学图像分割领域，用于评估算法在分割任务中的性能。
    在图像分割中，常常需要将图像中的不同物体或区域进行准确的定位和分割。
    Dice Loss通过计算预测的分割结果与真实的分割结果之间的重叠程度，提供了一种评估分割准确性的指标。
    具体而言，Dice Loss使用预测分割结果、真实分割结果和一个掩码（用于标记感兴趣的区域）来计算重叠度，并将其转化为损失值。
    Dice Loss的取值范围为0到1，其中0表示完全不重叠，1表示完全重叠。
    因此，Dice Loss的目标是最小化损失值，以使预测分割结果与真实分割结果尽可能接近。
    Dice Loss相比于其他常用的损失函数如交叉熵损失函数，更加适合于处理不平衡的类别分割问题，因为它对类别之间的重叠区域更加敏感。
    Dice Loss的应用不仅限于医学图像分割，还可以用于其他图像分割任务，如语义分割、实例分割等。
    它在深度学习模型训练中被广泛使用，并且经常与其他损失函数结合使用来提高分割模型的性能。
    总而言之，Dice Loss是一种用于图像分割任务的损失函数，通过衡量预测分割结果与真实分割结果之间的重叠程度来评估分割准确性。
    它在医学图像分割和其他相关领域中被广泛应用。
    """
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred, gt, mask, weights=None):
        """
        DiceLoss function.
        """

        assert pred.shape == gt.shape
        assert pred.shape == mask.shape
        if weights is not None:
            assert weights.shape == mask.shape
            mask = weights * mask
        intersection = torch.sum(pred * gt * mask)

        union = torch.sum(pred * mask) + torch.sum(gt * mask) + self.eps
        loss = 1 - 2.0 * intersection / union
        assert loss <= 1
        return loss


class MaskL1Loss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred, gt, mask):
        """
        Mask L1 Loss
        """
        loss = (torch.abs(pred - gt) * mask).sum() / (mask.sum() + self.eps)
        loss = torch.mean(loss)
        return loss



class BCELoss(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, input, label, mask=None, weight=None, name=None): # for mask
        loss = F.binary_cross_entropy(input, label, reduction=self.reduction)
        return loss
