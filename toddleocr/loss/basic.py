import torch
from torch import nn
from torch.nn import functional as F


def label_smooth(label, prior_dist=None, eps=0.1):
    """
    标签平滑是一种常用的正则化技术，用于改善深度学习模型在训练集上的性能。它通过减少标签的确定性，引入一定程度的噪声，以防止模型对训练数据的过拟合。

    具体而言，标签平滑会将每个目标标签的概率值从 0 或 1 调整为介于 0 和 1 之间的一个平滑值。这样可以避免模型过度自信地预测某一类别，并鼓励模型更加关注不确定性较大的类别。

    在实际应用中，标签平滑可通过以下步骤实现：

    将原始的离散标签转换为 one-hot 编码，得到独热编码的目标标签。
    对每个类别的独热编码进行平滑处理，使得每个类别的概率值更加平滑和接近均匀。
    将平滑后的概率值作为新的目标标签，用于模型的训练。
    Label smoothing is a mechanism to regularize the classifier layer and is called
    label-smoothing regularization (LSR).

    Label smoothing replaces the ground-truth label y with the weighted sum
    of itself and some fixed distribution mu. For class k, i.e.

    tilde{y_k} = (1 - eps) * y_k + eps * mu_k,

    where 1 - eps and eps are the weights respectively,
    and tilde{y}_k is the smoothed label. Usually uniform distribution is used for mu.

    Parameters:
        label (torch.Tensor): The input tensor containing the label data. The
            label data should use one-hot representation. Its shape should be [N_1, ..., Depth],
            where Depth is the number of classes. The dtype should be torch.float32 or torch.float64.
        prior_dist (torch.Tensor, optional): The prior distribution to be used to smooth labels.
            If not provided, a uniform distribution is used. Its shape should be [1, class_num].
            The default value is None.
        eps (float, optional): The weight used to mix up the original ground-truth distribution and
            the fixed distribution. The default value is 0.1.

    Returns:
        torch.Tensor: The tensor containing the smoothed labels.
    """
    if eps > 1.0 or eps < 0.0:
        raise ValueError("The value of eps must be between 0 and 1.")

    label = label.float()
    if prior_dist is not None:
        prior_dist = prior_dist.float()

    if prior_dist is None:
        prior_dist = torch.ones_like(label) / label.size(-1)

    smooth_label = (1 - eps) * label + eps * prior_dist
    return smooth_label


# class CELoss(nn.Module):
#     """交叉熵"""
#     def __init__(self, eps=None):
#         super().__init__()
#         if eps is not None and (eps <= 0 or eps >= 1):
#             eps = None
#         self.eps = eps
#
#     def _labelsmoothing(self, target, class_num):
#         if target.shape[-1] != class_num:
#             one_hot_target = F.one_hot(target, class_num)
#         else:
#             one_hot_target = target
#         soft_target = label_smooth(one_hot_target, eps=self.eps)
#         soft_target = torch.reshape(soft_target, shape=[-1, class_num])
#         return soft_target
#
#     def forward(self, x, label):
#         loss_dict = {}
#         if self.eps is not None:  # 如果 eps 不为空，则进行标签平滑处理
#             class_num = x.shape[-1]
#             label = self._labelsmoothing(label, class_num)  # 调用 _labelsmoothing 方法对目标标签进行平滑处理
#             x = -F.log_softmax(x, dim=-1)  # 对预测值进行 softmax 归一化，并取其负对数为新的预测值
#             loss = torch.sum(x * label, dim=-1)  # 计算加权的交叉熵损失，乘积相加
#         else:
#             if label.shape[-1] == x.shape[-1]:  # 如果标签的类别数量与预测值的类别数量相同，则认为标签已经进行了 softmax 归一化
#                 label = F.softmax(label, dim=-1)
#                 soft_label = True  # 标记为已进行软标签处理
#             else:
#                 soft_label = False  # 标记为未进行软标签处理
#             loss = F.cross_entropy(x, label)  # 直接使用 PyTorch 的交叉熵损失函数计算损失
#         return loss
#


class KLJSLoss:
    """用于计算 KL 散度（Kullback-Leibler Divergence）和 JS 散度（Jensen-Shannon Divergence）的损失函数"""

    def __init__(self, mode="kl"):
        assert mode in [
            "kl",
            "js",
            "KL",
            "JS",
        ], "mode can only be one of ['kl', 'KL', 'js', 'JS']"
        self.mode = mode

    def __call__(self, p1, p2, reduction="mean", eps=1e-5):
        if self.mode.lower() == "kl":
            loss = torch.multiply(p2, torch.log((p2 + eps) / (p1 + eps) + eps))
            loss = loss + torch.multiply(p1, torch.log((p1 + eps) / (p2 + eps) + eps))
            loss = loss * 0.5
        elif self.mode.lower() == "js":
            loss = torch.multiply(p2, torch.log((2 * p2 + eps) / (p1 + p2 + eps) + eps))
            loss = loss + torch.multiply(
                p1, torch.log((2 * p1 + eps) / (p1 + p2 + eps) + eps)
            )
            loss = loss * 0.5
        else:
            raise ValueError(
                "The mode.lower() if KLJSLoss should be one of ['kl', 'js']"
            )

        if reduction == "mean":
            loss = torch.mean(loss, dim=[1, 2])
        elif reduction == "none" or reduction is None:
            return loss
        else:
            loss = torch.sum(loss, dim=[1, 2])

        return loss


class DMLLoss(nn.Module):
    """
    用于计算特征蒸馏（Distilled Metric Learning）中的损失。
    在初始化函数 __init__ 中，根据传入的 act 参数选择激活函数类型，可选的值包括 "softmax" 和 "sigmoid"。如果 act 是 "softmax"，则使用 nn.Softmax 进行softmax激活；如果 act 是 "sigmoid"，则使用 nn.Sigmoid 进行sigmoid激活；如果 act 为其他值或为空，则不进行激活。同时，根据传入的 use_log 参数确定是否进行对数转换。
    在 _kldiv 函数中，计算 KL 散度的损失。根据输入的 x 和 target 计算 KL 散度公式的每个元素的差值，然后求和并取平均得到平均损失。
    在 forward 函数中，首先根据激活函数类型对输入进行激
    最后，返回计算得到的损失。

    这个类的作用是提供了一个方便计算特征蒸馏损失的方法，可用于在度量学习中进行模型训练和优化。
    """

    def __init__(self, act=None, use_log=False):
        super().__init__()
        if act is not None:
            assert act in ["softmax", "sigmoid"]
        if act == "softmax":
            self.act = nn.Softmax(dim=-1)
        elif act == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            self.act = None

        self.use_log = use_log
        self.jskl_loss = KLJSLoss(mode="kl")

    def _kldiv(self, x, target):
        eps = 1.0e-10
        loss = target * (torch.log(target + eps) - x)
        # batch mean loss
        loss = torch.sum(loss) / loss.shape[0]
        return loss

    def forward(self, out1, out2):
        if self.act is not None:
            out1 = self.act(out1) + 1e-10
            out2 = self.act(out2) + 1e-10
        if self.use_log:
            # for recognition distillation, log is needed for feature map
            log_out1 = torch.log(out1)
            log_out2 = torch.log(out2)
            loss = (self._kldiv(log_out1, out2) + self._kldiv(log_out2, out1)) / 2.0
        else:
            # for detection distillation log is not needed
            loss = self.jskl_loss(out1, out2)
        return loss


class DistanceLoss(nn.Module):
    """
    SmoothL1Loss 和 L1Loss 是两种常见的回归损失函数，主要用于度量预测值与目标值之间的差异。它们在计算损失时的计算方式有所不同。

    L1Loss（绝对值损失）：L1Loss 是一种简单的损失函数，计算预测值和目标值之间的绝对值差，并将这些差值求和作为损失。它对异常值比较敏感，因为差值的绝对值会直接参与求和计算。

    SmoothL1Loss（平滑 L1 损失）：SmoothL1Loss 是一种更平滑的损失函数，它在差值较小的情况下使用 L1 损失（绝对值损失），在差值较大的情况下使用 L2 损失（均方损失）。它通过引入一个平滑的转换区域，使得在差值较小的情况下损失函数更平滑，从而减少了异常值对损失的影响。

    具体来说，SmoothL1Loss 在差值小于一个阈值时，使用 L1 损失进行计算；在差值大于等于该阈值时，使用 L2 损失进行计算。这样能够在一定程度上平衡异常值对损失的影响，使得损失函数更加鲁棒。

    总的来说，SmoothL1Loss 相对于 L1Loss 具有更好的数值稳定性，并且能够更好地处理离群点或异常值。在一些需要平衡精度和鲁棒性的回归任务中，SmoothL1Loss 通常是一个较好的选择。
    """

    def __init__(self, mode="l2", **kargs):
        super().__init__()
        assert mode in ["l1", "l2", "smooth_l1"]
        if mode == "l1":
            self.loss_func = nn.L1Loss(**kargs)
        elif mode == "l2":
            self.loss_func = nn.MSELoss(**kargs)
        elif mode == "smooth_l1":
            self.loss_func = nn.SmoothL1Loss(**kargs)

    def forward(self, x, y):
        return self.loss_func(x, y)


class LossFromOutput(nn.Module):
    def __init__(self, key="loss", reduction="none"):
        super().__init__()
        self.key = key
        self.reduction = reduction

    def forward(self, predicts, batch):
        loss = predicts
        if self.key is not None and isinstance(predicts, dict):
            loss = loss[self.key]
        if self.reduction == "mean":
            loss = torch.mean(loss)
        elif self.reduction == "sum":
            loss = torch.sum(loss)
        return {"loss": loss}


class BalanceLoss(nn.Module):
    def __init__(
        self,
        balance_loss=True,
        main_loss_type="DiceLoss",
        negative_ratio=3,
        return_origin=False,
        eps=1e-6,
        **kwargs
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
        super().__init__()
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
            loss_type = [
                "CrossEntropy",
                "DiceLoss",
                "Euclidean",
                "BCELoss",
                "MaskL1Loss",
            ]
            raise Exception(
                "main_loss_type in BalanceLoss() can only be one of {}".format(
                    loss_type
                )
            )

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
        # torch.sort()
        if negative_count > 0:
            sort_loss, _ = negative_loss.sort(descending=True)
            # print(sort_loss,_)
            negative_loss = sort_loss[:negative_count]
            # negative_loss, _ = torch.topk(negative_loss, k=negative_count_int)
            balance_loss = (positive_loss.sum() + negative_loss.sum()) / (
                positive_count + negative_count + self.eps
            )
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

    def forward(self, input, label, mask=None, weight=None, name=None):  # for mask
        loss = F.binary_cross_entropy(input, label, reduction=self.reduction)
        return loss
