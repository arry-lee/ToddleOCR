import math
from bisect import bisect_right
from functools import partial, wraps

from torch.optim.lr_scheduler import LRScheduler


def warmup_scheduler(scheduler_class, warmup_epoch):
    """预热调度器，就是在warmup_epochs代内将学习率从0慢慢增加到预热学习率"""

    def decorator(cls):
        # @wraps(cls)
        class WarmUpScheduler(cls):
            def __init__(self, optimizer, *args, **kwargs):
                super().__init__(optimizer, *args, **kwargs)
                # self.warmup_epochs = warmup_epoch

            def get_lr(self):
                if self.last_epoch < warmup_epoch:
                    alpha = self.last_epoch / warmup_epoch
                    return [base_lr * alpha for base_lr in self.base_lrs]
                else:
                    return super().get_lr()

        return WarmUpScheduler

    return decorator(scheduler_class)


class PiecewiseLR(LRScheduler):
    """
    decay_epochs 是一个整数列表，其中包含一个值700，表示在第 700 个 epoch 时进行学习率衰减。
    values 是一个浮点数列表，其中包含两个值0.001和0.0001，分别表示学习率在第 700 个 epoch 前后的取值。
    warmup_epoch: 5 表示前 5 个 epoch 使用一个较小的学习率进行热身训练。
    """

    def __init__(self, optimizer, decay_epochs, values, last_epoch=-1):
        self.decay_epochs = decay_epochs
        self.values = values
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        current_lr_idx = bisect_right(self.decay_epochs, self.last_epoch)
        return [base_lr * self.values[current_lr_idx] for base_lr in self.base_lrs]


# T_max1, T_max2 无法从配置文件中获取，需要手动指定， 使用partial
class TwoStepCosineLR(LRScheduler):
    def __init__(
        self, optimizer, T_max1, T_max2, eta_min=0, last_epoch=-1, verbose=False
    ):
        if not isinstance(T_max1, int):
            raise TypeError(
                "The type of 'T_max1' must be 'int', but received %s." % type(T_max1)
            )
        if not isinstance(T_max2, int):
            raise TypeError(
                "The type of 'T_max2' must be 'int', but received %s." % type(T_max2)
            )
        if not isinstance(eta_min, (float, int)):
            raise TypeError(
                "The type of 'eta_min' must be 'float' or 'int', but received %s."
                % type(eta_min)
            )
        assert T_max1 > 0 and isinstance(
            T_max1, int
        ), "'T_max1' must be a positive integer."
        assert T_max2 > 0 and isinstance(
            T_max2, int
        ), "'T_max1' must be a positive integer."

        self.T_max1 = T_max1
        self.T_max2 = T_max2
        self.eta_min = float(eta_min)
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch <= self.T_max1:
            if self.last_epoch == 0:
                return self.base_lrs
            elif (self.last_epoch - 1 - self.T_max1) % (2 * self.T_max1) == 0:
                return [
                    lr
                    + (base_lr - self.eta_min)
                    * (1 - math.cos(math.pi / self.T_max1))
                    / 2
                    for lr, base_lr in zip(self.get_last_lr(), self.base_lrs)
                ]

            return [
                (1 + math.cos(math.pi * self.last_epoch / self.T_max1))
                / (1 + math.cos(math.pi * (self.last_epoch - 1) / self.T_max1))
                * (lr - self.eta_min)
                + self.eta_min
                for lr in self.get_last_lr()
            ]
        else:
            if (self.last_epoch - 1 - self.T_max2) % (2 * self.T_max2) == 0:
                return [
                    lr
                    + (base_lr - self.eta_min)
                    * (1 - math.cos(math.pi / self.T_max2))
                    / 2
                    for lr, base_lr in zip(self.get_last_lr(), self.base_lrs)
                ]

            return [
                (1 + math.cos(math.pi * self.last_epoch / self.T_max2))
                / (1 + math.cos(math.pi * (self.last_epoch - 1) / self.T_max2))
                * (lr - self.eta_min)
                + self.eta_min
                for lr in self.get_last_lr()
            ]

    def _get_closed_form_lr(self):
        if self.last_epoch <= self.T_max1:
            return [
                self.eta_min
                + (base_lr - self.eta_min)
                * (1 + math.cos(math.pi * self.last_epoch / self.T_max1))
                / 2
                for base_lr in self.base_lrs
            ]
        else:
            return [
                self.eta_min
                + (base_lr - self.eta_min)
                * (1 + math.cos(math.pi * self.last_epoch / self.T_max2))
                / 2
                for base_lr in self.base_lrs
            ]
