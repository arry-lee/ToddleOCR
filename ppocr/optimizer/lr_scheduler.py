# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

import math
from bisect import bisect_right

from torch.optim.lr_scheduler import LRScheduler


class PiecewiseLR(LRScheduler):
    """
    decay_epochs 是一个整数列表，其中包含一个值700，表示在第 700 个 epoch 时进行学习率衰减。
    values 是一个浮点数列表，其中包含两个值0.001和0.0001，分别表示学习率在第 700 个 epoch 前后的取值。
    warmup_epoch: 5 表示前 5 个 epoch 使用一个较小的学习率进行热身训练。
    """
    def __init__(self, optimizer, decay_epochs, values, warmup_epoch=0, last_epoch=-1):
        self.decay_epochs = decay_epochs
        self.values = values
        self.warmup_epoch = warmup_epoch
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epoch:
            return [base_lr * 0.1 for base_lr in self.base_lrs]
        else:
            current_lr_idx = bisect_right(self.decay_epochs, self.last_epoch)
            return [base_lr * self.values[current_lr_idx] for base_lr in self.base_lrs]
