# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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

# This code is refer from: https://github.com/viig99/LS-ACELoss

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn


class ACELoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.loss_func = nn.CrossEntropyLoss(weight=None, ignore_index=0, reduction="none", soft_label=True, axis=-1)

    def __call__(self, predicts, batch):
        if isinstance(predicts, (list, tuple)):
            predicts = predicts[-1]

        B, N = predicts.shape[:2]
        div = torch.Tensor([N]).astype("float32")

        predicts = nn.functional.softmax(predicts,dim=-1)
        aggregation_preds = torch.sum(predicts, dim=1)
        aggregation_preds = torch.divide(aggregation_preds, div)

        length = batch[2].astype("float32")
        batch = batch[3].astype("float32")
        batch[:, 0] = torch.subtract(div, length)
        batch = torch.divide(batch, div)

        loss = self.loss_func(aggregation_preds, batch)
        return {"loss_ace": loss}
