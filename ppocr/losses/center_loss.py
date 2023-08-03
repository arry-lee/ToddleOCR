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

# This code is refer from: https://github.com/KaiyangZhou/pytorch-center-loss

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F


class CenterLoss(nn.Module):
    """
    Reference: Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    """

    def __init__(self, num_classes=6625, feat_dim=96, center_file_path=None):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.centers = torch.randn([self.num_classes, self.feat_dim]).astype("float64")

        if center_file_path is not None:
            assert os.path.exists(center_file_path), f"center path({center_file_path}) must exist when it is not None."
            with open(center_file_path, "rb") as f:
                char_dict = pickle.load(f)
                for key in char_dict.keys():
                    self.centers[key] = torch.Tensor(char_dict[key])

    def __call__(self, predicts, batch):
        assert isinstance(predicts, (list, tuple))
        features, predicts = predicts

        feats_reshape = torch.reshape(features, [-1, features.shape[-1]]).astype("float64")
        label = torch.argmax(predicts, dim=2)
        label = torch.reshape(label, [label.shape[0] * label.shape[1]])

        batch_size = feats_reshape.shape[0]

        # calc l2 distance between feats and centers
        square_feat = torch.sum(torch.square(feats_reshape), dim=1, keepdim=True)
        square_feat = torch.unsqueeze(square_feat, 0).repeat(batch_size, self.num_classes)

        square_center = torch.sum(torch.square(self.centers), dim=1, keepdim=True)
        square_center = torch.unsqueeze(square_center, 0).repeat(self.num_classes, batch_size).astype("float64")
        square_center = torch.transpose(square_center, [1, 0])

        distmat = torch.add(square_feat, square_center)
        feat_dot_center = torch.matmul(feats_reshape, torch.transpose(self.centers, [1, 0]))
        distmat = distmat - 2.0 * feat_dot_center

        # generate the mask
        classes = torch.arange(self.num_classes).astype("int64")
        label = torch.unsqueeze(torch.unsqueeze(label, 0).repeat(1), (batch_size, self.num_classes))
        mask = torch.equal(torch.unsqueeze(classes, 0).repeat(batch_size, self.num_classes), label).astype("float64")
        dist = torch.multiply(distmat, mask)

        loss = torch.sum(torch.clip(dist, min=1e-12, max=1e12)) / batch_size
        return {"loss_center": loss}
