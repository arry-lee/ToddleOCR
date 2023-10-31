# This code is refer from: https://github.com/KaiyangZhou/pytorch-center-loss


import os
import pickle

import torch
import torch.nn as nn


class CenterLoss(nn.Module):
    """
    Reference: Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    """

    def __init__(self, num_classes=6625, feat_dim=96, center_file_path=None):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.centers = torch.randn(
            size=[self.num_classes, self.feat_dim], dtype=torch.float64
        )

        if center_file_path is not None:
            assert os.path.exists(
                center_file_path
            ), f"center path({center_file_path}) must exist when it is not None."
            with open(center_file_path, "rb") as f:
                char_dict = pickle.load(f)
                for key in char_dict.keys():
                    self.centers[key] = torch.tensor(char_dict[key])

    def forward(self, predicts, batch=None):
        assert isinstance(predicts, (list, tuple))
        features, predicts = predicts

        # Reshape特征向量为2D
        feats_reshape = torch.reshape(features, [-1, features.shape[-1]]).type(
            torch.float64
        )

        # 计算每个样本预测的类别
        label = torch.argmax(predicts, dim=2)
        label = torch.reshape(label, [label.shape[0] * label.shape[1]])

        batch_size = feats_reshape.shape[0]

        # 计算L2距离平方（特征和中心位置之间）
        square_feat = torch.sum(torch.square(feats_reshape), dim=1, keepdim=True)
        square_feat = square_feat.expand(batch_size, self.num_classes)

        square_center = torch.sum(torch.square(self.centers), dim=1, keepdim=True)
        square_center = square_center.expand(self.num_classes, batch_size).type(
            torch.float64
        )
        square_center = square_center.permute(1, 0)

        # 计算特征和中心之间的欧氏距离
        distmat = torch.add(square_feat, square_center)
        feat_dot_center = torch.matmul(feats_reshape, self.centers.transpose(1, 0))
        distmat = distmat - 2.0 * feat_dot_center

        # 生成掩码来选择对应类别的距离进行计算
        classes = torch.arange(self.num_classes).type(torch.int64)
        label = label.unsqueeze(1).view(batch_size, 1)
        label = label.expand(batch_size, self.num_classes)

        mask = torch.eq(
            classes.unsqueeze(0).expand(batch_size, self.num_classes), label
        ).type(torch.float64)
        dist = torch.multiply(distmat, mask)

        # 计算损失
        loss = torch.sum(torch.clip(dist, min=1e-12, max=1e12)) / batch_size

        return {"loss_center": loss}


# fixme
# # 创建 CenterLoss 实例
# center_loss = CenterLoss(num_classes=10, feat_dim=8)
#
# # 生成模拟数据
# features = torch.randn([32, 8])  # 特征向量
# predicts = torch.randn([32,1,10])  # 预测结果
#
# # 计算损失
# losses = center_loss((features, predicts), batch=32)
#
# # 打印损失值
# print(losses["loss_center"])
