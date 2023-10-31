import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import nn

__all__ = ["Kie_backbone"]


class Encoder(nn.Module):
    def __init__(self, num_features, num_filters):
        super().__init__()
        self.conv1 = nn.Conv2d(
            num_features, num_filters, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(num_filters)

        self.conv2 = nn.Conv2d(
            num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(num_filters)

        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x_pooled = self.pool(x)
        return x, x_pooled


class Decoder(nn.Module):
    def __init__(self, num_features, num_filters):
        super().__init__()

        self.conv1 = nn.Conv2d(
            num_features, num_filters, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(num_filters)

        self.conv2 = nn.Conv2d(
            num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(num_filters)

        self.conv0 = nn.Conv2d(
            num_features, num_filters, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn0 = nn.BatchNorm2d(num_filters)

    def forward(self, inputs_prev, inputs):
        x = self.conv0(inputs)
        x = self.bn0(x)
        x = F.relu(x)
        x = torch.nn.functional.interpolate(
            x, scale_factor=2, mode="bilinear", align_corners=False
        )
        x = torch.concat([inputs_prev, x], dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        return x


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = Encoder(num_features=3, num_filters=16)
        self.down2 = Encoder(num_features=16, num_filters=32)
        self.down3 = Encoder(num_features=32, num_filters=64)
        self.down4 = Encoder(num_features=64, num_filters=128)
        self.down5 = Encoder(num_features=128, num_filters=256)

        self.up1 = Decoder(32, 16)
        self.up2 = Decoder(64, 32)
        self.up3 = Decoder(128, 64)
        self.up4 = Decoder(256, 128)
        self.out_channels = 16

    def forward(self, inputs):
        x1, _ = self.down1(inputs)
        _, x2 = self.down2(x1)
        _, x3 = self.down3(x2)
        _, x4 = self.down4(x3)
        _, x5 = self.down5(x4)

        x = self.up4(x4, x5)
        x = self.up3(x3, x)
        x = self.up2(x2, x)
        x = self.up1(x1, x)
        return x


class Kie_backbone(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super().__init__()
        self.out_channels = 16
        self.img_feat = UNet()
        self.maxpool = nn.MaxPool2d(kernel_size=7)

    def bbox2roi(self, bbox_list):
        rois_list = []
        rois_num = []
        for img_id, bboxes in enumerate(bbox_list):
            rois_num.append(bboxes.shape[0])
            rois_list.append(bboxes)
        rois = torch.concat(rois_list, 0)
        rois_num = torch.Tensor(rois_num, dtype=torch.int32)
        return rois, rois_num

    def pre_process(self, img, relations, texts, gt_bboxes, tag, img_size):
        img, relations, texts, gt_bboxes, tag, img_size = (
            img.numpy(),
            relations.numpy(),
            texts.numpy(),
            gt_bboxes.numpy(),
            tag.numpy().tolist(),
            img_size.numpy(),
        )
        temp_relations, temp_texts, temp_gt_bboxes = [], [], []
        h, w = int(np.max(img_size[:, 0])), int(np.max(img_size[:, 1]))
        img = torch.Tensor(img[:, :, :h, :w])
        batch = len(tag)
        for i in range(batch):
            num, recoder_len = tag[i][0], tag[i][1]
            temp_relations.append(
                torch.Tensor(relations[i, :num, :num, :], dtype=torch.float32)
            )
            temp_texts.append(
                torch.Tensor(texts[i, :num, :recoder_len], dtype=torch.float32)
            )
            temp_gt_bboxes.append(
                torch.Tensor(gt_bboxes[i, :num, ...], dtype=torch.float32)
            )
        return img, temp_relations, temp_texts, temp_gt_bboxes

    def forward(self, inputs):
        img = inputs[0]
        relations, texts, gt_bboxes, tag, img_size = (
            inputs[1],
            inputs[2],
            inputs[3],
            inputs[5],
            inputs[-1],
        )
        img, relations, texts, gt_bboxes = self.pre_process(
            img, relations, texts, gt_bboxes, tag, img_size
        )
        x = self.img_feat(img)
        boxes, rois_num = self.bbox2roi(gt_bboxes)
        feats = torchvision.ops.roi_align(
            x, boxes, spatial_scale=1.0, output_size=7, boxes_num=rois_num
        )
        feats = self.maxpool(feats).squeeze(-1).squeeze(-1)
        return [relations, texts, feats]
