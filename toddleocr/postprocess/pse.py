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
"""
This code is refer from:
https://github.com/whai362/PSENet/blob/python3/models/head/psenet_head.py
"""


from queue import Queue

import cv2

# from ppocr.postprocess.pse_postprocess.pse import pse

import numpy as np
import torch
from torch.nn import functional as F


def _pse(kernels, label, kernel_num, label_num, min_area=0):
    pred = np.zeros((label.shape[0], label.shape[1]), dtype=np.int32)

    for label_idx in range(1, label_num):
        if np.sum(label == label_idx) < min_area:
            label[label == label_idx] = 0

    dx = [-1, 1, 0, 0]
    dy = [0, 0, -1, 1]

    points = np.array(np.where(label > 0)).transpose((1, 0))
    que = Queue()
    for point_idx in range(points.shape[0]):
        tmpx, tmpy = points[point_idx, 0], points[point_idx, 1]
        que.put((tmpx, tmpy))
        pred[tmpx, tmpy] = label[tmpx, tmpy]

    while not que.empty():
        cur = que.get()
        cur_label = pred[cur[0], cur[1]]

        is_edge = True
        for j in range(4):
            tmpx = cur[0] + dx[j]
            tmpy = cur[1] + dy[j]
            if tmpx < 0 or tmpx >= label.shape[0] or tmpy < 0 or tmpy >= label.shape[1]:
                continue
            if kernels[kernel_idx, tmpx, tmpy] == 0 or pred[tmpx, tmpy] > 0:
                continue

            que.put((tmpx, tmpy))
            pred[tmpx, tmpy] = cur_label
            is_edge = False
        if is_edge:
            nxt_que.put(cur)

    return pred


def pse(kernels, min_area):
    kernel_num = kernels.shape[0]
    label_num, label = cv2.connectedComponents(kernels[-1], connectivity=4)
    return _pse(kernels[:-1], label, kernel_num, label_num, min_area)


class PSEPostProcess:
    """
    The post process for PSE.
    """

    def __init__(
        self,
        thresh=0.5,
        box_thresh=0.85,
        min_area=16,
        box_type="quad",
        scale=4,
        **kwargs
    ):
        assert box_type in ["quad", "poly"], "Only quad and poly is supported"
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.min_area = min_area
        self.box_type = box_type
        self.scale = scale

    def __call__(self, outs_dict, shape_list):
        pred = outs_dict["maps"]
        if not isinstance(pred, torch.Tensor):
            pred = torch.Tensor(pred)
        pred = F.interpolate(pred, scale_factor=4 // self.scale, mode="bilinear")

        score = F.sigmoid(pred[:, 0, :, :])

        kernels = (pred > self.thresh).astype("float32")
        text_mask = kernels[:, 0, :, :]
        text_mask = torch.unsqueeze(text_mask, dim=1)

        kernels[:, 0:, :, :] = kernels[:, 0:, :, :] * text_mask

        score = score.numpy()
        kernels = kernels.numpy().astype(np.uint8)

        boxes_batch = []
        for batch_index in range(pred.shape[0]):
            boxes, scores = self.boxes_from_bitmap(
                score[batch_index], kernels[batch_index], shape_list[batch_index]
            )

            boxes_batch.append({"points": boxes, "scores": scores})
        return boxes_batch

    def boxes_from_bitmap(self, score, kernels, shape):
        label = pse(kernels, self.min_area)
        return self.generate_box(score, label, shape)

    def generate_box(self, score, label, shape):
        src_h, src_w, ratio_h, ratio_w = shape
        label_num = np.max(label) + 1

        boxes = []
        scores = []
        for i in range(1, label_num):
            ind = label == i
            points = np.array(np.where(ind)).transpose((1, 0))[:, ::-1]

            if points.shape[0] < self.min_area:
                label[ind] = 0
                continue

            score_i = np.mean(score[ind])
            if score_i < self.box_thresh:
                label[ind] = 0
                continue

            if self.box_type == "quad":
                rect = cv2.minAreaRect(points)
                bbox = cv2.boxPoints(rect)
            elif self.box_type == "poly":
                box_height = np.max(points[:, 1]) + 10
                box_width = np.max(points[:, 0]) + 10

                mask = np.zeros((box_height, box_width), np.uint8)
                mask[points[:, 1], points[:, 0]] = 255

                contours, _ = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                bbox = np.squeeze(contours[0], 1)
            else:
                raise NotImplementedError

            bbox[:, 0] = np.clip(np.round(bbox[:, 0] / ratio_w), 0, src_w)
            bbox[:, 1] = np.clip(np.round(bbox[:, 1] / ratio_h), 0, src_h)
            boxes.append(bbox)
            scores.append(score_i)
        return boxes, scores
