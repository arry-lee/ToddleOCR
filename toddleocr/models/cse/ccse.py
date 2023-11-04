"""
1. Dataset 的输出是 images, targets
2. DataLoader 的输出是 {'image':Any,'targets':Any}
3. Transform 的输出是 {'image':Any,'targets':Any,'meta':Any}
4. BackBone 的输出是 {'image':Any,'targets':Any,'meta':Any,'outs':{'backbone':Any}}
5. Neck 的输出是 {'image':Any,'targets':Any,'meta':Any,'outs':{'neck':Any}}
6. Head 的输出是 {'image':Any,'targets':Any,'meta':Any,'outs':{'head':Any}}
7. Loss
8. Solver

"""

import os
import sys
from typing import Any, Dict, List, Optional

import torch
from PIL import ImageDraw
from torchvision.datasets import CocoDetection
from torchvision.transforms import ToPILImage, ToTensor

sys.path.append(os.getcwd())
import numpy as np
from pycocotools.mask import decode, frPyObjects as seg2mask
from ...config import _, ConfigModel
from ...optim.lr_scheduler import PiecewiseLR
from torch import nn, Tensor
from torch.optim import Adam
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import (
    _resnet_fpn_extractor,
    _validate_trainable_layers,
    resnet_fpn_backbone,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead
from torchvision.models.detection.mask_rcnn import MaskRCNNHeads, MaskRCNNPredictor
from torchvision.models.detection.roi_heads import paste_masks_in_image, RoIHeads
from torchvision.models.detection.rpn import RegionProposalNetwork, RPNHead
from torchvision.models.detection.transform import (
    GeneralizedRCNNTransform,
    resize_boxes,
    resize_keypoints,
)
from torchvision.ops import MultiScaleRoIAlign


#
# data = dict(
#     dataset=(0, 1),
#     dataloader=(),
#     transform=(),
#     backbone=(),
#     neck=(),
#     head=(),
#     loss=(),
#     postprocess=(),
#     metric=()
# )
#
#
# class Body(nn.Module):
#
#     def __call__(self, data: dict):
#         assert isinstance(data, dict), "only dict data available"
#
#         return data


class _ResNet50(nn.Module):
    def __init__(
        self,
        in_channels,
        weights=None,
        progress: bool = True,
        num_classes: Optional[int] = None,
        weights_backbone: Optional[ResNet50_Weights] = ResNet50_Weights.IMAGENET1K_V1,
        trainable_backbone_layers: Optional[int] = None,
        *args,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        weights_backbone = ResNet50_Weights.verify(weights_backbone)

        is_trained = False
        trainable_backbone_layers = _validate_trainable_layers(
            is_trained, trainable_backbone_layers, 5, 3
        )
        norm_layer = nn.BatchNorm2d

        self.backbone = resnet50(
            weights=weights_backbone, progress=progress, norm_layer=norm_layer
        )
        self.backbone = _resnet_fpn_extractor(self.backbone, trainable_backbone_layers)
        self.out_channels = self.backbone.out_channels

    def __call__(self, data):
        data.update(backbone_out=self.backbone(data["images"].tensors))
        return data


class _RegionProposalNetwork(RegionProposalNetwork):
    def __init__(
        self,
        in_channels,
        anchor_sizes=((16,), (32,), (64,), (128,), (256,)),
        aspect_ratios=((0.5, 1.0, 2.0, 4.0),),
        pre_nms_top_n_train=2000,
        pre_nms_top_n_test=1000,
        post_nms_top_n_train=2000,
        post_nms_top_n_test=1000,
        nms_thresh=0.7,
        fg_iou_thresh=0.7,
        bg_iou_thresh=0.3,
        batch_size_per_image=256,
        positive_fraction=0.5,
        score_thresh=0.0,
    ):
        super().__init__(
            rpn_anchor_generator := AnchorGenerator(
                anchor_sizes, aspect_ratios * len(anchor_sizes)
            ),
            RPNHead(in_channels, rpn_anchor_generator.num_anchors_per_location()[0]),
            fg_iou_thresh,
            bg_iou_thresh,
            batch_size_per_image,
            positive_fraction,
            dict(training=pre_nms_top_n_train, testing=pre_nms_top_n_test),
            dict(training=post_nms_top_n_train, testing=post_nms_top_n_test),
            nms_thresh,
            score_thresh,
        )
        self.out_channels = in_channels

    def __call__(self, data):
        images = data["images"]
        features = data["backbone_out"]
        targets = data["targets"]
        # neck_out = self.forward(images, features, targets)
        data.update(neck_out=self.forward(images, features, targets))
        return data


class _RoIHeads(RoIHeads):
    def __init__(
        self,
        in_channels,  # neck的输出==# backbone的输出
        num_classes,
        box_roi_pool=None,
        box_head=None,
        box_predictor=None,
        box_score_thresh=0.05,
        box_nms_thresh=0.5,
        box_detections_per_img=100,
        box_fg_iou_thresh=0.5,
        box_bg_iou_thresh=0.5,
        box_batch_size_per_image=512,
        box_positive_fraction=0.25,
        bbox_reg_weights=None,
        # Mask parameters
        mask_roi_pool=None,
        mask_head=None,
        mask_predictor=None,
    ):
        if box_roi_pool is None:
            box_roi_pool = MultiScaleRoIAlign(
                featmap_names=["0", "1", "2", "3"],
                output_size=7,
                sampling_ratio=2,
            )

        representation_size = 1024

        if box_head is None:
            resolution = box_roi_pool.output_size[0]
            box_head = TwoMLPHead(in_channels * resolution**2, representation_size)

        if box_predictor is None:
            box_predictor = FastRCNNPredictor(representation_size, num_classes)
        # out_channels = backbone.out_channels

        if mask_roi_pool is None:
            mask_roi_pool = MultiScaleRoIAlign(
                featmap_names=["0", "1", "2", "3"],
                output_size=14,
                sampling_ratio=2,
            )

        if mask_head is None:
            mask_layers = (256, 256, 256, 256)
            mask_dilation = 1
            mask_head = MaskRCNNHeads(in_channels, mask_layers, mask_dilation)

        if mask_predictor is None:
            mask_predictor_in_channels = 256  # == mask_layers[-1]
            mask_dim_reduced = 256
            mask_predictor = MaskRCNNPredictor(
                mask_predictor_in_channels, mask_dim_reduced, num_classes
            )

        super().__init__(
            # Box
            box_roi_pool,
            box_head,
            box_predictor,
            box_fg_iou_thresh,
            box_bg_iou_thresh,
            box_batch_size_per_image,
            box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh,
            box_nms_thresh,
            box_detections_per_img,
            # Mask parameters
            mask_roi_pool,
            mask_head,
            mask_predictor,
        )

    def __call__(self, data):
        features = data["backbone_out"]
        proposals = data["neck_out"][0]
        image_shapes = data["images"].image_sizes
        targets = data["targets"]
        data.update(head_out=self.forward(features, proposals, image_shapes, targets))
        return data


class MaskRCNNLoss(nn.Module):
    def forward(self, predict, target=None):
        neck_loss = predict["neck_out"][1]
        head_loss = predict["head_out"][1]
        return dict(
            loss=head_loss["loss_classifier"]
            + head_loss["loss_box_reg"]
            + head_loss["loss_mask"]
            + neck_loss["loss_objectness"]
            + neck_loss["loss_rpn_box_reg"]
        )


class _GeneralizedRCNNTransform(GeneralizedRCNNTransform):
    def __call__(self, data):
        if len(data) == 2:
            images, targets = data
        else:
            images, targets = data, None
        xx = self.forward(images, targets)
        return {"images": xx[0], "targets": xx[1]}


class MaskRCNNPostprocessor:
    """经过model之后的后处理"""

    def __init__(self, size):
        self.original_image_sizes = size

    def __call__(
        self,
        result: List[Dict[str, Tensor]],
    ) -> List[Dict[str, Tensor]]:
        o_im_s = self.original_image_sizes
        for i, (pred, im_s) in enumerate(
            zip(result["head_out"][0], result["images"].image_sizes)
        ):
            boxes = pred["boxes"]
            boxes = resize_boxes(boxes, im_s, o_im_s)
            result["head_out"][0][i]["boxes"] = boxes
            if "masks" in pred:
                masks = pred["masks"]
                masks = paste_masks_in_image(masks, boxes, o_im_s)
                result["head_out"][0][i]["masks"] = masks
            if "keypoints" in pred:
                keypoints = pred["keypoints"]
                keypoints = resize_keypoints(keypoints, im_s, o_im_s)
                result["head_out"][0][i]["keypoints"] = keypoints
        return result


class SegMetric:
    def __init__(self, main_indicator="iou"):
        self.main_indicator = main_indicator
        self.reset()

    def __call__(self, preds, batch, **kwargs):
        """
        batch: a list produced by dataloaders.
            images: torch.Tensor of shape (N, C, H, W).
            masks: torch.Tensor of shape (N, 1, H, W), ground truth masks.
        preds: a list of dict produced by post process.
            masks: torch.Tensor of shape (N, 1, H, W), predicted masks.
        """
        for pred, gt_masks in zip(preds["head_out"][0], batch):
            gt_masks = gt_masks["masks"]
            pred_masks = pred["masks"]
            result = self.evaluate_image(gt_masks, pred_masks)
            self.results.append(result)

    def evaluate_image(self, gt_masks, pred_masks):
        """
        Evaluate a single image.

        gt_masks: torch.Tensor of shape (1, H, W), ground truth mask.
        pred_masks: torch.Tensor of shape (1, H, W), predicted mask.

        Returns a dict containing evaluation results for the image.
        """
        gt_masks = gt_masks.squeeze().cpu().numpy()
        pred_masks = pred_masks.squeeze().cpu().numpy()

        # compute IoU
        intersection = np.logical_and(gt_masks, pred_masks).sum()
        union = np.logical_or(gt_masks, pred_masks).sum()
        iou = intersection / union

        return {"iou": iou}

    def get_metric(self):
        """
        Return metrics as a dictionary:
        {
            'iou': float
        }
        """
        metrics = {
            result_type: np.mean([result[result_type] for result in self.results])
            for result_type in self.results[0].keys()
        }
        self.reset()
        return metrics

    def reset(self):
        self.results = []


def coco_target_transforms(target):
    out = []
    for ann in target:
        mask = decode(seg2mask(ann["segmentation"], 120, 120))
        if len(mask.shape) == 3:
            mask = torch.tensor(mask[..., 0], dtype=torch.uint8)

        else:
            mask = torch.tensor(mask, dtype=torch.uint8)
        mask = mask.unsqueeze(0).unsqueeze(0)
        if mask.shape != (120, 120):
            mask = (
                torch.nn.functional.interpolate(mask.float(), size=(120, 120))
                .squeeze(0)
                .byte()
            )  # 插值并移除额外的维度
        bbox = torch.tensor(
            [
                ann["bbox"][0] / 120,
                ann["bbox"][1] / 120,
                (ann["bbox"][0] + ann["bbox"][2]) / 120,
                (ann["bbox"][1] + ann["bbox"][3]) / 120,
            ]
        )
        label = torch.tensor(ann["category_id"])
        out.append(dict(masks=mask, boxes=bbox, labels=label))
    return out


def collate_fn(batch):
    images = [item[0] for item in batch]
    targets = []
    for item in batch:
        target = item[1]
        masks = torch.stack([i["masks"] for i in target])
        boxes = torch.stack([i["boxes"] for i in target], 0)
        labels = torch.stack([i["labels"] for i in target], 0)
        target = dict(masks=masks, boxes=boxes, labels=labels)
        targets.append(target)
    return images, targets


class Model(ConfigModel):
    use_gpu = False
    epoch_num = 1
    log_window_size = 20
    log_batch_step = 10
    save_model_dir = "D:\dev\github\ToddleOCR\output"
    save_epoch_step = 1
    eval_batch_step = [0, 9999]
    metric_during_train = False
    # pretrained_model = None
    # checkpoints = None
    # save_infer_dir = None
    # use_visualdl = False
    distributed = False
    model_type = "cse"
    algorithm = "maskrcnn"

    # 数据集
    class Data:
        dataset = CocoDetection
        root: "d:/dev/.data/CCSE/kaiti_chinese_stroke_2021/test2021" = (
            "d:/dev/.data/CCSE/kaiti_chinese_stroke_2021/train2021"
        )
        annFile: "D:/dev/.data/CCSE/kaiti_chinese_stroke_2021/annotations/instances_test2021.json" = "D:/dev/.data/CCSE/kaiti_chinese_stroke_2021/annotations/instances_train2021.json"
        target_transform = coco_target_transforms
        transform = ToTensor()

    # 数据集评估器 / 给数据集出一个总结报告 / 可视化

    # 加载器
    class Loader:
        shuffle: False = True
        drop_last: True = True
        batch_size: 1 = 1
        num_workers: 2 = 4
        collate_fn = collate_fn

    # 输入转换器 Transform 是批处理的在数据进入 backbone 之前
    Transform = _(
        _GeneralizedRCNNTransform,
        min_size=120,
        max_size=120,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
    )
    # 主干网络
    Backbone = _(
        resnet_fpn_backbone,
        "resnet50",
        weights=ResNet50_Weights.DEFAULT,
        trainable_layers=5,
    )  #'features'
    # 颈部
    Neck = _(
        _RegionProposalNetwork,
    )
    # 头部
    Head = _(_RoIHeads, num_classes=26)

    # 输出转换器，后处理是展现推理结果
    postprocessor = MaskRCNNPostprocessor(size=(120, 120))
    metric = SegMetric(main_indicator="iou")

    # 损失优化属于训练
    loss = MaskRCNNLoss()
    Optimizer = _(Adam, betas=[0.9, 0.999], lr=0.02)
    LRScheduler = _(
        PiecewiseLR, decay_epochs=[70], values=[0.001, 0.0001], warmup_epoch=5
    )


if __name__ == "__main__":
    m = Model("D:\dev\github\ToddleOCR\output\latest.pth")
    # try:
    #     m.train()
    # except Exception as e:
    #     logger.exception(e)
    #     m.save('output/ccse.pth', prefix='exception')

    m.model.eval()
    n = m._build_dataloader("train")
    for i in n:
        print(i)
        oi = ToPILImage(mode="RGB")(i[0][0])
        draw = ImageDraw.Draw(oi)
        # i[1]=None
        x = m.model(i[0])
        d = m.postprocessor(x)
        print(x)

        for box in d["head_out"][0][0]["boxes"]:
            print(box)
            draw.rectangle(
                (
                    int(box[0] * 120),
                    int(box[1] * 120),
                    int(box[2] * 120),
                    int(box[3] * 120),
                ),
                outline="green",
            )

        oi.show()
        break
