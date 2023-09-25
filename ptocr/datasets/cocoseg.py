from typing import List, Any

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision.models.detection.mask_rcnn import MaskRCNN
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.transforms import ToTensor

# 1 看看 MASKRCNN 训练里面用到的键和值的格式
#  target["boxes"] torch.Tensor (n,4)
# "labels","masks"

# d = DataLoader()
# c = CocoDetection()
# def transform()
from pycocotools.mask import frPyObjects as seg2mask, decode


class CocoSegment(CocoDetection):
    """CoCo 分割数据集的重新实现"""

    def __init__(
            self,
            root: str,
            label_file: str,
            transforms = None
    ) -> None:
        super().__init__(root, label_file,transforms=transforms)

    def _load_target(self, id: int):
        anns = self.coco.loadAnns(self.coco.getAnnIds(id))
        out = []
        for ann in anns:
            mask = decode(seg2mask(ann['segmentation'], 120, 120))

            if len(mask.shape)==3:
                mask = torch.tensor(mask, dtype=torch.uint8).permute(2, 0, 1)
            else:
                mask = torch.tensor(mask, dtype=torch.uint8).unsqueeze(0)
            print(mask.size())
            bbox = torch.tensor(ann['bbox'])
            label = torch.tensor(ann['category_id'])
            out.append(dict(masks=mask, boxes=bbox, labels=label))
        return out

    # def __getitem__(self, index: int):
    #     id = self.ids[index]
    #     image = self._load_image(id)
    #     target = self._load_target(id)
    #
    #     if self.transforms is not None:
    #         image, target = self.transforms(image, target)
    #
    #     return dict(images=image, targets=target)

# c = CocoSegment("d:/dev/.data/CCSE/kaiti_chinese_stroke_2021/test2021",
#     "D:/dev/.data/CCSE/kaiti_chinese_stroke_2021/annotations/instances_test2021.json",
#     transform=ToTensor())
#
# print(c)
# tr = GeneralizedRCNNTransform(120, 120, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# for images, targets in DataLoader(c, batch_size=1):
#     # print(images, targets)
#     # seg = targets[0]['masks']
#     # # mask = seg2mask(seg,120,120)
#     #
#     # # print(mask)
#     # print(targets[0].keys())
#     # print(targets[0]['boxes'])
#     print(tr(images, targets))
#     break
