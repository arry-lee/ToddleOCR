import torch

from pycocotools.mask import decode, frPyObjects as seg2mask
from torchvision.datasets import CocoDetection


class CocoSegment(CocoDetection):
    """CoCo 分割数据集的重新实现"""

    def __init__(self, root: str, label_file: str, transforms=None) -> None:
        super().__init__(root, label_file, transforms=transforms)

    def _load_target(self, id: int):
        anns = self.coco.loadAnns(self.coco.getAnnIds(id))
        out = []
        for ann in anns:
            mask = decode(seg2mask(ann["segmentation"], 120, 120))

            if len(mask.shape) == 3:
                mask = torch.tensor(mask, dtype=torch.uint8).permute(2, 0, 1)
            else:
                mask = torch.tensor(mask, dtype=torch.uint8).unsqueeze(0)
            print(mask.size())
            bbox = torch.tensor(ann["bbox"])
            label = torch.tensor(ann["category_id"])
            out.append(dict(masks=mask, boxes=bbox, labels=label))
        return out
