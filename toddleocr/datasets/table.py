import os
import re

import cv2
import numpy as np
from torch.utils.data import Dataset


class FolderDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.root_dir = root
        self.transforms = transforms

        # filter .jpg image
        self.image_list = os.listdir(root)
        self.image_list = list(filter(lambda x: x.endswith(".jpg"), os.listdir(root)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        img_path = os.path.join(self.root_dir, image_name)
        label_path = img_path.replace(".jpg", ".txt")

        data = {}
        data["image"] = cv2.imread(img_path)
        with open(label_path, "r", encoding="utf-8") as f:
            label_file = f.read()
        temp_list = label_file.splitlines(keepends=False)
        pat = re.compile(
            r"(?P<name>.+);(?P<x1>\d+);(?P<y1>\d+);"
            r"(?P<x2>\d+);(?P<y2>\d+);(?P<x3>\d+);"
            r"(?P<y3>\d+);(?P<x4>\d+);(?P<y4>\d+);"
            r"(?P<label>.+)@(?P<content>.*)"
        )
        boxes, txts, txt_tags = [], [], []
        for line in temp_list:
            matched = pat.match(line)
            if matched:
                # name = matched["name"]
                box = [
                    [int(matched["x1"]), int(matched["y1"])],
                    [int(matched["x2"]), int(matched["y2"])],
                    [int(matched["x3"]), int(matched["y3"])],
                    [int(matched["x4"]), int(matched["y4"])],
                ]
                label = matched["label"]
                content = matched["content"]
                if label == "text":
                    txt_tags.append(False)
                    boxes.append(box)
                    txts.append(content)

        boxes = np.array(boxes, dtype=np.float32)
        txt_tags = np.array(txt_tags, dtype=bool)

        data["polys"] = boxes
        data["texts"] = txts
        data["ignore_tags"] = txt_tags
        if self.transforms:
            data = self.transforms(data)
        return data
