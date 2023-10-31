import os
import random

import numpy as np
from torchvision.datasets import VisionDataset


class PGDataSet(VisionDataset):
    def __init__(self, root, transforms=None, **kwargs):
        super().__init__(root, transforms)
        self.delimiter = kwargs.get("delimiter", "\t")
        label_files = kwargs.pop("label_files")
        data_source_num = len(label_files)
        ratio_list = kwargs.get("ratio_list", [1.0])
        self.seed = kwargs.get("seed", None)

        self.mode = kwargs.get("mode", "train")
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
        if isinstance(ratio_list, (float, int)):
            ratio_list = [float(ratio_list)] * int(data_source_num)
        assert (
            len(ratio_list) == data_source_num
        ), "The length of ratio_list should be the same as the file_list."

        self.data_lines = self.get_image_info_list(label_files, ratio_list)
        self.data_idx_order_list = list(range(len(self.data_lines)))
        self.need_reset = True in [x < 1 for x in ratio_list]

    def get_image_info_list(self, file_list, ratio_list):
        if isinstance(file_list, str):
            file_list = [file_list]
        data_lines = []
        for idx, file in enumerate(file_list):
            with open(file, "rb") as f:
                lines = f.readlines()
                if ratio_list[idx] < 1.0:
                    lines = random.sample(lines, round(len(lines) * ratio_list[idx]))
                data_lines.extend(lines)
        return data_lines

    def __getitem__(self, idx):
        file_idx = self.data_idx_order_list[idx]
        data_line = self.data_lines[file_idx]
        try:
            data_line = data_line.decode("utf-8")
            substr = data_line.strip("\n").split(self.delimiter)
            file_name = substr[0]
            label = substr[1]
            img_path = os.path.join(self.root, file_name)
            data = {"img_path": img_path, "label": label}
            if not os.path.exists(img_path):
                raise Exception("{} does not exist!".format(img_path))
            with open(data["img_path"], "rb") as f:
                img = f.read()
                data["image"] = img
            outs = self.transforms(data)
        except Exception as e:
            print(
                "When parsing line {}, error happened with msg: {}".format(
                    self.data_idx_order_list[idx], e
                )
            )
            outs = None
        if outs is None:
            return self.__getitem__(np.random.randint(self.__len__()))
        return outs

    def __len__(self):
        return len(self.data_idx_order_list)
