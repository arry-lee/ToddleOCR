import json
import os
import random

import numpy as np
from torchvision.datasets import VisionDataset


class PubTabDataSet(VisionDataset):
    def __init__(self, root, transforms=None, **kwargs):
        super().__init__(root, transforms)

        label_files = kwargs.pop("label_files")
        data_source_num = len(label_files)

        ratio_list = kwargs.get("ratio_list", [1.0])
        self.seed = kwargs.get("seed", None)

        self.mode = kwargs.get("mode", "train")

        if self.seed is not None:
            random.seed(self.seed)

        if isinstance(ratio_list, (float, int)):
            ratio_list = [float(ratio_list)] * int(data_source_num)

        assert (
            len(ratio_list) == data_source_num
        ), "The length of ratio_list should be the same as the file_list."

        self.data_lines = self.get_image_info_list(label_files, ratio_list)

        # self.need_reset = True in [x < 1 for x in ratio_list]

    def get_image_info_list(self, file_list, ratio_list=None):
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

    def check(self, max_text_length):
        data_lines = []
        for line in self.data_lines:
            data_line = line.decode("utf-8").strip("\n")
            info = json.loads(data_line)
            file_name = info["filename"]
            cells = info["html"]["cells"].copy()
            structure = info["html"]["structure"]["tokens"].copy()

            img_path = os.path.join(self.root, file_name)
            if not os.path.exists(img_path):
                print("{} does not exist!".format(img_path))
                continue
            if len(structure) == 0 or len(structure) > max_text_length:
                continue
            # data = {'img_path': img_path, 'cells': cells, 'structure':structure,'file_name':file_name}
            data_lines.append(line)
        self.data_lines = data_lines

    def __getitem__(self, idx):
        data_line = self.data_lines[idx]
        data_line = data_line.decode("utf-8").strip("\n")
        info = json.loads(data_line)
        file_name = info["filename"]
        cells = info["html"]["cells"].copy()
        structure = info["html"]["structure"]["tokens"].copy()

        img_path = os.path.join(self.root, file_name)
        if not os.path.exists(img_path):
            raise Exception("{} does not exist!".format(img_path))
        data = {
            "img_path": img_path,
            "cells": cells,
            "structure": structure,
            "file_name": file_name,
        }

        with open(data["img_path"], "rb") as f:
            img = f.read()
            data["image"] = img
        outs = self.transforms(data)
        return outs

    def __len__(self):
        return len(self.data_lines)
