import json
import os
import random
import traceback

import numpy as np
from torchvision.datasets import VisionDataset


class SimpleDataSet(VisionDataset):
    def __init__(self, root, transforms=None, **dataset_config):
        super().__init__(root, transforms)
        # self.logger = logger
        self.mode = dataset_config.get("mode", "train").lower()
        self.delimiter = dataset_config.get("delimiter", "\t")
        label_files = dataset_config.pop("label_files")
        data_source_num = len(label_files)
        ratio_list = dataset_config.get("ratio_list", 1.0)
        if isinstance(ratio_list, (float, int)):
            ratio_list = [float(ratio_list)] * int(data_source_num)

        assert (
            len(ratio_list) == data_source_num
        ), "The length of ratio_list should be the same as the file_list."

        self.seed = dataset_config.get("seed", None)
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

        self.data_lines = self.get_image_info_list(label_files, ratio_list)
        self.data_idx_order_list = list(range(len(self.data_lines)))
        self.need_reset = True in [x < 1 for x in ratio_list]

    def get_image_info_list(self, file_list, ratio_list):
        if isinstance(file_list, str):
            file_list = [file_list]
        data_lines = []
        for idx, file in enumerate(file_list):
            file = os.path.join(self.root, file)
            with open(file, "rb") as f:
                lines = f.readlines()
                if ratio_list[idx] < 1.0:
                    lines = random.sample(lines, round(len(lines) * ratio_list[idx]))
                data_lines.extend(lines)
        return data_lines

    def _try_parse_filename_list(self, file_name):
        # multiple images -> one gt label
        if len(file_name) > 0 and file_name[0] == "[":
            try:
                info = json.loads(file_name)
                file_name = random.choice(info)
            except:
                pass
        return file_name

    def __getitem__(self, idx):
        file_idx = self.data_idx_order_list[idx]
        data_line = self.data_lines[file_idx]
        try:
            data_line = data_line.decode("utf-8")
            substr = data_line.strip("\n").split(self.delimiter)
            file_name = substr[0]
            file_name = self._try_parse_filename_list(file_name)
            label = substr[1]
            img_path = os.path.join(self.root, file_name)
            data = {"img_path": img_path, "label": label}
            if not os.path.exists(img_path):
                raise Exception("{} does not exist!".format(img_path))
            with open(data["img_path"], "rb") as f:
                img = f.read()
                data["image"] = img
            outs = self.transforms(data)
        except:
            print(
                "When parsing line {}, error happened with msg: {}".format(
                    data_line, traceback.format_exc()
                )
            )
            outs = None
        if outs is None:
            # during evaluation, we should fix the idx to get same results for many times of evaluation.
            rnd_idx = (
                np.random.randint(self.__len__())
                if self.mode == "train"
                else (idx + 1) % self.__len__()
            )
            return self.__getitem__(rnd_idx)
        return outs

    def __len__(self):
        return len(self.data_idx_order_list)
