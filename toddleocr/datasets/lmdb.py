import os
import string

import cv2
import lmdb
import numpy as np
import six
from PIL import Image
from torchvision.datasets.vision import VisionDataset


class LMDBDataSet(VisionDataset):
    def __init__(self, root, transforms=None, **kwargs):
        super().__init__(root, transforms)

        self.lmdb_sets = self.load_hierarchical_lmdb_dataset()
        self.data_idx_order_list = self.dataset_traversal()

        ratio_list = kwargs.get("ratio_list", [1.0])
        self.seed = kwargs.get("seed", None)
        if self.seed is not None:
            np.random.seed(self.seed)
        self.mode = kwargs.get("mode", "train")

        self.need_reset = True in [x < 1 for x in ratio_list]

    def load_hierarchical_lmdb_dataset(self):
        lmdb_sets = {}
        dataset_idx = 0
        for dirpath, dirnames, filenames in os.walk(self.root):
            if not dirnames:
                env = lmdb.open(
                    dirpath,
                    max_readers=32,
                    readonly=True,
                    lock=False,
                    readahead=False,
                    meminit=False,
                )
                txn = env.begin(write=False)
                num_samples = int(txn.get("num-samples".encode()))
                lmdb_sets[dataset_idx] = {
                    "dirpath": dirpath,
                    "env": env,
                    "txn": txn,
                    "num_samples": num_samples,
                }
                dataset_idx += 1
        return lmdb_sets

    def dataset_traversal(self):
        lmdb_num = len(self.lmdb_sets)
        total_sample_num = 0
        for lno in range(lmdb_num):
            total_sample_num += self.lmdb_sets[lno]["num_samples"]
        data_idx_order_list = np.zeros((total_sample_num, 2))
        beg_idx = 0
        for lno in range(lmdb_num):
            tmp_sample_num = self.lmdb_sets[lno]["num_samples"]
            end_idx = beg_idx + tmp_sample_num
            data_idx_order_list[beg_idx:end_idx, 0] = lno
            data_idx_order_list[beg_idx:end_idx, 1] = list(range(tmp_sample_num))
            data_idx_order_list[beg_idx:end_idx, 1] += 1
            beg_idx = beg_idx + tmp_sample_num
        return data_idx_order_list

    def get_img_data(self, value):
        """get_img_data"""
        if not value:
            return None
        imgdata = np.frombuffer(value, dtype="uint8")
        if imgdata is None:
            return None
        imgori = cv2.imdecode(imgdata, 1)
        if imgori is None:
            return None
        return imgori

    def get_lmdb_sample_info(self, txn, index):
        label_key = "label-%09d".encode() % index
        label = txn.get(label_key)
        if label is None:
            return None
        label = label.decode("utf-8")
        img_key = "image-%09d".encode() % index
        imgbuf = txn.get(img_key)
        return imgbuf, label

    def __getitem__(self, idx):
        lmdb_idx, file_idx = self.data_idx_order_list[idx]
        lmdb_idx = int(lmdb_idx)
        file_idx = int(file_idx)
        sample_info = self.get_lmdb_sample_info(
            self.lmdb_sets[lmdb_idx]["txn"], file_idx
        )
        if sample_info is None:
            return self.__getitem__(np.random.randint(self.__len__()))
        img, label = sample_info
        data = {"image": img, "label": label}
        data = self.transforms(data)
        return data

    def __len__(self):
        return self.data_idx_order_list.shape[0]


class LMDBDataSetSR(LMDBDataSet):
    def buf2PIL(self, txn, key, type="RGB"):
        imgbuf = txn.get(key)
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        im = Image.open(buf).convert(type)
        return im

    def str_filt(self, str_, voc_type):
        alpha_dict = {
            "digit": string.digits,
            "lower": string.digits + string.ascii_lowercase,
            "upper": string.digits + string.ascii_letters,
            "all": string.digits + string.ascii_letters + string.punctuation,
        }
        if voc_type == "lower":
            str_ = str_.lower()
        for char in str_:
            if char not in alpha_dict[voc_type]:
                str_ = str_.replace(char, "")
        return str_

    def get_lmdb_sample_info(self, txn, index):
        self.voc_type = "upper"
        self.max_len = 100
        self.test = False
        label_key = b"label-%09d" % index
        word = str(txn.get(label_key).decode())
        img_HR_key = b"image_hr-%09d" % index  # 128*32
        img_lr_key = b"image_lr-%09d" % index  # 64*16
        try:
            img_HR = self.buf2PIL(txn, img_HR_key, "RGB")
            img_lr = self.buf2PIL(txn, img_lr_key, "RGB")
        except IOError or len(word) > self.max_len:
            return self[index + 1]
        label_str = self.str_filt(word, self.voc_type)
        return img_HR, img_lr, label_str

    def __getitem__(self, idx):
        lmdb_idx, file_idx = self.data_idx_order_list[idx]
        lmdb_idx = int(lmdb_idx)
        file_idx = int(file_idx)
        sample_info = self.get_lmdb_sample_info(
            self.lmdb_sets[lmdb_idx]["txn"], file_idx
        )
        if sample_info is None:
            return self.__getitem__(np.random.randint(self.__len__()))
        img_HR, img_lr, label_str = sample_info
        data = {"image_hr": img_HR, "image_lr": img_lr, "label": label_str}
        outs = self.transforms(data)
        if outs is None:
            return self.__getitem__(np.random.randint(self.__len__()))
        return outs
