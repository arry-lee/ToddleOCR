import os
import sys

sys.path.append(os.getcwd())
from toddleocr.config import _, ConfigModel
from toddleocr.datasets.simple import SimpleDataSet
from toddleocr.loss.db import DBLoss
from toddleocr.metrics.det import DetMetric
from toddleocr.modules.backbones.resnet.det_resnet_vd import ResNet_vd
from toddleocr.modules.heads.db import DBHead
from toddleocr.modules.necks.db_fpn import DBFPN
from toddleocr.postprocess.db import DBPostProcess
from toddleocr.transforms import (
    DecodeImage,
    DetLabelEncode,
    DetResizeForTest,
    EastRandomCropData,
    IaaAugment,
    KeepKeys,
    MakeBorderMap,
    MakeShrinkMap,
    NormalizeImage,
    ToCHWImage,
)
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


class Model(ConfigModel):
    use_gpu = True
    epoch_num = 1200
    log_window_size = 20
    log_batch_step = 2
    save_epoch_step = 1200
    eval_batch_step = [3000, 2000]
    metric_during_train = False
    pretrained_model = "../model/ch_ppocr_server_v2.0_det_train/best_accuracy.pth"
    checkpoints = None
    model_type = "det"
    algorithm = "DB"
    Backbone = _(ResNet_vd, layers=18)
    Neck = _(DBFPN, out_channels=256)
    Head = _(DBHead, k=50)
    loss = DBLoss(
        balance_loss=True,
        main_loss_type="DiceLoss",
        alpha=5,
        beta=10,
        ohem_ratio=3,
    )
    metric = DetMetric(main_indicator="hmean")
    postprocessor = DBPostProcess(
        thresh=0.3, box_thresh=0.6, max_candidates=1000, unclip_ratio=2
    )
    Optimizer = _(Adam, betas=[0.9, 0.999], lr=0.001)
    LRScheduler = _(CosineAnnealingWarmRestarts, T_0=2)

    class Data:
        dataset = SimpleDataSet
        root = "./train_data/icdar2015/text_localization/"
        label_files: ["test_icdar2015_label.txt"] = ["train_icdar2015_label.txt"]
        ratio_list = [1.0]

    class Loader:
        shuffle: 0 = True
        drop_last: 0 = False
        batch_size: 1 = 8
        num_workers: 2 = 4

    Transforms = _[
        DecodeImage(img_mode="BGR", channel_first=False),
        DetLabelEncode() : ...,
        IaaAugment(
            augmenter_args=[
                {"type": "Fliplr", "args": {"p": 0.5}},
                {"type": "Affine", "args": {"rotate": [-10, 10]}},
                {"type": "Resize", "args": {"size": [0.5, 3]}},
            ]
        ) :,
        EastRandomCropData(size=[960, 960], max_tries=50, keep_ratio=True) :,
        MakeBorderMap(shrink_ratio=0.4, thresh_min=0.3, thresh_max=0.7) :,
        MakeShrinkMap(shrink_ratio=0.4, min_text_size=8) :,
        : DetResizeForTest() : ...,
        NormalizeImage(
            scale="1./255.",
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            order="hwc",
        ),
        ToCHWImage(),
        KeepKeys(
            "image",
            "threshold_map",
            "threshold_mask",
            "shrink_map",
            "shrink_mask",
        ) : KeepKeys("image", "shape", "polys", "ignore_tags") : KeepKeys(
            "image", "shape"
        ),
    ]


if __name__ == "__main__":
    m = Model()
    m.infer(
        "D:\\dev\\github\\ToddleOCR\\doc\\imgs\\11.jpg",
        "D:\\dev\\.model\\paddocr\\ch_ppocr_server_v2.0_det_train\\best_accuracy.pth",
    )
