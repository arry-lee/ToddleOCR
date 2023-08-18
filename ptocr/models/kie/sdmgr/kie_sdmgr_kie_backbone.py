# This .py is auto generated by the script in the root folder.
from ptocr.config import ConfigModel, _
from ptocr.modules.backbones.kie_unet_sdmgr import Kie_backbone
from ptocr.modules.heads.sdmgr import SDMGRHead
from ptocr.loss.sdmgr import SDMGRLoss
from ptocr.metrics.kie import KIEMetric
from torch.optim import Adam
from ptocr.optim.lr_scheduler import PiecewiseLR
from ptocr.datasets.simple import SimpleDataSet
from ptocr.transforms.operators import (
    ToCHWImage,
    DecodeImage,
    KieResize,
    NormalizeImage,
    KeepKeys,
)
from ptocr.transforms.label_ops import KieLabelEncode


class Model(ConfigModel):
    use_gpu = True
    epoch_num = 60
    log_window_size = 20
    log_batch_step = 50
    save_model_dir = "./output/kie_5/"
    save_epoch_step = 50
    eval_batch_step = [0, 80]
    load_static_weights = False
    metric_during_train = False
    pretrained_model = None
    checkpoints = None
    save_infer_dir = None
    use_visualdl = False
    class_path = "./train_data/wildreceipt/class_list.txt"
    infer_img = "./train_data/wildreceipt/1.txt"
    save_res_path = "./output/sdmgr_kie/predicts_kie.txt"
    img_scale = [1024, 512]
    model_type = "kie"
    algorithm = "SDMGR"
    Transform = None
    Backbone = _(
        Kie_backbone,
    )
    Head = _(
        SDMGRHead,
    )
    loss = SDMGRLoss()
    metric = KIEMetric(main_indicator="hmean")
    Optimizer = _(Adam, betas=[0.9, 0.999], lr=0.001)
    LRScheduler = _(
        PiecewiseLR,
        decay_epochs=[60, 80, 100],
        values=[0.001, 0.0001, 1e-05],
        warmup_epoch=2,
    )
    PostProcessor = None

    class Train:
        Dataset = _(
            SimpleDataSet,
            data_dir="./train_data/wildreceipt/",
            label_file_list=["./train_data/wildreceipt/wildreceipt_train.txt"],
            ratio_list=[1.0],
        )
        transforms = _[
            DecodeImage(img_mode="RGB", channel_first=False),
            NormalizeImage(
                scale=1,
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                order="hwc",
            ),
            KieLabelEncode(
                character_dict_path="./train_data/wildreceipt/dict.txt",
                class_path="./train_data/wildreceipt/class_list.txt",
            ),
            KieResize(),
            ToCHWImage(),
            KeepKeys(
                keep_keys=[
                    "image",
                    "relations",
                    "texts",
                    "points",
                    "labels",
                    "tag",
                    "shape",
                ]
            ),
        ]
        DATALOADER = _(
            shuffle=True, drop_last=False, batch_size=4, num_workers=4
        )

    class Eval:
        Dataset = _(
            SimpleDataSet,
            data_dir="./train_data/wildreceipt",
            label_file_list=["./train_data/wildreceipt/wildreceipt_test.txt"],
        )
        transforms = _[
            DecodeImage(img_mode="RGB", channel_first=False),
            KieLabelEncode(
                character_dict_path="./train_data/wildreceipt/dict.txt"
            ),
            KieResize(),
            NormalizeImage(
                scale=1,
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                order="hwc",
            ),
            ToCHWImage(),
            KeepKeys(
                keep_keys=[
                    "image",
                    "relations",
                    "texts",
                    "points",
                    "labels",
                    "tag",
                    "ori_image",
                    "ori_boxes",
                    "shape",
                ]
            ),
        ]
        DATALOADER = _(
            shuffle=False, drop_last=False, batch_size=1, num_workers=4
        )
